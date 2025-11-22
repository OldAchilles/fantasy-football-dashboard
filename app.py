import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from espn_api.football import League
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta

# Add your ESPN credentials at the top
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except:
        return default

ESPN_S2 = get_secret("ESPN_S2")
SWID = get_secret("SWID")
LEAGUE_ID = get_secret("LEAGUE_ID")
ODDS_API_KEY = get_secret("ODDS_API_KEY")

# Cache file path (v2 = includes duplicate week detection fix)
CACHE_FILE = "h2h_cache_v2.pkl"
CACHE_DURATION_DAYS = 7  # Cache for 7 days since data only changes weekly


# Manual mapping for name variations/duplicates
NAME_MAPPING = {
    'Russell': 'Russ',
    'Russ': 'Russ',
    'russell': 'Russ',
    'russ': 'Russ',
    'Brian': 'Brian',      # Map both to just 'Brian'
    'brian': 'Brian',      # Lowercase version
    'Brian1': 'Brian',     # Your mapped version
    'brian1': 'Brian',     # Just in case
}

def normalize_name(first_name):
    """Apply manual name mapping to handle duplicates/variations"""
    # Strip whitespace and convert to title case for consistency
    first_name = first_name.strip()

    # Check mapping (case-sensitive first, then case-insensitive)
    if first_name in NAME_MAPPING:
        return NAME_MAPPING[first_name]

    # Try title case version
    title_name = first_name.title()
    if title_name in NAME_MAPPING:
        return NAME_MAPPING[title_name]

    # Try lowercase version
    lower_name = first_name.lower()
    if lower_name in NAME_MAPPING:
        return NAME_MAPPING[lower_name]

    # Return title case version as default
    return title_name

def get_head_to_head_data(league_id, years, espn_s2, swid):
    """Build head-to-head record matrix across all years"""

    matchup_records = {}
    owner_first_name_map = {}

    for year in years:
        try:
            league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

            # Build firstName mapping for this year
            for team in league.teams:
                if hasattr(team, 'owner'):
                    owner_id = team.owner
                else:
                    owner_id = team.team_name

                first_name = None

                if hasattr(team, 'owners') and team.owners and len(team.owners) > 0:
                    owner_obj = team.owners[0]

                    if isinstance(owner_obj, dict) and 'firstName' in owner_obj:
                        first_name = owner_obj['firstName']
                    elif hasattr(owner_obj, 'firstName'):
                        first_name = owner_obj.firstName
                    else:
                        owner_str = str(owner_obj)
                        first_name = owner_str.split()[0] if ' ' in owner_str else owner_str

                if not first_name and hasattr(team, 'owner') and team.owner:
                    first_name = team.owner.split()[0] if ' ' in team.owner else team.owner

                if not first_name:
                    first_name = team.team_name

                first_name = normalize_name(first_name)
                owner_first_name_map[owner_id] = first_name

            # For 2019+, use box_scores
            if year >= 2019:

                week = 1
                max_week = 20
                previous_week_signature = None  # Track previous week to detect duplicates

                while week <= max_week:
                    try:
                        box_scores = league.box_scores(week=week)

                        if not box_scores:
                            break

                        # Check for all 0-0 scores (future weeks)
                        all_zero_scores = all(m.home_score == 0 and m.away_score == 0 for m in box_scores)
                        if all_zero_scores:
                            break

                        # Create a signature of this week's matchups (teams + scores)
                        # If it matches the previous week exactly, ESPN is repeating data - stop
                        current_week_signature = set()
                        for m in box_scores:
                            home_id = m.home_team.team_id if hasattr(m.home_team, 'team_id') else str(m.home_team.team_name)
                            away_id = m.away_team.team_id if hasattr(m.away_team, 'team_id') else str(m.away_team.team_name)
                            # Include scores in signature to detect exact duplicates
                            sig = (tuple(sorted([home_id, away_id])), m.home_score, m.away_score)
                            current_week_signature.add(sig)

                        # If this week's signature exactly matches the previous week, we're seeing duplicates
                        if previous_week_signature is not None and current_week_signature == previous_week_signature:
                            break

                        previous_week_signature = current_week_signature

                        for matchup in box_scores:
                            home_owner = matchup.home_team.owner if hasattr(matchup.home_team, 'owner') else matchup.home_team.team_name
                            away_owner = matchup.away_team.owner if hasattr(matchup.away_team, 'owner') else matchup.away_team.team_name

                            home_first = normalize_name(owner_first_name_map.get(home_owner, home_owner))
                            away_first = normalize_name(owner_first_name_map.get(away_owner, away_owner))

                            home_score = matchup.home_score
                            away_score = matchup.away_score

                            # Skip self-matchups (same owner, different team IDs)
                            if home_first == away_first:
                                continue

                            # Update BOTH perspectives consistently
                            # Update home team's record vs away team
                            key_home = (home_first, away_first)
                            if key_home not in matchup_records:
                                matchup_records[key_home] = {
                                    'wins': 0, 'losses': 0,
                                    'points_for': 0, 'points_against': 0,
                                    'games': 0
                                }

                            matchup_records[key_home]['games'] += 1
                            matchup_records[key_home]['points_for'] += home_score
                            matchup_records[key_home]['points_against'] += away_score

                            if home_score > away_score:
                                matchup_records[key_home]['wins'] += 1
                            else:
                                matchup_records[key_home]['losses'] += 1

                            # Update away team's record vs home team
                            key_away = (away_first, home_first)
                            if key_away not in matchup_records:
                                matchup_records[key_away] = {
                                    'wins': 0, 'losses': 0,
                                    'points_for': 0, 'points_against': 0,
                                    'games': 0
                                }

                            matchup_records[key_away]['games'] += 1
                            matchup_records[key_away]['points_for'] += away_score
                            matchup_records[key_away]['points_against'] += home_score

                            if away_score > home_score:
                                matchup_records[key_away]['wins'] += 1
                            else:
                                matchup_records[key_away]['losses'] += 1

                        week += 1

                    except Exception as e:
                        break

            # For pre-2019, use team schedules
            else:
                processed_matchups = set()

                for team in league.teams:
                    owner = team.owner if hasattr(team, 'owner') else team.team_name
                    owner_first = normalize_name(owner_first_name_map.get(owner, owner))
                    team_id = team.team_id if hasattr(team, 'team_id') else str(team.team_name)

                    if not hasattr(team, 'schedule') or not team.schedule:
                        continue

                    if not hasattr(team, 'outcomes') or not team.outcomes:
                        continue

                    total_weeks = max(len(team.schedule), len(team.outcomes))

                    for week_idx in range(total_weeks):
                        opponent_team = team.schedule[week_idx] if week_idx < len(team.schedule) else None

                        if opponent_team is None:
                            continue

                        opponent_owner = opponent_team.owner if hasattr(opponent_team, 'owner') else opponent_team.team_name
                        opponent_first = normalize_name(owner_first_name_map.get(opponent_owner, opponent_owner))
                        opponent_team_id = opponent_team.team_id if hasattr(opponent_team, 'team_id') else str(opponent_team.team_name)

                        # Skip self-matchups (same owner, different team IDs)
                        if owner_first == opponent_first:
                            continue

                        # WORKAROUND: Skip known bad schedule data
                        # In 2016, Jesse's team_id=1 and Russ's data has corrupt schedules
                        if year == 2016 and (team_id == 1 or owner_first == 'Russ'):
                            continue

                        # Use smaller team_id first to ensure consistent ordering
                        # Include week_idx from the lower team_id to maintain consistency
                        if team_id < opponent_team_id:
                            matchup_id = (team_id, opponent_team_id, year, week_idx)
                        else:
                            # Skip - we'll process this matchup when we get to the other team
                            continue

                        if matchup_id in processed_matchups:
                            continue

                        processed_matchups.add(matchup_id)

                        if week_idx < len(team.outcomes):
                            outcome = team.outcomes[week_idx]

                            if outcome == 'W':
                                winner = owner_first
                                loser = opponent_first
                            elif outcome == 'L':
                                winner = opponent_first
                                loser = owner_first
                            else:
                                continue

                            # Update BOTH perspectives consistently
                            # Winner's perspective
                            key_winner = (winner, loser)
                            if key_winner not in matchup_records:
                                matchup_records[key_winner] = {
                                    'wins': 0, 'losses': 0,
                                    'points_for': 0, 'points_against': 0,
                                    'games': 0
                                }

                            matchup_records[key_winner]['games'] += 1
                            matchup_records[key_winner]['wins'] += 1

                            # Loser's perspective
                            key_loser = (loser, winner)
                            if key_loser not in matchup_records:
                                matchup_records[key_loser] = {
                                    'wins': 0, 'losses': 0,
                                    'points_for': 0, 'points_against': 0,
                                    'games': 0
                                }

                            matchup_records[key_loser]['games'] += 1
                            matchup_records[key_loser]['losses'] += 1

        except Exception as e:
            pass

    # HARDCODED FIX: 2016 has corrupt ESPN schedule data for both Jesse and Russ
    # Russ's 2016 matchups: 6-11 record (17 games total)
    russ_2016_games = [
        ('Sean', 'W'),   # Week 1
        ('David', 'L'),  # Week 2
        ('Eddie', 'L'),  # Week 3
        ('Tom', 'W'),    # Week 4
        ('Bradley', 'L'), # Week 5
        ('Jesse', 'W'),  # Week 6
        ('Matthew', 'L'), # Week 7
        ('Eric', 'L'),   # Week 8
        ('Brian', 'L'),  # Week 9
        ('Evan', 'W'),   # Week 10
        ('John', 'L'),   # Week 11
        ('Sean', 'L'),   # Week 12
        ('David', 'L'),  # Week 13
        ('Eddie', 'W'),  # Week 14
        ('Tom', 'W'),    # Week 15
        ('Bradley', 'L'), # Week 16
        ('Jesse', 'L'),  # Week 17
    ]

    for opponent, result in russ_2016_games:
        # Add from Russ's perspective
        key_russ = ('Russ', opponent)
        if key_russ not in matchup_records:
            matchup_records[key_russ] = {
                'wins': 0, 'losses': 0,
                'points_for': 0, 'points_against': 0,
                'games': 0
            }
        matchup_records[key_russ]['games'] += 1
        if result == 'W':
            matchup_records[key_russ]['wins'] += 1
        else:
            matchup_records[key_russ]['losses'] += 1

        # Add from opponent's perspective
        key_opponent = (opponent, 'Russ')
        if key_opponent not in matchup_records:
            matchup_records[key_opponent] = {
                'wins': 0, 'losses': 0,
                'points_for': 0, 'points_against': 0,
                'games': 0
            }
        matchup_records[key_opponent]['games'] += 1
        if result == 'W':
            matchup_records[key_opponent]['losses'] += 1
        else:
            matchup_records[key_opponent]['wins'] += 1

    unique_first_names = set()
    for (owner1, owner2) in matchup_records.keys():
        unique_first_names.add(owner1)
        unique_first_names.add(owner2)

    return matchup_records, sorted(list(unique_first_names))


def get_rivalry_stats(matchup_records, owner1, owner2):
    """Get detailed stats for a specific rivalry"""

    key = (owner1, owner2)

    if key not in matchup_records:
        return None

    record = matchup_records[key]

    if record['games'] == 0:
        return None

    return {
        'owner1': owner1,
        'owner2': owner2,
        'owner1_wins': record['wins'],
        'owner1_losses': record['losses'],
        'owner2_wins': record['losses'],  # owner2's wins = owner1's losses
        'owner2_losses': record['wins'],  # owner2's losses = owner1's wins
        'owner1_points': record['points_for'],
        'owner2_points': record['points_against'],
        'total_games': record['games']
    }

def load_cached_data():
    """Load data from cache file (committed to git)"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, CACHE_FILE)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                # Don't check expiration - use the committed cache
                return cached.get('data'), cached.get('first_names')
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None, None

def save_cached_data(data, first_names):
    """Save data to cache file (only works locally)"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({
                'timestamp': datetime.now(),
                'data': data,
                'first_names': first_names
            }, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

@st.cache_resource
def load_h2h_data_cached():
    """Load h2h data - uses file cache first, then fetches if needed"""
    cached_data, cached_names = load_cached_data()
    if cached_data is not None and cached_names is not None:
        return cached_data, cached_names

    # Fetch fresh data only if no cache exists
    h2h_data, first_names = get_head_to_head_data(
        league_id=LEAGUE_ID,
        years=range(2011, 2026),
        espn_s2=ESPN_S2,
        swid=SWID
    )
    return h2h_data, first_names

def load_h2h_data(force_refresh=False):
    """Load h2h data with optional force refresh"""
    if force_refresh:
        # Clear the cache and fetch fresh
        load_h2h_data_cached.clear()
        h2h_data, first_names = get_head_to_head_data(
            league_id=LEAGUE_ID,
            years=range(2011, 2026),
            espn_s2=ESPN_S2,
            swid=SWID
        )
        # Save locally
        save_cached_data(h2h_data, first_names)
        return h2h_data, first_names

    return load_h2h_data_cached()

@st.cache_data(ttl=timedelta(days=7))
def load_trades_data():
    """Load and process trades CSV data"""
    filepath = "data/league_trades.csv"

    # Read CSV with proper encoding handling
    df = pd.read_csv(filepath, encoding='latin-1')

    # Clean up player names - fix encoding issues with special characters
    def clean_player_name(name):
        if pd.isna(name) or name == '':
            return ''
        # Remove the � character (which appears as a box with X when encoding fails)
        # These are decorative quotes around comma-separated player lists
        name = name.replace('�', '')
        # Remove any remaining double quotes
        name = name.replace('"', '')
        name = name.strip()
        return name

    # Apply cleaning to player columns
    for col in ['manager_a_players', 'manager_a_playoff_keepers', 'manager_a_season_keepers',
                'manager_b_players', 'manager_b_playoff_keepers', 'manager_b_season_keepers']:
        df[col] = df[col].apply(clean_player_name)

    # Convert trade_date to datetime
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Add a trade_id for easy reference
    df['trade_id'] = range(1, len(df) + 1)

    return df

# ============================================
# PLAYER PROPS FUNCTIONS
# ============================================

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

PROP_MARKETS = [
    'player_pass_yds',
    'player_pass_tds',
    'player_pass_interceptions',
    'player_rush_yds',
    'player_receptions',
    'player_reception_yds',
    'player_anytime_td',
]

MARKET_NAMES = {
    'player_pass_yds': 'Pass Yds',
    'player_pass_tds': 'Pass TDs',
    'player_pass_interceptions': 'INTs',
    'player_rush_yds': 'Rush Yds',
    'player_receptions': 'Receptions',
    'player_reception_yds': 'Rec Yds',
    'player_anytime_td': 'Anytime TD',
}

FANTASY_POINTS = {
    'player_pass_yds': 0.04,
    'player_pass_tds': 4.0,
    'player_pass_interceptions': -2.0,
    'player_rush_yds': 0.1,
    'player_receptions': 0.5,
    'player_reception_yds': 0.1,
    'player_anytime_td': 6.0,
}

SLOT_ORDER = ['QB', 'RB', 'RB/WR', 'WR', 'WR/TE', 'TE', 'FLEX', 'OP', 'K', 'D/ST', 'BE', 'IR']

@st.cache_resource
def get_props_cache_store():
    """Returns a persistent dict that survives across reruns"""
    return {'data': None, 'fetched_date': None}

def get_cached_props():
    """Get the globally cached props data"""
    return get_props_cache_store()['data']

def set_cached_props(data):
    """Set the globally cached props data"""
    store = get_props_cache_store()
    store['data'] = data
    store['fetched_date'] = datetime.utcnow().date().isoformat()

def should_refresh_props():
    """
    Determine if we should fetch fresh props data.
    Returns True if:
    - No cached data exists (first run)
    - It's Wednesday or Sunday AND we haven't fetched today
    """
    store = get_props_cache_store()
    cached_data = store['data']
    fetched_date_str = store['fetched_date']

    if cached_data is None:
        return True  # First run - always fetch

    if not fetched_date_str:
        return True  # No date - fetch

    now = datetime.utcnow()
    today_str = now.date().isoformat()

    # Check if it's Wednesday (2) or Sunday (6)
    is_refresh_day = now.weekday() in [2, 6]

    if not is_refresh_day:
        return False  # Not a refresh day - use cached data

    # It's a refresh day - check if we already fetched today
    if fetched_date_str == today_str:
        return False  # Already fetched today

    return True  # It's a refresh day and we haven't fetched today

def fetch_player_props_from_api():
    """Fetch player props directly from The Odds API"""
    if not ODDS_API_KEY:
        return None

    # Get this week's NFL events
    url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/events"
    today = datetime.utcnow()
    days_until_tuesday = (1 - today.weekday()) % 7
    if days_until_tuesday == 0:
        days_until_tuesday = 7
    week_end = today + timedelta(days=days_until_tuesday)

    params = {
        'apiKey': ODDS_API_KEY,
        'dateFormat': 'iso',
        'commenceTimeTo': week_end.strftime('%Y-%m-%dT23:59:59Z')
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return None
        events = response.json()
    except:
        return None

    if not events:
        return None

    # Fetch props for each game
    all_data = {
        'fetched_at': datetime.utcnow().isoformat(),
        'players': {}
    }

    for event in events:
        event_id = event.get('id')
        home = event.get('home_team')
        away = event.get('away_team')

        props_url = f"{ODDS_API_BASE}/sports/americanfootball_nfl/events/{event_id}/odds"
        props_params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': ','.join(PROP_MARKETS),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            props_response = requests.get(props_url, params=props_params)
            if props_response.status_code != 200:
                continue
            props_data = props_response.json()
        except:
            continue

        # Parse into player lookup
        for bookmaker in props_data.get('bookmakers', []):
            book_name = bookmaker.get('title', 'Unknown')

            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')

                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')

                    if not player_name:
                        continue

                    if 'D/ST' in player_name or 'Defense' in player_name or player_name == 'No Touchdown':
                        continue

                    if player_name not in all_data['players']:
                        all_data['players'][player_name] = {
                            'nfl_game': f"{away} @ {home}",
                            'props': {}
                        }

                    if market_key not in all_data['players'][player_name]['props']:
                        all_data['players'][player_name]['props'][market_key] = {
                            'books': {}
                        }

                    if book_name not in all_data['players'][player_name]['props'][market_key]['books']:
                        all_data['players'][player_name]['props'][market_key]['books'][book_name] = {
                            'line': None,
                            'over_odds': None,
                            'under_odds': None,
                            'odds': None
                        }

                    book_data = all_data['players'][player_name]['props'][market_key]['books'][book_name]

                    if market_key == 'player_anytime_td':
                        if outcome.get('name') == 'Yes':
                            book_data['odds'] = outcome.get('price')
                    else:
                        if outcome.get('name') == 'Over':
                            book_data['over_odds'] = outcome.get('price')
                            book_data['line'] = outcome.get('point')
                        elif outcome.get('name') == 'Under':
                            book_data['under_odds'] = outcome.get('price')
                            if book_data['line'] is None:
                                book_data['line'] = outcome.get('point')

    return all_data

def odds_to_probability(odds):
    """Convert American odds to implied probability"""
    if odds is None:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def calculate_fantasy_points(market_key, book_data):
    """Calculate expected fantasy points from a prop line"""
    multiplier = FANTASY_POINTS.get(market_key, 0)
    if market_key == 'player_anytime_td':
        odds = book_data.get('odds')
        prob = odds_to_probability(odds)
        if prob is not None:
            return prob * multiplier
        return None
    else:
        line = book_data.get('line')
        if line is not None:
            return line * multiplier
        return None

def clean_name_for_matching(name):
    """Remove suffixes like Jr., Sr., II, III and normalize periods for better matching"""
    name = name.replace('.', '')
    suffixes = ['jr', 'sr', 'ii', 'iii', 'iv', 'v']
    parts = name.split()
    while parts and parts[-1].lower() in suffixes:
        parts = parts[:-1]
    return parts

def find_player_props(player_name, all_props):
    """Find props for a player by name"""
    if player_name in all_props:
        return all_props[player_name]

    player_parts = clean_name_for_matching(player_name)

    if len(player_parts) >= 2:
        first_name = player_parts[0].lower()
        last_name = player_parts[-1].lower()

        for prop_name in all_props:
            prop_parts = clean_name_for_matching(prop_name)
            if len(prop_parts) >= 2:
                prop_first = prop_parts[0].lower()
                prop_last = prop_parts[-1].lower()
                if last_name == prop_last and first_name == prop_first:
                    return all_props[prop_name]

    return None

def get_player_fantasy_projection(player_name, position, all_props, sportsbook):
    """Get fantasy point projection for a player from a specific sportsbook"""
    player_data = find_player_props(player_name, all_props)
    if not player_data:
        return None, {}

    props = player_data.get('props', {})
    if not props:
        return None, {}

    if position == 'QB':
        required_markets = {'player_pass_yds', 'player_pass_tds', 'player_anytime_td'}
    elif position == 'RB':
        required_markets = {'player_rush_yds', 'player_receptions', 'player_anytime_td'}
    elif position in ['WR', 'TE']:
        required_markets = {'player_reception_yds', 'player_receptions', 'player_anytime_td'}
    else:
        required_markets = set()

    total_pts = 0
    breakdown = {}
    markets_found = set()

    for market_key, market_data in props.items():
        books = market_data.get('books', {})
        if sportsbook in books:
            book_data = books[sportsbook]
            fpts = calculate_fantasy_points(market_key, book_data)
            if fpts is not None:
                markets_found.add(market_key)
                total_pts += fpts
                line = book_data.get('line') or book_data.get('odds')
                breakdown[market_key] = {'line': line, 'fpts': fpts}

    if required_markets.issubset(markets_found):
        return total_pts, breakdown
    else:
        return None, breakdown

def get_best_projection_across_books(player_name, position, all_props, sportsbooks):
    """Get the MAX fantasy projection across all sportsbooks"""
    best_total = None
    best_book = None
    all_book_projections = {}

    for book in sportsbooks:
        total_pts, breakdown = get_player_fantasy_projection(player_name, position, all_props, book)
        if total_pts is not None:
            all_book_projections[book] = {'total': total_pts, 'breakdown': breakdown}
            if best_total is None or total_pts > best_total:
                best_total = total_pts
                best_book = book

    return best_total, best_book, all_book_projections

def get_partial_projection(player_name, position, all_props, sportsbooks):
    """Get partial projection data even when incomplete"""
    all_book_data = {}
    for book in sportsbooks:
        total_pts, breakdown = get_player_fantasy_projection(player_name, position, all_props, book)
        if breakdown:
            partial_total = sum(d['fpts'] for d in breakdown.values())
            all_book_data[book] = {'total': partial_total, 'breakdown': breakdown, 'complete': total_pts is not None}
    return all_book_data

def get_current_week_matchups_for_props():
    """Get current week's fantasy matchups with rosters"""
    league = League(league_id=LEAGUE_ID, year=2025, espn_s2=ESPN_S2, swid=SWID)
    current_week = league.current_week
    box_scores = league.box_scores(week=current_week)

    matchups = []
    for matchup in box_scores:
        home_owner = "Unknown"
        away_owner = "Unknown"

        if hasattr(matchup.home_team, 'owners') and matchup.home_team.owners:
            owner_obj = matchup.home_team.owners[0]
            if isinstance(owner_obj, dict) and 'firstName' in owner_obj:
                home_owner = owner_obj['firstName']

        if hasattr(matchup.away_team, 'owners') and matchup.away_team.owners:
            owner_obj = matchup.away_team.owners[0]
            if isinstance(owner_obj, dict) and 'firstName' in owner_obj:
                away_owner = owner_obj['firstName']

        home_players = []
        away_players = []

        if hasattr(matchup, 'home_lineup'):
            for player in matchup.home_lineup:
                home_players.append({
                    'name': player.name,
                    'position': getattr(player, 'position', 'Unknown'),
                    'pro_team': getattr(player, 'proTeam', 'Unknown'),
                    'slot': getattr(player, 'slot_position', 'Unknown'),
                })

        if hasattr(matchup, 'away_lineup'):
            for player in matchup.away_lineup:
                away_players.append({
                    'name': player.name,
                    'position': getattr(player, 'position', 'Unknown'),
                    'pro_team': getattr(player, 'proTeam', 'Unknown'),
                    'slot': getattr(player, 'slot_position', 'Unknown'),
                })

        matchups.append({
            'home_owner': home_owner,
            'away_owner': away_owner,
            'home_team_name': matchup.home_team.team_name,
            'away_team_name': matchup.away_team.team_name,
            'home_players': home_players,
            'away_players': away_players,
        })

    return matchups, current_week

# Page config
st.set_page_config(page_title="The Franchise Dashboard", page_icon="🏈", layout="wide")

st.title("🏈 The Franchise Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 All-Time Standings", "⚔️ Head-to-Head", "📈 Visualizations", "🔄 Trades", "🎰 4 The Degenz", "ℹ️ About"])

with tab2:
    st.subheader("⚔️ Head-to-Head Records")

    # Add refresh button in sidebar
    with st.sidebar:
        if st.button("🔄 Refresh Data"):
            h2h_data, first_names = load_h2h_data(force_refresh=True)
            st.success("Data refreshed!")
        else:
            with st.spinner("Loading head-to-head data..."):
                h2h_data, first_names = load_h2h_data()

    st.markdown("### Select Your Team")

    # Create buttons using first names
    cols = st.columns(min(len(first_names), 5))  # Max 5 buttons per row

    selected_owner = None
    for idx, first_name in enumerate(first_names):
        col_idx = idx % 5

        with cols[col_idx]:
            if st.button(first_name, key=f"btn_{first_name}", use_container_width=True):
                st.session_state['selected_owner'] = first_name

    # Get selected owner from session state
    # In tab2, after loading h2h_data
    if 'selected_owner' in st.session_state:
        selected_owner = st.session_state['selected_owner']

        st.markdown(f"## 📊 {selected_owner}'s Head-to-Head Records")
        st.markdown("---")

        # Build records against all opponents
        records = []
        for opponent in first_names:
            if opponent != selected_owner:
                rivalry = get_rivalry_stats(h2h_data, selected_owner, opponent)

                if rivalry and rivalry['total_games'] > 0:
                    win_pct = (rivalry['owner1_wins'] / rivalry['total_games']) * 100

                    records.append({
                        'Opponent': opponent,
                        'Record': f"{rivalry['owner1_wins']}-{rivalry['owner1_losses']}",
                        'Win %': win_pct,
                        'Wins': rivalry['owner1_wins'],
                        'Losses': rivalry['owner1_losses'],
                        'Points For (starting 2019)': rivalry['owner1_points'],
                        'Points Against (starting 2019)': rivalry['owner2_points'],
                        'Avg Margin (starting 2019)': (rivalry['owner1_points'] - rivalry['owner2_points']) / rivalry['total_games']
                    })

        if records:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)

            total_wins = sum(r['Wins'] for r in records)
            total_losses = sum(r['Losses'] for r in records)
            total_games = total_wins + total_losses
            overall_win_pct = (total_wins / total_games * 100) if total_games > 0 else 0

            with col1:
                st.metric("Overall Record", f"{total_wins}-{total_losses}")
            with col2:
                st.metric("Win Percentage", f"{overall_win_pct:.1f}%")
            with col3:
                best_matchup = max(records, key=lambda x: x['Win %'])
                st.metric("Best Matchup", best_matchup['Opponent'],
                         f"{best_matchup['Win %']:.0f}% win rate")
            with col4:
                worst_matchup = min(records, key=lambda x: x['Win %'])
                st.metric("Worst Matchup", worst_matchup['Opponent'],
                         f"{worst_matchup['Win %']:.0f}% win rate")

            st.markdown("---")

            # Detailed records table
            st.markdown("### Detailed Records by Opponent")

            records_df = pd.DataFrame(records)
            records_df = records_df.sort_values('Win %', ascending=False)

            # Format for display
            display_df = records_df[['Opponent', 'Record', 'Win %', 'Points For (starting 2019)', 'Points Against (starting 2019)', 'Avg Margin (starting 2019)']].copy()
            display_df['Win %'] = display_df['Win %'].apply(lambda x: f"{x:.1f}%")
            display_df['Points For (starting 2019)'] = display_df['Points For (starting 2019)'].apply(lambda x: f"{x:.1f}")
            display_df['Points Against (starting 2019)'] = display_df['Points Against (starting 2019)'].apply(lambda x: f"{x:.1f}")
            display_df['Avg Margin (starting 2019)'] = display_df['Avg Margin (starting 2019)'].apply(lambda x: f"{x:+.1f}")

            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # Visualization
            st.markdown("---")
            st.markdown("### Win Rate by Opponent")

            fig = px.bar(
                records_df.sort_values('Win %', ascending=True),
                x='Win %',
                y='Opponent',
                orientation='h',
                color='Win %',
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                text='Record'
            )

            fig.update_traces(textposition='outside')
            fig.add_vline(x=50, line_dash="dash", line_color="gray",
                         annotation_text="50%", annotation_position="top")

            fig.update_layout(
                height=max(400, len(records) * 40),
                showlegend=False,
                xaxis_title="Win Percentage",
                yaxis_title=""
            )

            st.plotly_chart(fig, use_container_width=True)

            # Dominance / Struggles
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💪 Dominating")
                dominant = [r for r in records if r['Win %'] >= 60]
                if dominant:
                    for r in sorted(dominant, key=lambda x: x['Win %'], reverse=True)[:5]:
                        st.success(f"**{r['Opponent']}** - {r['Record']} ({r['Win %']:.0f}%)")
                else:
                    st.info("No dominant matchups (60%+ win rate)")

            with col2:
                st.markdown("### 😰 Struggling Against")
                struggling = [r for r in records if r['Win %'] <= 40]
                if struggling:
                    for r in sorted(struggling, key=lambda x: x['Win %'])[:5]:
                        st.error(f"**{r['Opponent']}** - {r['Record']} ({r['Win %']:.0f}%)")
                else:
                    st.info("No struggling matchups (40%- win rate)")

        else:
            st.info(f"No head-to-head data available for {selected_owner}")

    else:
        st.info("👆 Select a team above to view their head-to-head records")

with tab3:
    st.markdown("### 📈 Visualizations")
    st.info("Coming soon!")

with tab4:
    st.subheader("🔄 Trade Analysis")

    # Load trades data
    trades_df = load_trades_data()

    st.markdown(f"### {len(trades_df)} Trades from {trades_df['year'].min()} to {trades_df['year'].max()}")

    # ============================================
    # SECTION 1: Trading Activity Timeline Heatmap
    # ============================================
    st.markdown("---")
    st.markdown("### 📅 Trading Activity by Season & Week")

    # Calculate NFL week based on date
    # NFL season typically starts first week of September, runs ~18 weeks
    def get_nfl_week(date, year):
        """Estimate NFL week number based on date"""
        # NFL season usually starts first Thursday after Labor Day (first Mon in Sept)
        # Approximate: Week 1 starts around Sept 5-11
        import datetime
        season_start = datetime.date(year, 9, 7)  # Approximate start

        # Adjust if date is before September (could be previous season)
        if date.month < 9:
            return None  # Offseason

        # Calculate weeks since season start
        days_diff = (date - season_start).days
        nfl_week = max(1, (days_diff // 7) + 1)

        # Cap at 18 (regular season)
        return min(nfl_week, 18)

    # Build matrix: rows = years, columns = NFL weeks
    from collections import defaultdict
    trade_activity = defaultdict(lambda: defaultdict(int))
    trade_details_map = defaultdict(lambda: defaultdict(list))

    for _, row in trades_df.iterrows():
        year = row['year']
        nfl_week = get_nfl_week(row['trade_date'].date(), year)

        if nfl_week:  # Only count trades during NFL season
            trade_activity[year][nfl_week] += 1
            trade_details_map[year][nfl_week].append({
                'date': row['trade_date'].strftime('%Y-%m-%d'),
                'manager_a': row['manager_a'],
                'manager_a_players': row['manager_a_players'],
                'manager_b': row['manager_b'],
                'manager_b_players': row['manager_b_players']
            })

    # Get all years (including years with no trades) and NFL weeks
    all_years = list(range(trades_df['year'].min(), trades_df['year'].max() + 1))
    nfl_weeks = range(1, 19)  # NFL Weeks 1-18

    # Build matrix
    matrix = []
    hover_text = []
    for year in all_years:
        year_row = []
        hover_row = []
        for week in nfl_weeks:
            count = trade_activity[year][week]
            year_row.append(count)

            # Build hover text with trade details
            if count > 0:
                trades_info = trade_details_map[year][week]
                hover = f"<b>NFL Week {week}, {year}</b><br>{count} trade(s)<br><br>"
                for i, trade in enumerate(trades_info, 1):
                    hover += f"Trade {i}:<br>"
                    hover += f"  {trade['manager_a']} → {trade['manager_b']}<br>"
                hover_row.append(hover)
            else:
                hover_row.append(f"NFL Week {week}, {year}<br>No trades")
        matrix.append(year_row)
        hover_text.append(hover_row)

    # Create heatmap
    fig_timeline = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(nfl_weeks),
        y=[str(y) for y in all_years],
        colorscale='YlOrRd',
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        showscale=True,
        colorbar=dict(title="Trades")
    ))

    fig_timeline.update_layout(
        height=500,
        xaxis_title="NFL Week",
        yaxis_title="Season",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(tickmode='linear', dtick=1, showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white'
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

    # ============================================
    # SECTION 2: Summary Statistics
    # ============================================
    st.markdown("---")
    st.markdown("### 📊 Trade Statistics")

    # Calculate stats
    manager_a_counts = trades_df['manager_a'].value_counts()
    manager_b_counts = trades_df['manager_b'].value_counts()
    total_trades_per_manager = manager_a_counts.add(manager_b_counts, fill_value=0).sort_values(ascending=False)

    # Display in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", len(trades_df))

    with col2:
        most_active = total_trades_per_manager.index[0]
        most_active_count = int(total_trades_per_manager.iloc[0])
        st.metric("Most Active Trader", most_active, f"{most_active_count} trades")

    with col3:
        least_active = total_trades_per_manager.index[-1]
        least_active_count = int(total_trades_per_manager.iloc[-1])
        st.metric("Least Active Trader", least_active, f"{least_active_count} trades")

    with col4:
        trades_this_year = len(trades_df[trades_df['year'] == trades_df['year'].max()])
        st.metric(f"{trades_df['year'].max()} Trades", trades_this_year)

    # Bar chart of trades per manager
    st.markdown("#### Trades by Manager")
    trades_per_manager_df = pd.DataFrame({
        'Manager': total_trades_per_manager.index,
        'Trades': total_trades_per_manager.values
    })

    fig_bar = px.bar(
        trades_per_manager_df,
        x='Manager',
        y='Trades',
        color='Trades',
        color_continuous_scale='Blues',
        text='Trades'
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ============================================
    # SECTION 3: Filterable Trades Table
    # ============================================
    st.markdown("---")
    st.markdown("### 🔍 Find Trades by Manager")

    # Get list of all unique managers
    all_managers = sorted(list(set(trades_df['manager_a'].unique()) | set(trades_df['manager_b'].unique())))

    # Manager filter
    selected_manager = st.selectbox("Select a manager to view their trades:", ["All Managers"] + all_managers)

    # Year range filter
    col_year1, col_year2 = st.columns(2)
    with col_year1:
        min_year = st.selectbox("From Year:", sorted(trades_df['year'].unique()), index=0)
    with col_year2:
        max_year = st.selectbox("To Year:", sorted(trades_df['year'].unique(), reverse=True), index=0)

    # Filter data
    filtered_df = trades_df.copy()

    if selected_manager != "All Managers":
        filtered_df = filtered_df[
            (filtered_df['manager_a'] == selected_manager) |
            (filtered_df['manager_b'] == selected_manager)
        ]

    filtered_df = filtered_df[
        (filtered_df['year'] >= min_year) &
        (filtered_df['year'] <= max_year)
    ]

    st.markdown(f"**Showing {len(filtered_df)} trades**")

    # Display trades table
    if len(filtered_df) > 0:
        # Reformat trades from the perspective of the selected manager
        if selected_manager != "All Managers":
            # Create perspective-based view
            perspective_trades = []
            for _, row in filtered_df.iterrows():
                if row['manager_a'] == selected_manager:
                    # Selected manager is Manager A
                    perspective_trades.append({
                        'Date': row['trade_date'].strftime('%Y-%m-%d'),
                        'Year': row['year'],
                        'Trading Partner': row['manager_b'],
                        'I Gave': row['manager_a_players'],
                        'I Received': row['manager_b_players']
                    })
                else:
                    # Selected manager is Manager B
                    perspective_trades.append({
                        'Date': row['trade_date'].strftime('%Y-%m-%d'),
                        'Year': row['year'],
                        'Trading Partner': row['manager_a'],
                        'I Gave': row['manager_b_players'],
                        'I Received': row['manager_a_players']
                    })

            display_trades = pd.DataFrame(perspective_trades)
        else:
            # For "All Managers", keep the original format
            display_trades = filtered_df[['trade_date', 'year', 'manager_a', 'manager_a_players',
                                          'manager_b', 'manager_b_players']].copy()
            display_trades['trade_date'] = display_trades['trade_date'].dt.strftime('%Y-%m-%d')
            display_trades.columns = ['Date', 'Year', 'Manager A', 'Manager A Gave', 'Manager B', 'Manager B Gave']

        # Sort by date descending (most recent first)
        display_trades = display_trades.sort_values('Date', ascending=False)

        st.dataframe(
            display_trades,
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Show individual trade details on expand
        with st.expander("📋 View Individual Trade Details"):
            trade_to_view = st.selectbox(
                "Select a trade:",
                options=filtered_df['trade_id'].tolist(),
                format_func=lambda x: f"Trade #{x} - {filtered_df[filtered_df['trade_id']==x]['trade_date'].iloc[0].strftime('%Y-%m-%d')} - {filtered_df[filtered_df['trade_id']==x]['manager_a'].iloc[0]} ↔ {filtered_df[filtered_df['trade_id']==x]['manager_b'].iloc[0]}"
            )

            trade_detail = filtered_df[filtered_df['trade_id'] == trade_to_view].iloc[0]

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"### {trade_detail['manager_a']}")
                st.markdown(f"**Traded away:**")
                for player in trade_detail['manager_a_players'].split(','):
                    st.markdown(f"- {player.strip()}")

                if trade_detail['manager_a_playoff_keepers']:
                    st.markdown(f"**Playoff Keepers:** {trade_detail['manager_a_playoff_keepers']}")
                if trade_detail['manager_a_season_keepers']:
                    st.markdown(f"**Season Keepers:** {trade_detail['manager_a_season_keepers']}")

            with col_b:
                st.markdown(f"### {trade_detail['manager_b']}")
                st.markdown(f"**Traded away:**")
                for player in trade_detail['manager_b_players'].split(','):
                    st.markdown(f"- {player.strip()}")

                if trade_detail['manager_b_playoff_keepers']:
                    st.markdown(f"**Playoff Keepers:** {trade_detail['manager_b_playoff_keepers']}")
                if trade_detail['manager_b_season_keepers']:
                    st.markdown(f"**Season Keepers:** {trade_detail['manager_b_season_keepers']}")
    else:
        st.info("No trades found matching the selected filters.")

# ============================================
# PLAYER PROPS TAB
# ============================================

with tab5:
    st.subheader("🎰 Player Props & Fantasy Projections")

    if not ODDS_API_KEY:
        st.error("ODDS_API_KEY not configured. Please add it to Streamlit secrets.")
    else:
        # Check if we need to refresh
        if should_refresh_props():
            with st.spinner("Fetching player props from API..."):
                props_cache = fetch_player_props_from_api()
                set_cached_props(props_cache)
        else:
            props_cache = get_cached_props()

        if props_cache is None:
            st.error("Could not fetch player props data. The API may be unavailable or you may have exceeded your quota.")
        else:
            fetched_at = props_cache.get('fetched_at', 'Unknown')
            st.caption(f"Data fetched: {fetched_at} (refreshes Wed & Sun mornings)")

            all_props = props_cache.get('players', {})

            sportsbooks = set()
            for player_data in all_props.values():
                for market_data in player_data.get('props', {}).values():
                    sportsbooks.update(market_data.get('books', {}).keys())
            sportsbooks = sorted(list(sportsbooks))

            with st.spinner("Loading fantasy matchups..."):
                matchups, current_week = get_current_week_matchups_for_props()

            st.markdown(f"### Week {current_week} Matchups")

            matchup_options = [f"{m['away_owner']} vs {m['home_owner']}" for m in matchups]
            selected_matchup_idx = st.selectbox("Select Matchup:", range(len(matchup_options)), format_func=lambda x: matchup_options[x])

            st.markdown("---")

            selected_matchup = matchups[selected_matchup_idx]

            col_away, col_home = st.columns(2)

            def sort_players_by_slot(players):
                def slot_key(p):
                    slot = p.get('slot', 'BE')
                    if slot in SLOT_ORDER:
                        return SLOT_ORDER.index(slot)
                    return len(SLOT_ORDER)
                return sorted(players, key=slot_key)

            def render_player_row(slot, name, pro_team, best_total, best_book, all_book_data, is_partial=False):
                if best_total is not None:
                    pts_display = f"~{best_total:.1f}*" if is_partial else f"{best_total:.1f}"
                    expander_label = f"**{slot}** {name} ({pro_team}) — **{pts_display} pts** ({best_book})"
                else:
                    expander_label = f"**{slot}** {name} ({pro_team}) — *No data*"

                with st.expander(expander_label):
                    if all_book_data:
                        sorted_books = sorted(all_book_data.items(), key=lambda x: x[1]['total'], reverse=True)

                        for book, data in sorted_books:
                            total = data['total']
                            breakdown = data['breakdown']
                            is_complete = data.get('complete', True)

                            if book == best_book:
                                st.markdown(f"**🏆 {book}: {total:.1f} pts**")
                            else:
                                incomplete_marker = " *(partial)*" if not is_complete else ""
                                st.markdown(f"**{book}: {total:.1f} pts{incomplete_marker}**")

                            breakdown_parts = []
                            for mkt, mkt_data in breakdown.items():
                                mkt_name = MARKET_NAMES.get(mkt, mkt)
                                line = mkt_data['line']
                                fpts = mkt_data['fpts']
                                if mkt == 'player_anytime_td':
                                    breakdown_parts.append(f"{mkt_name}: {line:+d} → {fpts:.1f}")
                                else:
                                    breakdown_parts.append(f"{mkt_name}: {line} → {fpts:.1f}")

                            st.caption(" | ".join(breakdown_parts))
                    else:
                        st.caption("No prop data available for this player")

            def display_team_roster(owner, players, all_props, sportsbooks, column):
                with column:
                    st.markdown(f"### {owner}")

                    sorted_players = sort_players_by_slot(players)

                    starters = [p for p in sorted_players if p.get('slot') not in ['BE', 'IR']]
                    bench = [p for p in sorted_players if p.get('slot') == 'BE']

                    starters = [p for p in starters if p['position'] not in ['K', 'D/ST']]
                    bench = [p for p in bench if p['position'] not in ['K', 'D/ST']]

                    st.markdown("#### 🏈 Starters")

                    for player in starters:
                        slot = player.get('slot', 'BE')
                        name = player['name']
                        position = player['position']
                        pro_team = player['pro_team']

                        if slot in ['RB/WR', 'WR/TE', 'RB/WR/TE', 'OP']:
                            slot = 'FLEX'

                        best_total, best_book, all_book_data = get_best_projection_across_books(
                            name, position, all_props, sportsbooks
                        )

                        if best_total is not None:
                            render_player_row(slot, name, pro_team, best_total, best_book, all_book_data, is_partial=False)
                        else:
                            partial_data = get_partial_projection(name, position, all_props, sportsbooks)

                            if partial_data:
                                best_partial = max(partial_data.values(), key=lambda x: x['total'])
                                best_partial_book = [k for k, v in partial_data.items() if v['total'] == best_partial['total']][0]
                                render_player_row(slot, name, pro_team, best_partial['total'], best_partial_book, partial_data, is_partial=True)
                            else:
                                render_player_row(slot, name, pro_team, None, None, {})

                    if bench:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown("#### 🪑 Bench")

                        for player in bench:
                            slot = player.get('slot', 'BE')
                            name = player['name']
                            position = player['position']
                            pro_team = player['pro_team']

                            best_total, best_book, all_book_data = get_best_projection_across_books(
                                name, position, all_props, sportsbooks
                            )

                            if best_total is not None:
                                render_player_row(slot, name, pro_team, best_total, best_book, all_book_data, is_partial=False)
                            else:
                                partial_data = get_partial_projection(name, position, all_props, sportsbooks)

                                if partial_data:
                                    best_partial = max(partial_data.values(), key=lambda x: x['total'])
                                    best_partial_book = [k for k, v in partial_data.items() if v['total'] == best_partial['total']][0]
                                    render_player_row(slot, name, pro_team, best_partial['total'], best_partial_book, partial_data, is_partial=True)
                                else:
                                    render_player_row(slot, name, pro_team, None, None, {})

            display_team_roster(
                selected_matchup['away_owner'],
                selected_matchup['away_players'],
                all_props,
                sportsbooks,
                col_away
            )

            display_team_roster(
                selected_matchup['home_owner'],
                selected_matchup['home_players'],
                all_props,
                sportsbooks,
                col_home
            )

with tab6:
    st.markdown("""
    ## About This Dashboard

    Welcome to our Fantasy Football League Dashboard! This dashboard tracks league history from 2011-2025.

    **Features:**
    - Head-to-head analysis across all years
    - Historical matchup records
    - Win percentages and rivalry tracking
    - Player Props & Fantasy Projections from The Odds API

    **Data Updates:**
    - H2H data is cached for 7 days to improve performance
    - Use the refresh button in the sidebar to manually update
    - Player props are fetched live and cached for 6 hours

    *Dashboard built with Streamlit, ESPN Fantasy API, and The Odds API*
    """)
