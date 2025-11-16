import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from espn_api.football import League
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# Add your ESPN credentials at the top
ESPN_S2 = st.secrets.get("ESPN_S2", "")
SWID = st.secrets.get("SWID", "")
LEAGUE_ID = st.secrets.get("LEAGUE_ID", "")

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
    """Load data from cache if available and not expired"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached = pickle.load(f)
                cache_time = cached.get('timestamp')
                if cache_time and (datetime.now() - cache_time) < timedelta(days=CACHE_DURATION_DAYS):
                    return cached.get('data'), cached.get('first_names')
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None, None

def save_cached_data(data, first_names):
    """Save data to cache file"""
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({
                'timestamp': datetime.now(),
                'data': data,
                'first_names': first_names
            }, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

# Update the years to include 2011-2025
def load_h2h_data(force_refresh=False):
    """Load h2h data with persistent caching"""
    # Try to load from cache first
    if not force_refresh:
        cached_data, cached_names = load_cached_data()
        if cached_data is not None and cached_names is not None:
            return cached_data, cached_names

    # Fetch fresh data
    h2h_data, first_names = get_head_to_head_data(
        league_id=LEAGUE_ID,
        years=range(2011, 2026),
        espn_s2=ESPN_S2,
        swid=SWID
    )

    # Save to cache
    save_cached_data(h2h_data, first_names)

    return h2h_data, first_names

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

# Page config
st.set_page_config(page_title="The Franchise Dashboard", page_icon="🏈", layout="wide")

st.title("🏈 The Franchise Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All-Time Standings", "⚔️ Head-to-Head", "📈 Visualizations", "🔄 Trades", "ℹ️ About"])

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
        # Format the display
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

with tab5:
    st.markdown("""
    ## About This Dashboard

    Welcome to our Fantasy Football League Dashboard! This dashboard tracks league history from 2011-2025.

    **Features:**
    - Head-to-head analysis across all years
    - Historical matchup records
    - Win percentages and rivalry tracking

    **Data Updates:**
    - Data is cached for 7 days to improve performance
    - Use the refresh button in the sidebar to manually update

    *Dashboard built with Streamlit and ESPN Fantasy API*
    """)
