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

# Cache file path
CACHE_FILE = "h2h_cache.pkl"
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

def get_head_to_head_data(league_id, years, espn_s2, swid, debug_owners=None):
    """Build head-to-head record matrix across all years"""

    matchup_records = {}
    owner_first_name_map = {}
    debug_matchups = []

    for year in years:
        try:
            league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)

            print(f"\n=== Processing {year} ===")

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
            # For 2019+, right after building the firstName mapping
            if year >= 2019:
                print(f"Using box_scores for {year}")

                # Debug: Show name mappings for debug owners
                if debug_owners:
                    print(f"  Name mappings for {year}:")
                    for owner_id, first_name in owner_first_name_map.items():
                        if first_name in debug_owners or any(debug_name.lower() in str(owner_id).lower() for debug_name in debug_owners):
                            print(f"    '{owner_id}' -> '{first_name}'")

                week = 1
                max_week = 20

                while week <= max_week:
                    try:
                        box_scores = league.box_scores(week=week)

                        if not box_scores:
                            print(f"  Week {week}: No matchups found")
                            break

                        for matchup in box_scores:
                            home_owner = matchup.home_team.owner if hasattr(matchup.home_team, 'owner') else matchup.home_team.team_name
                            away_owner = matchup.away_team.owner if hasattr(matchup.away_team, 'owner') else matchup.away_team.team_name

                            home_first = normalize_name(owner_first_name_map.get(home_owner, home_owner))
                            away_first = normalize_name(owner_first_name_map.get(away_owner, away_owner))

                            home_score = matchup.home_score
                            away_score = matchup.away_score

                            # ========== DEBUG CODE MOVED INSIDE LOOP ==========
                            if debug_owners and set([home_first, away_first]) == set(debug_owners):
                                winner = home_first if home_score > away_score else away_first
                                loser = away_first if home_score > away_score else home_first
                                debug_matchups.append({
                                    'year': year,
                                    'week': week,
                                    'home': home_first,
                                    'away': away_first,
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'winner': winner,
                                    'loser': loser
                                })
                                print(f"  DEBUG 2019+: Week {week}: {winner} beat {loser} ({home_score:.1f} - {away_score:.1f})")
                            # ========== END DEBUG CODE ==========

                            # FIXED: Update BOTH perspectives consistently
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
                        print(f"  Week {week}: {e}")
                        break

            # For pre-2019, use team schedules
            else:
                print(f"Using schedules for {year}")
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

                        # WORKAROUND: Skip known bad schedule data
                        # In 2016, Jesse's team_id=1 has misaligned schedule data
                        if year == 2016 and team_id == 1 and opponent_first in ['David', 'Dave']:
                            continue

                        # Use smaller team_id first to ensure consistent ordering
                        # Include week_idx from the lower team_id to maintain consistency
                        if team_id < opponent_team_id:
                            matchup_id = (team_id, opponent_team_id, year, week_idx)
                        else:
                            # Skip - we'll process this matchup when we get to the other team
                            continue

                        # Additional debug: print team info for mismatched schedules
                        if debug_owners and year == 2016 and set([owner_first, opponent_first]) == set(debug_owners):
                            print(f"    Processing from {owner_first}'s schedule: team_id={team_id}, opponent_team_id={opponent_team_id}, week_idx={week_idx}")

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

                            # Debug
                            if debug_owners and set([owner_first, opponent_first]) == set(debug_owners):
                                debug_matchups.append({
                                    'year': year, 'week': week_idx + 1,
                                    'winner': winner, 'loser': loser,
                                    'matchup_id': matchup_id,
                                    'processing_from_team': owner_first
                                })
                                print(f"  DEBUG PRE-2019: Week {week_idx+1}: {winner} beat {loser} | From {owner_first}'s schedule (outcome={outcome}) | matchup_id: {matchup_id}")


                            # FIXED: Update BOTH perspectives consistently
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
            print(f"Error processing {year}: {e}")

    # Debug summary
    if debug_owners and debug_matchups:
        print(f"\n=== DEBUG: {debug_owners[0]} vs {debug_owners[1]} ===")
        print(f"Total matchups: {len(debug_matchups)}")
        wins_0 = sum(1 for m in debug_matchups if m.get('winner') == debug_owners[0])
        wins_1 = sum(1 for m in debug_matchups if m.get('winner') == debug_owners[1])
        print(f"{debug_owners[0]} wins: {wins_0}")
        print(f"{debug_owners[1]} wins: {wins_1}")

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

# Update the years to include all 14 years
def load_h2h_data(debug_owners=None, force_refresh=False):
    """Load h2h data with persistent caching"""
    # Try to load from cache first
    if not force_refresh:
        cached_data, cached_names = load_cached_data()
        if cached_data is not None and cached_names is not None:
            return cached_data, cached_names

    # Fetch fresh data
    h2h_data, first_names = get_head_to_head_data(
        league_id=LEAGUE_ID,
        years=range(2011, 2025),
        espn_s2=ESPN_S2,
        swid=SWID,
        debug_owners=debug_owners
    )

    # Save to cache
    save_cached_data(h2h_data, first_names)

    return h2h_data, first_names

# Page config
st.set_page_config(page_title="Fantasy Football Dashboard", page_icon="🏈", layout="wide")

st.title("🏈 Fantasy Football League Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["📊 All-Time Standings", "⚔️ Head-to-Head", "📈 Visualizations", "ℹ️ About"])

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
    st.markdown("""
    ## About This Dashboard

    Welcome to our Fantasy Football League Dashboard! This dashboard tracks league history from 2011-2024.

    **Features:**
    - Head-to-head analysis across all years
    - Historical matchup records
    - Win percentages and rivalry tracking

    **Data Updates:**
    - Data is cached for 7 days to improve performance
    - Use the refresh button in the sidebar to manually update

    *Dashboard built with Streamlit and ESPN Fantasy API*
    """)
