import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Fantasy Football Dashboard", page_icon="🏈", layout="wide")

# Title
st.title("🏈 Fantasy Football League Dashboard")
st.markdown("### 15 Years of League History")

# Dummy data for now
dummy_data = pd.DataFrame({
    'owner': ['Owner A', 'Owner B', 'Owner C', 'Owner D', 'Owner E'],
    'total_wins': [95, 87, 82, 78, 73],
    'total_losses': [55, 63, 68, 72, 77],
    'championships': [3, 2, 1, 1, 0],
    'total_points': [15234, 14876, 14521, 14203, 13987]
})

# Add win percentage
dummy_data['win_pct'] = (dummy_data['total_wins'] / 
                          (dummy_data['total_wins'] + dummy_data['total_losses']) * 100).round(1)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 All-Time Standings", "📈 Visualizations", "ℹ️ About"])

with tab1:
    st.subheader("All-Time League Standings")
    st.dataframe(
        dummy_data.sort_values('win_pct', ascending=False),
        hide_index=True,
        use_container_width=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Most Wins", dummy_data.loc[dummy_data['total_wins'].idxmax(), 'owner'], 
                  f"{dummy_data['total_wins'].max()} wins")
    with col2:
        st.metric("Most Championships", 
                  dummy_data.loc[dummy_data['championships'].idxmax(), 'owner'],
                  f"{dummy_data['championships'].max()} titles")
    with col3:
        st.metric("Highest Points", 
                  dummy_data.loc[dummy_data['total_points'].idxmax(), 'owner'],
                  f"{dummy_data['total_points'].max():,} pts")

with tab2:
    st.subheader("Win Percentage Comparison")
    fig = px.bar(
        dummy_data.sort_values('win_pct', ascending=True),
        x='win_pct',
        y='owner',
        orientation='h',
        title="All-Time Win Percentage",
        labels={'win_pct': 'Win %', 'owner': 'Owner'},
        color='win_pct',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Championships Won")
    fig2 = px.pie(
        dummy_data,
        values='championships',
        names='owner',
        title="Championship Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("""
    ## About This Dashboard
    
    Welcome to our Fantasy Football League Dashboard! This dashboard tracks 15 years of league history.
    
    **Features:**
    - All-time standings and records
    - Head-to-head analysis
    - Historical trends and statistics
    - Championship history
    
    **Coming Soon:**
    - Live ESPN data integration
    - Elo ratings over time
    - Luck analysis
    - Weekly matchup predictions
    
    *Dashboard built with Streamlit*
    """)

st.sidebar.success("✅ Dashboard is live!")
st.sidebar.info("This is a dummy dashboard. Real data coming soon!")