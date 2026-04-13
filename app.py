import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

# ── Brand Colors (exact from notebook) ───────────────────────────────────
TL  = '#9EC3C7'; TM  = '#4E7D82'; GM  = '#6E6969'
DC  = '#474440'; RD  = '#902E28'; GLD = '#C8A436'
CRM = '#FFFCE8'; PUR = '#7B2FBE'

DATA = "data/"

st.set_page_config(
    page_title="Maryland TOD — AI Dashboard",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fix white-on-white: dark background with white text
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #1a1a2e;
}
[data-testid="stSidebar"] {
    background-color: #16213e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
.main .block-container {
    background-color: #1a1a2e;
    color: #f0f0f0;
}
h1, h2, h3, p, li, label, div {
    color: #f0f0f0 !important;
}
.stMetric label { color: #9EC3C7 !important; }
.stMetric [data-testid="metric-container"] { 
    background-color: #16213e; 
    border: 1px solid #4E7D82;
    border-radius: 8px;
    padding: 10px;
}
.stMetric [data-testid="stMetricValue"] { color: #C8A436 !important; }
.stDataFrame { background-color: #16213e; }
.stTabs [data-baseweb="tab"] { color: #9EC3C7 !important; }
.stTabs [aria-selected="true"] { 
    border-bottom: 3px solid #C8A436 !important;
    color: #C8A436 !important;
}
.stRadio label { color: #e0e0e0 !important; }
.winner-badge {
    background: linear-gradient(135deg, #C8A436, #902E28);
    color: white; padding: 10px 18px; border-radius: 20px;
    font-weight: bold; display: inline-block; margin-bottom: 12px;
    font-size: 13px;
}
.stSpinner { color: #9EC3C7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def merc_to_ll(x, y):
    lon = x / 20037508.34 * 180
    lat = np.degrees(2 * np.arctan(np.exp(y / 20037508.34 * np.pi)) - np.pi / 2)
    return lat, lon

def parse_point(g):
    n = re.findall(r'[-\d.]+', str(g))
    if len(n) >= 2:
        return float(n[1]), float(n[0])
    return np.nan, np.nan

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_marc():
    df = pd.read_csv(DATA + "Maryland_Transit_-_MARC_Trains_Stations.csv")
    df["lat"], df["lon"] = zip(*df.apply(lambda r: merc_to_ll(r["X"], r["Y"]), axis=1))
    df["Avg_Wkdy"] = pd.to_numeric(df["Avg_Wkdy"], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_marc_v2():
    df = pd.read_csv(DATA + "MARC_Trains_Stations.csv")
    df["lat"], df["lon"] = zip(*df["the_geom"].apply(parse_point))
    df["riders"] = pd.to_numeric(df.get("Average Weekday Ridership", pd.Series([0]*len(df))), errors="coerce").fillna(0)
    return df

@st.cache_data
def load_aadt_points():
    df = pd.read_csv(DATA + "Annual_Average_Daily_Traffic_-_MDOT_SHA_Statewide_AADT_Points.csv",
                     low_memory=False)
    if "the_geom" in df.columns:
        df["lat"], df["lon"] = zip(*df["the_geom"].apply(parse_point))
    col_map = {}
    for c in df.columns:
        if "AADT" in c and "Current" in c and "AAWDT" not in c:
            col_map[c] = "AADT Current"
        if "Car" in c and "AADT" in c:
            col_map[c] = "AADT Car"
        if "Bus" in c and "AADT" in c:
            col_map[c] = "AADT Bus"
        if "Rural" in c:
            col_map[c] = "Rural / Urban"
        if "County" in c and "Name" in c:
            col_map[c] = "County Name"
        if "Road" in c and "Name" in c:
            col_map[c] = "Road Name"
    df = df.rename(columns=col_map)
    if "AADT Current" not in df.columns:
        for c in df.columns:
            if "AADT" in c and df[c].dtype in [np.float64, np.int64]:
                df["AADT Current"] = pd.to_numeric(df[c], errors="coerce").fillna(0)
                break
    if "AADT Current" not in df.columns:
        df["AADT Current"] = 0
    df["AADT Current"] = pd.to_numeric(df["AADT Current"], errors="coerce").fillna(0)
    if "Rural / Urban" not in df.columns:
        df["Rural / Urban"] = "Urban"
    if "County Name" not in df.columns:
        df["County Name"] = "Unknown"
    if "Road Name" not in df.columns:
        df["Road Name"] = "Unknown"
    df = df.dropna(subset=["lat", "lon"])
    return df

@st.cache_data
def load_qcew():
    df = pd.read_csv(DATA + "Quarterly_Census_of_Employment_and_Wages_-_Statewide_and_All_Counties.csv")
    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_emp_sector():
    df = pd.read_csv(DATA + "2006 to 2025 employment by sector.csv")
    date_cols = [c for c in df.columns if c != "Series title"]
    emp_long = df.melt(id_vars="Series title", value_vars=date_cols,
                       var_name="Date", value_name="Value")
    emp_long["Date"]  = pd.to_datetime(emp_long["Date"], errors="coerce")
    emp_long["Value"] = pd.to_numeric(emp_long["Value"], errors="coerce")
    emp_long["Year"]  = emp_long["Date"].dt.year
    return emp_long

@st.cache_data
def load_trips():
    df = pd.read_csv(DATA + "Passenger_Trips_Per_Revenue_Vehicle_Mile.csv")
    df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")
    for c in ["MARC", "Metro", "Core Bus", "Light Rail"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data
def load_emp_ue():
    df = pd.read_csv(DATA + "Employment__Unemployment__and_Labor_Force_Data.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Employed"] = pd.to_numeric(df["Employed"], errors="coerce")
    df["Unemployment Rate"] = pd.to_numeric(df["Unemployment Rate"], errors="coerce")
    return df

@st.cache_data
def load_bls():
    df = pd.read_csv(DATA + "BLS_Jobs_by_Industry_Category.csv")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

@st.cache_data
def load_hospitals():
    df = pd.read_csv(DATA + "Maryland_Hospitals_-_Hospitals.csv")
    df["lat"], df["lon"] = zip(*df.apply(lambda r: merc_to_ll(r["X"], r["Y"]), axis=1))
    df["License_Capacity"] = pd.to_numeric(df["License_Capacity"], errors="coerce").fillna(0)
    return df

@st.cache_data
def load_ontime():
    df = pd.read_csv(DATA + "On_Time_Performance_20260327.csv")
    df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="winner-badge">🏆 1st Place — UMD SAC Datathon 2026</div>',
                unsafe_allow_html=True)
    st.markdown("### Maryland TOD AI Dashboard")
    st.markdown("*Transit-Oriented Development Strategy*")
    st.markdown("---")
    st.markdown("**Sponsor:** Maryland DoIT + Deloitte")
    st.markdown("**Team:** Eshan & Team")
    st.markdown("**Date:** March 27, 2026")
    st.markdown("---")
    page = st.radio("Navigate to:", [
        "🏠 Overview",
        "🤖 AI Feature 1: Corridor Scoring",
        "📈 AI Feature 2: Demand Forecasting",
        "⚖️ AI Feature 3: Equity Recommender",
        "⚠️ AI Feature 4: Conflict Detection",
    ])
    st.markdown("---")
    st.markdown("**Datasets:** 12 Maryland gov datasets")
    st.markdown("**AI:** K-Means · PCA · Monte Carlo · Cosine Similarity · Haversine")

# ════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🚆 Maryland Transit-Oriented Development")
    st.subheader("AI-Powered Investment Strategy — UMD SAC 7th Annual Datathon 2026")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Result", "🏆 1st Place")
    c2.metric("Corridors Analyzed", "21")
    c3.metric("Datasets Used", "12")
    c4.metric("AI Models Built", "4")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **The Problem:**
        Maryland's MARC commuter rail and BRT network has a ridership gap.
        Low-income communities lack transit access to employment centers.

        **Our Solution:**
        4 AI systems built on top of standard consulting analysis —
        all running on real Maryland government data.
        """)
    with col_b:
        st.markdown("""
        | AI Feature | Algorithm |
        |---|---|
        | 🎯 Corridor Scoring | K-Means + PCA + Weighted Scoring |
        | 📈 Demand Forecasting | Poly Regression + Monte Carlo 5K |
        | ⚖️ Equity Recommender | Content Filtering + Cosine Similarity |
        | ⚠️ Conflict Detection | Haversine + Rule-Based AI Agent |
        """)

    st.markdown("---")
    marc   = load_marc()
    trips  = load_trips()
    ontime = load_ontime()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MARC Station Ridership — Top 10**")
        top = marc.nlargest(10, "Avg_Wkdy")[["Name", "Line_Name", "CITY", "Avg_Wkdy"]]
        top.columns = ["Station", "Line", "City", "Avg Weekday Riders"]
        st.dataframe(top, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Ridership Trends by Mode**")
        mode_cols = [c for c in ["MARC", "Metro", "Core Bus", "Light Rail"] if c in trips.columns]
        colors_m  = [GLD, TL, TM, RD]
        fig = go.Figure()
        for i, m in enumerate(mode_cols):
            df_m = trips.dropna(subset=["Fiscal Year", m])
            fig.add_trace(go.Scatter(x=df_m["Fiscal Year"], y=df_m[m],
                name=m, line=dict(color=colors_m[i], width=2.5), mode="lines+markers"))
        fig.update_layout(height=320, margin=dict(t=20, b=20),
                          plot_bgcolor=CRM, paper_bgcolor="white",
                          font=dict(family="Arial", size=10),
                          yaxis_title="Trips/Revenue Mile",
                          legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### On-Time Performance by Mode")
    otp_cols = [c for c in ["MARC", "Metro", "Core Bus", "Light Rail"] if c in ontime.columns]
    fig2 = go.Figure()
    for i, m in enumerate(otp_cols):
        df_m = ontime.dropna(subset=["Fiscal Year", m]).copy()
        df_m[m] = pd.to_numeric(df_m[m], errors="coerce")
        fig2.add_trace(go.Scatter(x=df_m["Fiscal Year"], y=df_m[m],
            name=m, line=dict(color=colors_m[i], width=2.5)))
    fig2.update_layout(height=300, plot_bgcolor=CRM, paper_bgcolor="white",
                       font=dict(family="Arial", size=10),
                       yaxis_title="On-Time %", margin=dict(t=20, b=20),
                       legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# AI FEATURE 1 — CORRIDOR SCORING (exact notebook code)
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Feature 1: Corridor Scoring":
    st.title("🤖 AI Feature 1: TOD Investment Priority Scoring Engine")
    st.markdown("**Algorithms:** Weighted Multi-Factor Scoring · K-Means Clustering (k=4) · PCA")
    st.markdown("Evaluates **21 corridors** across **12 real features** (MARC ridership anchored to real data).")
    st.markdown("---")

    with st.spinner("Running ML scoring model on real data..."):
        marc = load_marc()
        qcew = load_qcew()

        corridors = pd.DataFrame({
            'corridor_id': range(1, 22),
            'name': [
                'Silver Spring → Greenbelt (US-1)',
                'Baltimore Penn → Odenton (MARC Penn)',
                'Rockville → Germantown (I-270 BRT)',
                'Baltimore → BWI → Odenton (MD-170)',
                'College Park → New Carrollton',
                'Frederick → Germantown (I-270 Ext)',
                'Annapolis → BWI → New Carrollton',
                'West Baltimore → Downtown (MARC+)',
                'Gaithersburg → Shady Grove',
                'Laurel → Bowie → New Carrollton',
                'Towson → Baltimore Downtown (BRT)',
                'Columbia → BWI → Baltimore (BRT)',
                'Greenbelt → Branch Ave (BRT)',
                'Germantown → Rockville → Metro',
                'Edgewood → Baltimore (MARC Penn)',
                'Odenton → Annapolis (BRT)',
                'Hagerstown → Frederick (MARC Ext)',
                'Elkton → Aberdeen (MARC Penn Ext)',
                'Waldorf → Branch Ave (BRT)',
                'Bowie → College Park (BRT)',
                'Sphinx Loop (all 4 hubs)',
            ],
            'nearest_marc_riders': [
                marc[marc['Name'].str.contains('Silver Spring', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('Pennsylvania', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('Rockville', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('BWI', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('College Park', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('Frederick', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('New Carrollton', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('West Baltimore', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('Gaithersburg', na=False)]['Avg_Wkdy'].sum(),
                marc[marc['Name'].str.contains('Laurel', na=False)]['Avg_Wkdy'].sum(),
                200, 400, 85, 300, 148, 1401, 99, 110, 55, 338, 8000,
            ],
            'jobs_within_1_5mi_k':   [68,82,45,52,55,38,41,62,49,36,58,44,32,40,24,28,18,15,29,33,95],
            'employment_growth_pct': [8.2,6.1,9.4,7.8,7.2,11.3,6.5,3.2,8.8,7.0,5.4,8.1,6.2,8.9,2.8,5.5,3.1,2.2,9.6,7.4,9.8],
            'high_wage_jobs_pct':    [0.72,0.68,0.65,0.58,0.75,0.55,0.62,0.48,0.70,0.60,0.52,0.60,0.55,0.65,0.42,0.50,0.38,0.35,0.55,0.63,0.68],
            'transit_gap_score':     [6,3,8,5,4,9,7,5,7,6,4,6,8,7,5,7,10,10,9,7,3],
            'aadt_corridor_k':       [88,112,145,95,76,72,68,82,138,64,95,88,58,110,52,62,42,38,78,65,110],
            'congestion_hrs_peak':   [3.2,2.8,4.1,3.5,2.4,2.1,2.6,2.9,3.8,2.2,2.5,3.1,2.0,3.6,1.8,2.3,1.5,1.2,3.0,2.4,3.8],
            'low_income_pct':        [0.32,0.22,0.18,0.28,0.30,0.22,0.25,0.55,0.20,0.35,0.38,0.24,0.42,0.21,0.40,0.28,0.45,0.48,0.44,0.32,0.38],
            'zero_car_pct':          [0.28,0.15,0.12,0.18,0.25,0.14,0.16,0.42,0.13,0.20,0.30,0.16,0.35,0.14,0.32,0.18,0.38,0.40,0.36,0.22,0.30],
            'cost_estimate_M':       [145,285,98,112,65,180,155,92,88,75,110,135,80,72,55,95,340,290,125,85,68],
            'row_difficulty':        [4,2,6,3,2,7,5,3,5,4,5,6,4,5,2,4,8,7,6,4,2],
            'timeline_months':       [20,18,24,22,16,30,26,18,22,20,20,24,18,20,15,22,42,38,28,20,14],
            'lat': [38.985,39.298,39.130,39.190,38.963,39.290,38.978,39.293,39.155,39.095,
                    39.350,39.180,38.920,39.115,39.420,39.090,39.640,39.540,38.640,38.990,39.150],
            'lon': [-76.971,-76.634,-77.231,-76.695,-76.900,-77.270,-76.872,-76.653,-77.210,-76.843,
                    -76.615,-76.750,-76.870,-77.190,-76.300,-76.740,-77.720,-76.100,-76.890,-76.878,-76.750],
        })

        corridors['nearest_marc_riders'] = pd.to_numeric(corridors['nearest_marc_riders'], errors='coerce').fillna(100)

        feature_cols = ['jobs_within_1_5mi_k','employment_growth_pct','high_wage_jobs_pct',
                        'nearest_marc_riders','transit_gap_score','aadt_corridor_k',
                        'congestion_hrs_peak','low_income_pct','zero_car_pct',
                        'cost_estimate_M','row_difficulty','timeline_months']

        weights = {'jobs_within_1_5mi_k':0.18,'employment_growth_pct':0.10,'high_wage_jobs_pct':0.07,
                   'nearest_marc_riders':0.10,'transit_gap_score':0.12,'aadt_corridor_k':0.10,
                   'congestion_hrs_peak':0.08,'low_income_pct':0.10,'zero_car_pct':0.08,
                   'cost_estimate_M':-0.08,'row_difficulty':-0.06,'timeline_months':-0.04}

        X      = corridors[feature_cols].fillna(0).values
        scaler = MinMaxScaler()
        X_s    = scaler.fit_transform(X)

        raw_score = np.zeros(len(corridors))
        for i, col in enumerate(feature_cols):
            w = weights[col]
            raw_score += (1 - X_s[:, i]) * abs(w) if w < 0 else X_s[:, i] * w

        corridors['ai_score'] = ((raw_score - raw_score.min()) /
                                  (raw_score.max() - raw_score.min()) * 100).round(1)

        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        corridors['cluster'] = km.fit_predict(X_s)
        cluster_means = corridors.groupby('cluster')['ai_score'].mean().sort_values(ascending=False)
        tier_map    = {c: t for t, (c, _) in enumerate(cluster_means.items())}
        tier_labels = {0: 'Invest Now', 1: 'High Priority', 2: 'Plan Later', 3: 'Monitor'}
        tier_colors = {0: RD, 1: GLD, 2: TM, 3: TL}
        corridors['tier']       = corridors['cluster'].map(tier_map)
        corridors['tier_label'] = corridors['tier'].map(tier_labels)
        corridors['tier_color'] = corridors['tier'].map(tier_colors)
        corridors = corridors.sort_values('ai_score', ascending=False)

        pca   = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_s)
        corridors['pca1'] = X_pca[:, 0]
        corridors['pca2'] = X_pca[:, 1]
        var_exp = pca.explained_variance_ratio_

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Corridor", corridors.iloc[0]['name'].split('(')[0].strip())
    c2.metric("Top AI Score", f"{corridors.iloc[0]['ai_score']:.1f}/100")
    c3.metric("Invest Now Corridors", len(corridors[corridors['tier_label'] == 'Invest Now']))
    c4.metric("PCA Variance Explained", f"{sum(var_exp)*100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ranked Leaderboard", "🔵 PCA Clusters", "🕸️ Feature Radar", "💰 Efficiency Frontier"])

    with tab1:
        fig_rank = go.Figure()
        for tier in [0, 1, 2, 3]:
            sub = corridors[corridors['tier'] == tier].sort_values('ai_score')
            fig_rank.add_trace(go.Bar(
                x=sub['ai_score'], y=sub['name'],
                orientation='h', name=tier_labels[tier],
                marker_color=tier_colors[tier], opacity=0.88,
                text=sub['ai_score'].apply(lambda v: f'{v:.1f}'),
                textposition='outside',
            ))
        fig_rank.add_vline(x=75, line_dash='dash', line_color=DC, line_width=1.5,
                           annotation_text='High Priority Threshold',
                           annotation_font=dict(size=9, color=DC))
        fig_rank.update_layout(
            title='<b>AI TOD Investment Ranking — All 21 Corridors</b><br>'
                  '<sup>Weighted ML score across 12 features | Real MARC ridership data | Sphinx Loop top-ranked</sup>',
            barmode='stack', height=680,
            plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=10),
            xaxis=dict(title='AI Investment Score (0–100)', range=[0, 115], gridcolor='#e0e0e0'),
            yaxis=dict(tickfont=dict(size=9)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
            margin=dict(l=260, r=60),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    with tab2:
        fig_pca = px.scatter(corridors, x='pca1', y='pca2',
            color='tier_label', text='name',
            color_discrete_map={v: tier_colors[k] for k, v in tier_labels.items()},
            size='ai_score', size_max=28,
            title=f'<b>K-Means Investment Tiers — PCA Visualization</b><br>'
                  f'<sup>PC1 explains {var_exp[0]:.0%} | PC2 {var_exp[1]:.0%} of variance | 4 data-driven clusters</sup>',
            labels={'pca1': f'PC1 ({var_exp[0]:.0%} variance)', 'pca2': f'PC2 ({var_exp[1]:.0%} variance)'}
        )
        fig_pca.update_traces(textposition='top center', textfont=dict(size=7))
        fig_pca.update_layout(
            height=540, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=11),
            xaxis=dict(gridcolor='#e0e0e0'), yaxis=dict(gridcolor='#e0e0e0'),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab3:
        top3 = corridors.head(3)
        radar_features = ['jobs_within_1_5mi_k','transit_gap_score','low_income_pct',
                          'aadt_corridor_k','nearest_marc_riders','congestion_hrs_peak']
        radar_labels   = ['Jobs Density','Transit Gap','Equity Need','Traffic Vol','MARC Ridership','Congestion']
        radar_colors   = [RD, GLD, TM]
        fig_radar      = go.Figure()
        X_r = scaler.transform(corridors[feature_cols].fillna(0).values)
        idx_map = {name: i for i, name in enumerate(corridors.index)}
        for idx_c, (orig_idx, row) in enumerate(top3.iterrows()):
            pos = list(corridors.index).index(orig_idx)
            col_idx = [feature_cols.index(f) for f in radar_features]
            vals = [X_r[pos, ci] * 100 for ci in col_idx]
            vals += [vals[0]]
            cats  = radar_labels + [radar_labels[0]]
            r, g, b = int(radar_colors[idx_c][1:3], 16), int(radar_colors[idx_c][3:5], 16), int(radar_colors[idx_c][5:7], 16)
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself',
                name=row['name'].split('(')[0].strip()[:35],
                line=dict(color=radar_colors[idx_c], width=2),
                fillcolor=f'rgba({r},{g},{b},0.15)',
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9), gridcolor='#ddd'),
                       angularaxis=dict(tickfont=dict(size=10))),
            title='<b>Feature Profile — Top 3 Corridors</b><br>'
                  '<sup>Higher = more important | Real MARC data anchors MARC Ridership axis</sup>',
            height=500, paper_bgcolor='white', font=dict(family='Arial', size=11),
            legend=dict(orientation='h', yanchor='bottom', y=-0.08, xanchor='center', x=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        fig_eff = px.scatter(corridors, x='cost_estimate_M', y='ai_score',
            color='tier_label', size='jobs_within_1_5mi_k', text='name', size_max=30,
            color_discrete_map={v: tier_colors[k] for k, v in tier_labels.items()},
            title='<b>Efficiency Frontier — Cost vs AI Score</b><br>'
                  '<sup>Upper-left = best ROI | Bubble size = jobs density | Sphinx Loop: low cost + top score</sup>',
            labels={'cost_estimate_M': 'Estimated Cost ($M)', 'ai_score': 'AI Investment Score'}
        )
        fig_eff.update_traces(textposition='top center', textfont=dict(size=7))
        fig_eff.update_layout(
            height=520, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=11),
            xaxis=dict(title='Estimated Cost ($M)', gridcolor='#e0e0e0'),
            yaxis=dict(title='AI Score (0–100)', gridcolor='#e0e0e0'),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        st.plotly_chart(fig_eff, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# AI FEATURE 2 — DEMAND FORECASTING (exact notebook code)
# ════════════════════════════════════════════════════════════════════
elif page == "📈 AI Feature 2: Demand Forecasting":
    st.title("📈 AI Feature 2: Ridership Demand Forecasting Engine")
    st.markdown("**Algorithms:** Polynomial Regression · Employment-Driven Linear Regression · Monte Carlo (5,000 runs) · S-Curve Adoption")
    st.markdown("---")

    with st.spinner("Running forecasting models..."):
        trips    = load_trips()
        emp_long = load_emp_sector()
        emp_ue   = load_emp_ue()
        bls      = load_bls()

        emp_annual = emp_ue.groupby('Year').agg({'Employed': 'mean', 'Unemployment Rate': 'mean'}).reset_index()

        key_sectors = ['Total Nonfarm ', 'Government ', 'Professional and Business Services ', 'Education and Health Services ']
        emp_sector_ann = emp_long[emp_long['Series title'].isin(key_sectors)]\
            .groupby(['Series title', 'Year'])['Value'].mean().reset_index()

        sector_2006_2025 = emp_sector_ann[emp_sector_ann['Series title'] == 'Total Nonfarm ']
        sector_2006_2025 = sector_2006_2025[sector_2006_2025['Year'].between(2006, 2025)]

        tot_2019 = emp_long[(emp_long['Series title'] == 'Total Nonfarm ') & (emp_long['Year'] == 2019)]['Value'].mean()
        tot_2024 = emp_long[(emp_long['Series title'] == 'Total Nonfarm ') & (emp_long['Year'] == 2024)]['Value'].mean()
        real_growth = ((tot_2024 / tot_2019)**(1/5) - 1) if (tot_2019 and tot_2019 > 0) else 0.018

        trips_m = trips.copy()
        trips_m['Year'] = trips_m['Fiscal Year']
        trips_m = trips_m.merge(emp_annual, on='Year', how='left')

        def make_forecast_poly(series_vals, year_vals, n_future=10):
            X  = np.array(year_vals).reshape(-1, 1)
            y  = np.array(series_vals, dtype=float)
            poly = PolynomialFeatures(degree=2)
            Xp = poly.fit_transform(X)
            lr = LinearRegression().fit(Xp, y)
            future_years = list(range(int(min(year_vals)), int(max(year_vals)) + n_future + 1))
            Xf = poly.transform(np.array(future_years).reshape(-1, 1))
            yp  = lr.predict(Xf)
            std = np.std(y - lr.predict(Xp))
            return pd.DataFrame({'year': future_years, 'yhat': yp,
                                 'yhat_lower': yp - 1.96 * std, 'yhat_upper': yp + 1.96 * std})

        modes     = {'MARC': GLD, 'Metro': TL, 'Core Bus': TM}
        forecasts = {}
        for mode in modes:
            if mode in trips.columns:
                valid = trips.dropna(subset=['Fiscal Year', mode])
                forecasts[mode] = make_forecast_poly(valid[mode].tolist(), valid['Fiscal Year'].tolist())

    c1, c2, c3 = st.columns(3)
    c1.metric("MD Employment Growth (2019–2024)", f"{real_growth*100:.2f}%/yr")
    c2.metric("Monte Carlo Simulations", "5,000")
    c3.metric("Forecast Horizon", "2026–2035")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Ridership Forecast", "💼 Employment Driver", "🎲 Monte Carlo", "📈 S-Curve Scenarios"])

    with tab1:
        fig_fc = make_subplots(rows=1, cols=3,
            subplot_titles=[f'{m} — Trips/Rev-Mile' for m in modes.keys()],
            horizontal_spacing=0.08)
        targets = {'MARC': 2.2, 'Metro': 3.5, 'Core Bus': 4.5}
        for col_idx, (mode, color) in enumerate(modes.items(), 1):
            if mode not in forecasts: continue
            fc      = forecasts[mode]
            hist_y  = trips.dropna(subset=['Fiscal Year', mode])['Fiscal Year'].tolist()
            hist_v  = trips.dropna(subset=['Fiscal Year', mode])[mode].tolist()
            fut_mask = fc['year'] > max(hist_y)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig_fc.add_trace(go.Scatter(
                x=list(fc[fut_mask]['year']) + list(fc[fut_mask]['year'])[::-1],
                y=list(fc[fut_mask]['yhat_upper']) + list(fc[fut_mask]['yhat_lower'])[::-1],
                fill='toself', fillcolor=f'rgba({r},{g},{b},0.12)',
                line=dict(width=0), showlegend=False
            ), row=1, col=col_idx)
            fig_fc.add_trace(go.Scatter(
                x=hist_y, y=hist_v, mode='markers+lines',
                marker=dict(size=8, color=color), line=dict(color=color, width=2.5),
                name=f'{mode} (actual)', showlegend=(col_idx == 1)
            ), row=1, col=col_idx)
            fig_fc.add_trace(go.Scatter(
                x=fc['year'], y=fc['yhat'], mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f'{mode} forecast', showlegend=(col_idx == 1)
            ), row=1, col=col_idx)
            fig_fc.add_hline(y=targets[mode], line_dash='dash', line_color=DC, line_width=1.5,
                             annotation_text=f'Target {targets[mode]}',
                             annotation_font_size=8, row=1, col=col_idx)
        fig_fc.update_layout(
            title='<b>AI Ridership Forecast — MARC · Metro · Core Bus</b><br>'
                  '<sup>Polynomial regression | 95% CI shaded | Colors match network maps</sup>',
            height=440, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        fig_fc.update_xaxes(gridcolor='#e0e0e0', dtick=2)
        fig_fc.update_yaxes(gridcolor='#e0e0e0')
        st.plotly_chart(fig_fc, use_container_width=True)

    with tab2:
        future_years      = list(range(2025, 2036))
        last_nonfarm_vals = sector_2006_2025[sector_2006_2025['Year'] == sector_2006_2025['Year'].max()]['Value'].values
        base_nonfarm      = float(last_nonfarm_vals[0]) if len(last_nonfarm_vals) > 0 else 2800
        proj_nonfarm      = [base_nonfarm * (1.018)**i for i in range(len(future_years))]
        proj_nonfarm_tod  = [v * (1 + 0.03 * min(i, 5)) for i, v in enumerate(proj_nonfarm)]

        r2_val = 0.0
        trips_emp = trips_m[trips_m['MARC'].notna()].copy() if 'MARC' in trips_m.columns else pd.DataFrame()
        if len(trips_emp) > 3 and 'Employed' in trips_emp.columns:
            X_e  = trips_emp['Employed'].values.reshape(-1, 1)
            y_e  = trips_emp['MARC'].values
            lr_e = LinearRegression().fit(X_e, y_e)
            r2_val = r2_score(y_e, lr_e.predict(X_e))

        fig_emp_d = make_subplots(rows=1, cols=2,
            subplot_titles=['MD Total Nonfarm Employment 2006–2035', f'MARC Efficiency vs Employment (R²={r2_val:.2f})'],
            horizontal_spacing=0.12)
        fig_emp_d.add_trace(go.Scatter(
            x=sector_2006_2025['Year'], y=sector_2006_2025['Value'],
            mode='lines+markers', name='Actual (2006–2025)',
            line=dict(color=TM, width=2.5), marker=dict(size=5)
        ), row=1, col=1)
        fig_emp_d.add_trace(go.Scatter(
            x=future_years, y=proj_nonfarm, mode='lines', name='Forecast (no TOD)',
            line=dict(color=GM, width=2, dash='dash')
        ), row=1, col=1)
        fig_emp_d.add_trace(go.Scatter(
            x=future_years, y=proj_nonfarm_tod, mode='lines', name='Forecast (with TOD)',
            line=dict(color=GLD, width=2.5),
            fill='tonexty', fillcolor='rgba(200,164,54,0.10)'
        ), row=1, col=1)
        if len(trips_emp) > 3 and 'Employed' in trips_emp.columns:
            fig_emp_d.add_trace(go.Scatter(
                x=trips_emp['Employed'], y=trips_emp['MARC'],
                mode='markers', name='MARC (actual)',
                marker=dict(size=9, color=GLD, opacity=0.9)
            ), row=1, col=2)
            X_line = np.linspace(trips_emp['Employed'].min(), trips_emp['Employed'].max(), 50)
            fig_emp_d.add_trace(go.Scatter(
                x=X_line, y=lr_e.predict(X_line.reshape(-1, 1)),
                mode='lines', name=f'Linear fit (R²={r2_val:.2f})',
                line=dict(color=TM, width=2, dash='dot')
            ), row=1, col=2)
        fig_emp_d.update_layout(
            title=f'<b>Employment-Driven Demand Model — 2006–2035</b><br>'
                  f'<sup>Real BLS data | MARC R²={r2_val:.2f} | TOD adds ~3%/yr uplift</sup>',
            height=440, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        fig_emp_d.update_yaxes(gridcolor='#e0e0e0')
        fig_emp_d.update_xaxes(gridcolor='#e0e0e0')
        st.plotly_chart(fig_emp_d, use_container_width=True)
        st.success(f"R² = {r2_val:.3f} — employment explains {r2_val*100:.1f}% of MARC ridership variance.")

    with tab3:
        np.random.seed(42)
        N = 5000
        fare_per_rider = np.random.normal(5.20, 0.80, N)
        adoption_yr3   = np.random.triangular(0.12, 0.22, 0.35, N)
        annual_growth  = np.random.normal(0.04, 0.015, N)
        op_cost_rider  = np.random.normal(3.80, 0.60, N)
        sphinx_premium = np.random.normal(0.85, 0.15, N)
        total_base_riders = 41700 + 8000
        yr5_revenue = total_base_riders * adoption_yr3 * (fare_per_rider + sphinx_premium) * 365 / 1e6
        yr5_cost    = total_base_riders * adoption_yr3 * op_cost_rider * 365 / 1e6
        yr5_net     = yr5_revenue - yr5_cost
        p5, p50, p95 = np.percentile(yr5_net, [5, 50, 95])

        c1, c2, c3 = st.columns(3)
        c1.metric("Median Net Revenue (Yr 5)", f"${p50:.0f}M")
        c2.metric("5th Percentile (Downside)", f"${p5:.0f}M")
        c3.metric("95th Percentile (Upside)",  f"${p95:.0f}M")

        fig_mc = make_subplots(rows=1, cols=2,
            subplot_titles=['Year 5 Net Revenue (5,000 scenarios)', 'Cumulative Break-Even Probability'],
            horizontal_spacing=0.12)
        fig_mc.add_trace(go.Histogram(
            x=yr5_net, nbinsx=60, marker_color=TM, opacity=0.80
        ), row=1, col=1)
        for val, lbl, col in [(p5, '5th pct', RD), (p50, 'Median', GLD), (p95, '95th pct', TL)]:
            fig_mc.add_vline(x=val, line_dash='dash', line_color=col, line_width=2, row=1, col=1,
                             annotation_text=f'{lbl}: ${val:.0f}M',
                             annotation_font=dict(size=9, color=col))
        years_be = list(range(2, 11))
        prob_be  = [np.mean(yr5_net * (y / 5) > 0) for y in years_be]
        fig_mc.add_trace(go.Scatter(
            x=years_be, y=[p * 100 for p in prob_be],
            mode='lines+markers', line=dict(color=GLD, width=3),
            marker=dict(size=8, color=GLD),
            fill='tozeroy', fillcolor='rgba(200,164,54,0.12)'
        ), row=1, col=2)
        fig_mc.add_hline(y=80, line_dash='dash', line_color=TM, line_width=1.5,
                         annotation_text='80% threshold',
                         annotation_font=dict(size=9, color=TM), row=1, col=2)
        fig_mc.update_layout(
            title=f'<b>Monte Carlo Financial Risk Model — {N:,} Scenarios</b><br>'
                  f'<sup>Median Yr5: ${p50:.0f}M | 90% CI: ${p5:.0f}M–${p95:.0f}M | Sphinx Loop premium included</sup>',
            height=440, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=11), showlegend=False,
        )
        fig_mc.update_xaxes(gridcolor='#e0e0e0')
        fig_mc.update_yaxes(gridcolor='#e0e0e0')
        st.plotly_chart(fig_mc, use_container_width=True)

    with tab4:
        months = list(range(1, 37))
        def scurve(m, k, x0, L):
            return [L / (1 + np.exp(-k * (t - x0))) for t in m]
        scenarios = {
            'Optimistic':   (scurve(months, 0.28, 10, 38000), TL,  'dot'),
            'Base+Sphinx':  (scurve(months, 0.22, 14, 36000), GLD, 'solid'),
            'Base':         (scurve(months, 0.20, 15, 28000), TM,  'dash'),
            'Conservative': (scurve(months, 0.15, 18, 20000), GM,  'dash'),
            'Pessimistic':  (scurve(months, 0.10, 22, 15000), DC,  'dot'),
        }
        fig_fan = go.Figure()
        for name, (vals, col, dash) in scenarios.items():
            is_base = 'Base+' in name
            fig_fan.add_trace(go.Scatter(
                x=months, y=[int(v) for v in vals], name=name,
                mode='lines',
                fill='tozeroy' if is_base else None,
                fillcolor='rgba(200,164,54,0.10)' if is_base else None,
                line=dict(color=col, width=3 if is_base else 2, dash=dash),
            ))
        for vx, lbl in [(6, 'Trial launch'), (12, 'Employer program'), (18, 'Sphinx launch'), (24, 'Full BRT')]:
            fig_fan.add_vline(x=vx, line_dash='dash', line_color=TM, line_width=1.5)
            fig_fan.add_annotation(x=vx, y=37000, text=lbl, font=dict(size=8, color=DC), showarrow=False)
        fig_fan.update_layout(
            title='<b>Ridership Scenario Fan — 36 Month Ramp</b><br>'
                  '<sup>5 scenarios | Base+Sphinx adds 8K riders | S-curve adoption model</sup>',
            xaxis=dict(title='Month of Operation', gridcolor='#e0e0e0', dtick=3),
            yaxis=dict(title='Daily New Riders', gridcolor='#e0e0e0', tickformat=','),
            height=440, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
            hovermode='x unified',
        )
        st.plotly_chart(fig_fan, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# AI FEATURE 3 — EQUITY RECOMMENDER (exact notebook code)
# ════════════════════════════════════════════════════════════════════
elif page == "⚖️ AI Feature 3: Equity Recommender":
    st.title("⚖️ AI Feature 3: Equity Impact Recommender System")
    st.markdown("**Algorithms:** Content-Based Filtering + Cosine Similarity to ideal high-need profile")
    st.markdown("---")

    with st.spinner("Running equity recommender..."):
        qcew = load_qcew()
        qcew = qcew.dropna(subset=['Total', 'Year'])

        sector_cols = ['Total', 'Total Private', 'Educational services',
                       'Health care and social assistance', 'Professional and technical services',
                       'Public Administration', 'Retail Trade', 'Transportation and Warehousing',
                       'Construction', 'Manufacturing']
        for c in sector_cols:
            if c in qcew.columns:
                qcew[c] = pd.to_numeric(qcew[c], errors='coerce').fillna(0)

        latest_year = qcew['Year'].max()
        latest_q    = qcew[(qcew['Year'] == latest_year)]
        agg_dict    = {c: 'sum' for c in sector_cols if c in qcew.columns}
        county_sectors = latest_q.groupby('County').agg(agg_dict).reset_index()
        county_sectors = county_sectors[county_sectors['County'] != 'Statewide'].fillna(0)

        earliest  = qcew['Year'].min()
        emp_start = qcew[qcew['Year'] == earliest].groupby('County')['Total'].sum().reset_index().rename(columns={'Total': 'emp_start'})
        emp_lat   = county_sectors[['County', 'Total']].rename(columns={'Total': 'emp_latest'})
        emp_growth = emp_start.merge(emp_lat, on='County', how='inner')
        emp_growth['growth_rate'] = ((emp_growth['emp_latest'] - emp_growth['emp_start']) /
                                     emp_growth['emp_start'].replace(0, np.nan)).fillna(0)
        county_sectors = county_sectors.merge(emp_growth[['County', 'growth_rate']], on='County', how='left').fillna(0)

        county_sectors['high_wage_job_pct'] = (
            (county_sectors.get('Professional and technical services', 0) +
             county_sectors.get('Educational services', 0) +
             county_sectors.get('Public Administration', 0)) /
            county_sectors['Total'].replace(0, np.nan)
        ).fillna(0)
        county_sectors['service_job_pct'] = (
            (county_sectors.get('Retail Trade', 0) +
             county_sectors.get('Health care and social assistance', 0)) /
            county_sectors['Total'].replace(0, np.nan)
        ).fillna(0)
        county_sectors['total_jobs_k'] = county_sectors['Total'] / 1000

        equity_features = ['total_jobs_k', 'service_job_pct', 'growth_rate',
                           'Transportation and Warehousing', 'Manufacturing']
        equity_features = [f for f in equity_features if f in county_sectors.columns]

        cs_clean = county_sectors[county_sectors['Total'] > 0].copy()
        cs_clean[equity_features] = cs_clean[equity_features].fillna(0)

        X_eq   = cs_clean[equity_features].values
        scaler = MinMaxScaler()
        X_s    = scaler.fit_transform(X_eq)

        eq_weights = np.array([0.20, 0.22, -0.15, 0.18, 0.12])[:len(equity_features)]
        X_w = X_s.copy()
        for i, w in enumerate(eq_weights):
            X_w[:, i] = (1 - X_s[:, i]) * abs(w) if w < 0 else X_s[:, i] * w
        cs_clean['equity_score'] = (X_w.sum(axis=1) * 100).round(1)

        ideal   = np.array([0.8, 0.85, 0.1, 0.9, 0.7])[:len(equity_features)]
        ideal_n = ideal / np.linalg.norm(ideal)
        cs_clean['cosine_sim'] = [np.dot(r / (np.linalg.norm(r) + 1e-10), ideal_n) for r in X_s]
        cs_clean['combined_score'] = (cs_clean['equity_score'] * 0.6 + cs_clean['cosine_sim'] * 40).round(1)

        rec_colors = {
            'URGENT: Heavy Rail or BRT':           RD,
            'HIGH: BRT + Equity Stations':         GLD,
            'MEDIUM: Enhanced Bus + Microtransit': TM,
            'LOW: Park & Ride + Connections':      TL,
        }

        def recommend(row):
            s   = row['combined_score']
            svc = row['service_job_pct']
            j   = row['total_jobs_k']
            if s >= 65 and j > 200:   return 'URGENT: Heavy Rail or BRT'
            elif s >= 55 and svc > 0.35: return 'HIGH: BRT + Equity Stations'
            elif s >= 45:              return 'MEDIUM: Enhanced Bus + Microtransit'
            else:                      return 'LOW: Park & Ride + Connections'

        cs_clean['recommendation'] = cs_clean.apply(recommend, axis=1)
        cs_clean = cs_clean.sort_values('combined_score', ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("#1 Equity Need", cs_clean.iloc[0]['County'])
    c2.metric("#2 Equity Need", cs_clean.iloc[1]['County'])
    c3.metric("Counties Scored", len(cs_clean))

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Equity Leaderboard", "📊 Sector Breakdown", "📐 Equity Quadrant", "🗺️ County Map"])

    with tab1:
        df_s = cs_clean.sort_values('combined_score', ascending=True)
        fig_eq_bar = go.Figure()
        for rec, color in rec_colors.items():
            sub = df_s[df_s['recommendation'] == rec]
            if sub.empty: continue
            fig_eq_bar.add_trace(go.Bar(
                x=sub['combined_score'], y=sub['County'],
                orientation='h', name=rec, marker_color=color, opacity=0.88,
                text=sub['combined_score'].apply(lambda v: f'{v:.1f}'),
                textposition='outside',
            ))
        fig_eq_bar.update_layout(
            title='<b>AI Equity Recommender — Maryland County Rankings</b><br>'
                  '<sup>Cosine similarity to ideal high-need profile | QCEW sector data</sup>',
            barmode='stack', height=640,
            xaxis=dict(title='Equity Impact Score (0–100)', range=[0, 115], gridcolor='#e0e0e0'),
            plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
            margin=dict(l=180, r=80),
        )
        st.plotly_chart(fig_eq_bar, use_container_width=True)
        disp = cs_clean[['County', 'combined_score', 'recommendation', 'service_job_pct', 'growth_rate']].copy()
        disp.columns = ['County', 'Equity Score', 'Recommendation', 'Service Job %', 'Growth Rate']
        disp['Equity Score']  = disp['Equity Score'].round(1)
        disp['Service Job %'] = disp['Service Job %'].apply(lambda x: f"{x:.0%}")
        disp['Growth Rate']   = disp['Growth Rate'].apply(lambda x: f"{x:.0%}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        top8 = cs_clean.head(8).copy()
        sec_map = {
            'Educational services': 'Education',
            'Health care and social assistance': 'Health',
            'Professional and technical services': 'Professional',
            'Public Administration': 'Government',
            'Retail Trade': 'Retail',
            'Transportation and Warehousing': 'Transport',
            'Manufacturing': 'Manufacturing',
            'Construction': 'Construction',
        }
        sec_colors_list = [TM, TL, GLD, RD, GM, DC, PUR, TL]
        fig_sec = go.Figure()
        for i, (col, label) in enumerate(sec_map.items()):
            if col not in top8.columns: continue
            fig_sec.add_trace(go.Bar(
                name=label, x=top8['County'], y=top8[col] / 1000,
                marker_color=sec_colors_list[i % len(sec_colors_list)],
            ))
        fig_sec.update_layout(
            title='<b>Worker Sector Composition — Top 8 Equity Counties</b>',
            barmode='stack', height=480, plot_bgcolor=CRM, paper_bgcolor='white',
            yaxis=dict(title='Workers (thousands)', gridcolor='#e0e0e0'),
            xaxis=dict(tickangle=-20),
            font=dict(family='Arial', size=11),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    with tab3:
        med_g = cs_clean['growth_rate'].median()
        med_s = cs_clean['service_job_pct'].median()
        fig_quad = px.scatter(cs_clean, x='growth_rate', y='service_job_pct',
            color='recommendation', size='combined_score', text='County',
            color_discrete_map=rec_colors, size_max=40,
            title='<b>Equity Quadrant: Employment Growth vs Service Worker Share</b><br>'
                  '<sup>Lower-left = priority zone | Bubble size = AI equity score</sup>',
            labels={'growth_rate': 'Employment Growth Rate', 'service_job_pct': 'Service Worker Share'}
        )
        fig_quad.add_hline(y=med_s, line_dash='dash', line_color=GM, opacity=0.5)
        fig_quad.add_vline(x=med_g, line_dash='dash', line_color=GM, opacity=0.5)
        fig_quad.add_annotation(x=cs_clean['growth_rate'].min() + 0.005, y=0.95,
            text='EQUITY PRIORITY\n(slow growth, high service)',
            font=dict(size=10, color=RD), showarrow=False,
            bgcolor='rgba(255,252,232,0.85)', bordercolor=RD, borderwidth=1)
        fig_quad.update_traces(textposition='top center', textfont=dict(size=9))
        fig_quad.update_layout(
            height=540, plot_bgcolor=CRM, paper_bgcolor='white',
            font=dict(family='Arial', size=11),
            xaxis=dict(tickformat='.0%', gridcolor='#e0e0e0'),
            yaxis=dict(tickformat='.0%', gridcolor='#e0e0e0'),
            legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='center', x=0.5,
                        bgcolor='rgba(255,252,232,0.9)', bordercolor=TM, borderwidth=1),
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    with tab4:
        county_coords = {
            'Allegany': (39.63,-78.73), 'Anne Arundel': (38.99,-76.58),
            'Baltimore': (39.45,-76.60), 'Baltimore City': (39.29,-76.61),
            'Calvert': (38.53,-76.54), 'Caroline': (38.88,-75.88),
            'Carroll': (39.56,-77.00), 'Cecil': (39.57,-76.07),
            'Charles': (38.47,-76.94), 'Dorchester': (38.44,-76.10),
            'Frederick': (39.42,-77.41), 'Garrett': (39.55,-79.20),
            'Harford': (39.54,-76.33), 'Howard': (39.24,-76.89),
            'Kent': (39.21,-76.10), 'Montgomery': (39.14,-77.21),
            "Prince George's": (38.83,-76.87), "Queen Anne's": (39.02,-76.05),
            'Somerset': (38.07,-75.84), "St. Mary's": (38.22,-76.61),
            'Talbot': (38.75,-76.20), 'Washington': (39.64,-77.72),
            'Wicomico': (38.37,-75.64), 'Worcester': (38.22,-75.19),
        }
        cs_clean['lat'] = cs_clean['County'].map(lambda c: county_coords.get(c, (39.1,-76.9))[0])
        cs_clean['lon'] = cs_clean['County'].map(lambda c: county_coords.get(c, (39.1,-76.9))[1])
        m_eq = folium.Map(location=[39.1,-76.9], zoom_start=8, tiles='CartoDB positron')
        for _, row in cs_clean.iterrows():
            color = rec_colors.get(row['recommendation'], TL)
            r_sz  = max(10, min(28, int(row['combined_score'] / 4)))
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=r_sz, color=color, fill=True,
                fill_color=color, fill_opacity=0.75, weight=2,
                popup=folium.Popup(
                    f"<div style='font-family:sans-serif;border-left:4px solid {color};"
                    f"padding:8px 12px;min-width:220px'>"
                    f"<b style='color:{color}'>{row['County']}</b><br>"
                    f"Score: <b>{row['combined_score']:.1f}</b><br>"
                    f"Rec: {row['recommendation']}<br>"
                    f"Service jobs: {row['service_job_pct']:.0%}<br>"
                    f"Total jobs: {row['total_jobs_k']:.0f}K"
                    f"</div>", max_width=260)
            ).add_to(m_eq)
        st_folium(m_eq, height=480, use_container_width=True)
        st.caption(f"🔴 URGENT: Heavy Rail/BRT  |  🟡 HIGH: BRT+Equity  |  🟢 MEDIUM: Enhanced Bus  |  🔵 LOW: Park & Ride")

# ════════════════════════════════════════════════════════════════════
# AI FEATURE 4 — CONFLICT DETECTION (exact notebook code)
# ════════════════════════════════════════════════════════════════════
elif page == "⚠️ AI Feature 4: Conflict Detection":
    st.title("⚠️ AI Feature 4: Route Conflict Detection & Risk Agent")
    st.markdown("**Algorithm:** Haversine Spatial Scanning + Rule-Based AI Mitigation Generator")
    st.markdown("Scans every waypoint against real MDOT AADT data points, scores conflicts 0–10, and auto-writes mitigations.")
    st.markdown("---")

    with st.spinner("Running route conflict detection agent on MDOT data..."):
        aadt = load_aadt_points()

        routes = {
            'BRT-A: Silver Spring → Greenbelt': {
                'waypoints': [
                    (38.993,-77.030,'Silver Spring Station'),
                    (38.985,-77.002,'Takoma/Langley Crossroads'),
                    (38.978,-76.970,'University Blvd Junction'),
                    (38.978,-76.940,'Riggs Rd Intersection'),
                    (38.975,-76.912,'Greenbelt Metro Station'),
                ],
                'mode': 'BRT', 'color': '#E63946', 'road_corridor': 'University Boulevard / US-1'
            },
            'BRT-B: Rockville → Germantown': {
                'waypoints': [
                    (39.084,-77.146,'Rockville Station'),
                    (39.110,-77.185,'Shady Grove Metro'),
                    (39.130,-77.200,'Gaithersburg City Center'),
                    (39.155,-77.220,'Lakeforest Transit Hub'),
                    (39.173,-77.270,'Germantown MARC Station'),
                ],
                'mode': 'BRT', 'color': '#FB8500', 'road_corridor': 'I-270 / Shady Grove Rd'
            },
            'MARC+: Baltimore Penn → Odenton': {
                'waypoints': [
                    (39.307,-76.616,'Penn Station Baltimore'),
                    (39.238,-76.691,'Halethorpe Station'),
                    (39.193,-76.695,'BWI Airport Station'),
                    (39.120,-76.700,'Jessup Area'),
                    (39.087,-76.706,'Odenton Station'),
                ],
                'mode': 'MARC+', 'color': '#457B9D', 'road_corridor': 'Penn Line / MD-295'
            },
            'BRT-C: Baltimore → BWI → Odenton': {
                'waypoints': [
                    (39.290,-76.614,'Baltimore West Side'),
                    (39.250,-76.665,'Catonsville Junction'),
                    (39.215,-76.695,'BWI Airport Approach'),
                    (39.180,-76.700,'MD-170 Corridor'),
                    (39.087,-76.706,'Odenton Hub'),
                ],
                'mode': 'BRT', 'color': '#8338EC', 'road_corridor': 'MD-170 / BWI Loop Rd'
            }
        }

        def detect_traffic_conflicts(waypoints, aadt_df, radius_km=2.0):
            conflicts = []
            for lat, lon, wp_name in waypoints:
                aadt_sub = aadt_df[
                    (abs(aadt_df['lat'] - lat) < 0.05) &
                    (abs(aadt_df['lon'] - lon) < 0.08)
                ].copy()
                if len(aadt_sub) == 0: continue
                aadt_sub['dist_km'] = aadt_sub.apply(
                    lambda r: haversine(lat, lon, r['lat'], r['lon']), axis=1)
                nearby = aadt_sub[aadt_sub['dist_km'] <= radius_km].copy()
                if len(nearby) == 0: continue
                for _, road in nearby.nlargest(3, 'AADT Current').iterrows():
                    severity    = min(10, road['AADT Current'] / 15000)
                    urban_bonus = 1.5 if str(road.get('Rural / Urban','')).lower() in ['urban','u'] else 1.0
                    conflict_score = severity * urban_bonus
                    if conflict_score > 2.0:
                        conflicts.append({
                            'waypoint': wp_name, 'wp_lat': lat, 'wp_lon': lon,
                            'road': str(road.get('Road Name','Unknown'))[:40],
                            'county': str(road.get('County Name','Unknown')),
                            'aadt': int(road['AADT Current']),
                            'dist_km': round(road['dist_km'], 2),
                            'urban': road.get('Rural / Urban', 'Urban'),
                            'conflict_score': round(conflict_score, 2),
                            'severity_label': ('CRITICAL' if conflict_score > 8 else
                                               'HIGH'     if conflict_score > 5 else
                                               'MEDIUM'   if conflict_score > 3 else 'LOW'),
                        })
            return pd.DataFrame(conflicts) if conflicts else pd.DataFrame()

        def generate_mitigation(conflict_row):
            score  = conflict_row['conflict_score']
            road   = conflict_row['road']
            county = conflict_row['county']
            urban  = conflict_row['urban']
            if score > 8:
                return {'strategy': 'Off-hours construction with full lane closure 10PM–5AM',
                        'action': f'Coordinate with SHA for temporary traffic management on {road}',
                        'cost_impact': 'High (+$2–4M per mile)', 'delay_risk': '2–4 months', 'priority': 'IMMEDIATE'}
            elif score > 5:
                if 'urban' in str(urban).lower():
                    return {'strategy': 'Rolling closure — one lane at a time, off-peak only',
                            'action': f'Install real-time CHART sensors on {road}',
                            'cost_impact': 'Medium (+$0.8–1.5M)', 'delay_risk': '1–2 months', 'priority': 'HIGH'}
                else:
                    return {'strategy': 'Weekend full closures with advance detour signage',
                            'action': 'MDOT CHART advance notification 30 days prior',
                            'cost_impact': 'Low (+$0.3–0.8M)', 'delay_risk': '2–3 weeks', 'priority': 'HIGH'}
            else:
                return {'strategy': 'Standard construction with flagging and reduced speed zone',
                        'action': f'Standard MDOT SHA work zone permit for {county} County',
                        'cost_impact': 'Minimal (<$0.3M)', 'delay_risk': 'Negligible', 'priority': 'STANDARD'}

        all_conflicts = []
        route_summary = []
        for route_name, route_data in routes.items():
            conflicts = detect_traffic_conflicts(route_data['waypoints'], aadt)
            if len(conflicts) > 0:
                conflicts['route']       = route_name
                conflicts['mode']        = route_data['mode']
                conflicts['route_color'] = route_data['color']
                mitigations = conflicts.apply(generate_mitigation, axis=1)
                conflicts['mitigation_strategy'] = [m['strategy']     for m in mitigations]
                conflicts['mitigation_action']   = [m['action']       for m in mitigations]
                conflicts['cost_impact']         = [m['cost_impact']  for m in mitigations]
                conflicts['delay_risk']          = [m['delay_risk']   for m in mitigations]
                conflicts['mit_priority']        = [m['priority']     for m in mitigations]
                all_conflicts.append(conflicts)
            route_summary.append({
                'route': route_name, 'mode': route_data['mode'],
                'total_conflicts': len(conflicts),
                'critical': (conflicts['severity_label'] == 'CRITICAL').sum() if len(conflicts) > 0 else 0,
                'high':     (conflicts['severity_label'] == 'HIGH').sum()     if len(conflicts) > 0 else 0,
                'medium':   (conflicts['severity_label'] == 'MEDIUM').sum()   if len(conflicts) > 0 else 0,
                'max_aadt': conflicts['aadt'].max() if len(conflicts) > 0 else 0,
                'avg_conflict_score': conflicts['conflict_score'].mean().round(2) if len(conflicts) > 0 else 0,
                'color': route_data['color'],
            })

        conflicts_df    = pd.concat(all_conflicts, ignore_index=True) if all_conflicts else pd.DataFrame()
        route_summary_df = pd.DataFrame(route_summary)

    n    = len(conflicts_df)
    nc   = int(conflicts_df['severity_label'].eq('CRITICAL').sum()) if n > 0 else 0
    avgs = float(conflicts_df['conflict_score'].mean()) if n > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AADT Points Scanned", f"{len(aadt):,}")
    c2.metric("Conflicts Detected", n)
    c3.metric("Critical Conflicts", nc)
    c4.metric("Avg Conflict Score", f"{avgs:.1f}/10")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Conflicts by Route", "🗺️ Conflict Map", "⏱️ Risk Timeline", "📋 Mitigation Table", "🎯 Route Ease Gauges"])

    with tab1:
        sev_colors = {'CRITICAL': '#E63946', 'HIGH': '#FB8500', 'MEDIUM': '#FFBF69', 'LOW': '#95D5B2'}
        route_names_short = [r.split(':')[0] for r in route_summary_df['route']]
        fig_conf = go.Figure()
        for sev, color in sev_colors.items():
            col = sev.lower()
            if col in route_summary_df.columns:
                fig_conf.add_trace(go.Bar(
                    name=f'{sev} Conflict', x=route_names_short,
                    y=route_summary_df[col], marker_color=color,
                    text=route_summary_df[col], textposition='inside',
                ))
        fig_conf.update_layout(
            title='<b>🤖 AI Conflict Detector — Construction Conflicts per Route</b><br>'
                  f'<sup>Agent scanned {sum(len(r["waypoints"]) for r in routes.values())} waypoints against {len(aadt):,} MDOT traffic data points</sup>',
            barmode='stack', height=420, plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(title='Number of Conflict Points', gridcolor='#f0f0f0'),
            font=dict(family='Arial', size=12),
            legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        if n > 0:
            fig2 = px.scatter(conflicts_df, x='aadt', y='conflict_score',
                              color='route', size='conflict_score', size_max=20,
                              color_discrete_sequence=['#E63946','#FB8500','#457B9D','#8338EC'],
                              title='Conflict Score vs AADT Volume',
                              labels={'aadt':'AADT (vehicles/day)','conflict_score':'Conflict Score'})
            fig2.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                               font=dict(family='Arial', size=11))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        m_conf = folium.Map(location=[39.1,-76.9], zoom_start=9, tiles='CartoDB positron')
        sev_map_colors = {'CRITICAL':'#E63946','HIGH':'#FB8500','MEDIUM':'#FFBF69','LOW':'#95D5B2'}
        for route_name, route_data in routes.items():
            wps = route_data['waypoints']
            folium.PolyLine(
                locations=[(lat, lon) for lat, lon, _ in wps],
                color=route_data['color'], weight=4, opacity=0.7,
                popup=route_name,
            ).add_to(m_conf)
        if n > 0:
            for _, row in conflicts_df.iterrows():
                color = sev_map_colors.get(row['severity_label'], '#888')
                folium.CircleMarker(
                    location=[row['wp_lat'], row['wp_lon']],
                    radius=max(6, int(row['conflict_score'])),
                    color=color, fill=True, fill_color=color, fill_opacity=0.8, weight=2,
                    popup=folium.Popup(
                        f"<b>⚠️ {row['severity_label']} CONFLICT</b><br>"
                        f"<b>Waypoint:</b> {row['waypoint']}<br>"
                        f"<b>Road:</b> {row['road']}<br>"
                        f"<b>AADT:</b> {row['aadt']:,} vehicles/day<br>"
                        f"<b>Score:</b> {row['conflict_score']:.1f}/10<br>"
                        f"<hr><b>AI Mitigation:</b><br>{row['mitigation_strategy']}<br>"
                        f"<i>{row['mitigation_action']}</i><br>"
                        f"<b>Cost:</b> {row['cost_impact']}<br>"
                        f"<b>Delay:</b> {row['delay_risk']}",
                        max_width=320)
                ).add_to(m_conf)
        st_folium(m_conf, height=500, use_container_width=True)
        st.caption("Colored lines = proposed BRT/MARC routes. Circles = detected conflicts. Click for AI mitigation plan.")

    with tab3:
        months_tl = list(range(1, 49))
        risk_timeline = np.zeros(48)
        construction_phases = {
            'Phase 1: MARC+ Upgrades':    (9, 27),
            'Phase 2: BRT Silver Spring': (12, 32),
            'Phase 3: BRT Rockville':     (18, 31),
            'Phase 4: BRT BWI':           (24, 38),
        }
        phase_risk = {'Phase 1: MARC+ Upgrades': 3.5, 'Phase 2: BRT Silver Spring': 7.8,
                      'Phase 3: BRT Rockville': 5.9, 'Phase 4: BRT BWI': 6.4}
        for phase, (start, end) in construction_phases.items():
            base = phase_risk.get(phase, 3.0)
            mid  = (start + end) / 2
            for mo in range(min(start, 47), min(end, 48)):
                bell = np.exp(-0.5 * ((mo - mid) / 6)**2)
                risk_timeline[mo] += base * bell

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=months_tl, y=risk_timeline[:48],
            fill='tozeroy', fillcolor='rgba(230,57,70,0.12)',
            line=dict(color='#E63946', width=2.5),
            name='Aggregate Construction Risk Score', mode='lines',
        ))
        phase_colors_tl = {'Phase 1: MARC+ Upgrades':'#457B9D','Phase 2: BRT Silver Spring':'#E63946',
                           'Phase 3: BRT Rockville':'#FB8500','Phase 4: BRT BWI':'#8338EC'}
        for phase, (start, end) in construction_phases.items():
            color = phase_colors_tl.get(phase, '#888')
            fig_timeline.add_shape(type='rect', x0=start, x1=end, y0=8.2, y1=8.8,
                                   fillcolor=color, opacity=0.8, line=dict(width=0))
            fig_timeline.add_annotation(x=(start+end)/2, y=9.1,
                                        text=phase.split(':')[0], font=dict(size=8,color=color), showarrow=False)
        fig_timeline.add_hline(y=5.0, line_dash='dash', line_color='#2D6A4F', line_width=2,
                               annotation_text='Mitigation Threshold',
                               annotation_font=dict(size=10, color='#2D6A4F'))
        fig_timeline.update_layout(
            title='<b>Construction Risk Timeline — Monthly Risk Score Over 4 Years</b><br>'
                  '<sup>AI agent projects when conflict risk is highest | Phase overlaps = compounded risk</sup>',
            xaxis=dict(title='Project Month', dtick=6, gridcolor='#f0f0f0'),
            yaxis=dict(title='Aggregate Risk Score', gridcolor='#f0f0f0', range=[0, 10]),
            height=440, plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family='Arial', size=12), showlegend=False,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab4:
        if n > 0:
            top_conflicts = conflicts_df.nlargest(15, 'conflict_score')
            table_data = top_conflicts[['route','waypoint','road','aadt','conflict_score',
                                        'severity_label','mitigation_strategy','cost_impact','delay_risk']].copy()
            table_data['aadt_fmt']  = table_data['aadt'].apply(lambda v: f'{v:,}')
            table_data['score_fmt'] = table_data['conflict_score'].apply(lambda v: f'{v:.1f}/10')

            def row_color(sev):
                return {'CRITICAL':'#FFD7D7','HIGH':'#FFE8C8','MEDIUM':'#FEFBD4','LOW':'#D4F1DC'}.get(sev,'white')
            cell_colors = [[row_color(s) for s in table_data['severity_label']]] * 8

            fig_table = go.Figure(data=[go.Table(
                columnwidth=[100,120,140,80,70,80,200,120],
                header=dict(
                    values=['<b>Route</b>','<b>Waypoint</b>','<b>Road</b>','<b>AADT</b>',
                            '<b>Score</b>','<b>Severity</b>','<b>AI Mitigation</b>','<b>Cost Impact</b>'],
                    fill_color='#2D6A4F', font=dict(color='white', size=10),
                    align='left', height=35,
                ),
                cells=dict(
                    values=[table_data['route'].str.split(':').str[0],
                            table_data['waypoint'], table_data['road'],
                            table_data['aadt_fmt'], table_data['score_fmt'],
                            table_data['severity_label'],
                            table_data['mitigation_strategy'], table_data['cost_impact']],
                    fill_color=cell_colors, font=dict(size=9), align='left', height=28,
                ),
            )])
            fig_table.update_layout(
                title='<b>🤖 AI-Generated Mitigation Action Plan — Top 15 Conflicts</b>',
                height=520, font=dict(family='Arial'),
            )
            st.plotly_chart(fig_table, use_container_width=True)
        else:
            st.info("No conflict data detected. Try increasing radius_km or check that AADT points file has coordinates.")

    with tab5:
        route_summary_df['risk_score'] = (
            route_summary_df['critical'] * 10 +
            route_summary_df['high'] * 5 +
            route_summary_df['medium'] * 2
        ).clip(0, 100)
        route_summary_df['ease_score'] = (100 - route_summary_df['risk_score']).clip(0, 100)

        fig_gauge = make_subplots(rows=1, cols=4, specs=[[{'type': 'indicator'}] * 4])
        gauge_colors = ['#E63946','#FB8500','#457B9D','#8338EC']
        for i, (_, row) in enumerate(route_summary_df.iterrows()):
            ease = float(row['ease_score'])
            fig_gauge.add_trace(go.Indicator(
                mode='gauge+number+delta',
                value=ease,
                title={'text': row['route'].split(':')[0], 'font': {'size': 10}},
                delta={'reference': 75, 'valueformat': '.0f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': gauge_colors[i]},
                    'steps': [
                        {'range': [0, 40],  'color': '#FFD7D7'},
                        {'range': [40, 70], 'color': '#FFF3CD'},
                        {'range': [70,100], 'color': '#D4EDDA'},
                    ],
                    'threshold': {'line': {'color': '#333','width': 2},
                                  'thickness': 0.75, 'value': 75},
                },
                number={'suffix': '/100', 'font': {'size': 18}},
            ), row=1, col=i+1)
        fig_gauge.update_layout(
            title='<b>🤖 Agent Route Ease Score — Higher = Fewer Conflicts</b><br>'
                  '<sup>Green zone (70+) = proceed | Yellow = mitigate | Red = redesign</sup>',
            height=300, font=dict(family='Arial', size=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
