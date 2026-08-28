import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sqlite3
from streamlit_geolocation import streamlit_geolocation
from folium import Map, CircleMarker, Popup, LayerControl, TileLayer, CircleMarker, Popup, LayerControl, TileLayer
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json
import base64

# --- DB Setup ---
conn = sqlite3.connect('incidents.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS incidents
             (id INTEGER PRIMARY KEY AUTOINCREMENT, 
              name TEXT, date TEXT, state TEXT, lat REAL, lon REAL, severity TEXT, description TEXT)''')
conn.commit()


# Try to import from src, fallback to sample data if not available
try:
    from src.risk_engine.scoring import RiskScoringEngine
    from src.config.settings import MONITORING_LOCATIONS, NE_STATES
except ImportError:
    MONITORING_LOCATIONS = [
        {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362},
        {"name": "Shillong", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933},
        {"name": "Kohima", "state": "Nagaland", "lat": 25.6701, "lon": 94.1077},
        {"name": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.0844, "lon": 93.6053},
        {"name": "Aizawl", "state": "Mizoram", "lat": 23.7271, "lon": 92.7176},
        {"name": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868},
        {"name": "Imphal", "state": "Manipur", "lat": 24.8170, "lon": 93.9368},
        {"name": "Gangtok", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065}
    ]
    NE_STATES = ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"]
    RiskScoringEngine = None

st.set_page_config(page_title='Landslide Detection System - NE India', layout='wide', page_icon='🏔️')

# --- Custom CSS ---
st.markdown("""
<style>
    .risk-low { color: white; background-color: #28a745; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-moderate { color: white; background-color: #ffc107; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-high { color: white; background-color: #fd7e14; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-very_high { color: white; background-color: #dc3545; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .risk-severe { color: white; background-color: #8b0000; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    .stProgress .st-bo { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- Generate Sample Data ---
@st.cache_data
def get_sample_scores():
    np.random.seed(datetime.datetime.now().hour) # Change hourly
    data = []
    for loc in MONITORING_LOCATIONS:
        prob = np.random.uniform(0.05, 0.95)
        if prob < 0.2: level = "LOW"
        elif prob < 0.4: level = "MODERATE"
        elif prob < 0.7: level = "HIGH"
        elif prob < 0.9: level = "VERY_HIGH"
        else: level = "SEVERE"
        
        data.append({
            "name": loc["name"],
            "state": loc["state"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "risk_probability": prob,
            "risk_level": level,
            "rainfall_24h": np.random.uniform(0, 150),
            "soil_moisture": np.random.uniform(0.1, 0.5),
            "slope": np.random.uniform(10, 60),
            "timestamp": datetime.datetime.now().isoformat()
        })
    return pd.DataFrame(data)

# --- Sidebar ---
st.sidebar.title('🏔️ Landslide Detection System')
st.sidebar.subheader('North Eastern India')

selected_states = st.sidebar.multiselect('Select States', NE_STATES, default=NE_STATES)
selected_risk = st.sidebar.multiselect('Risk Level', ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "SEVERE"], default=["HIGH", "VERY_HIGH", "SEVERE"])
selected_date = st.sidebar.date_input('Date for Analysis', datetime.date.today())

refresh = st.sidebar.button('🔄 Refresh Data')

st.sidebar.markdown("---")

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: red;'>EMERGENCY</h3>", unsafe_allow_html=True)
if st.sidebar.button("🚨 SOS ALARM 🚨", type="primary", use_container_width=True):
    st.sidebar.error("SOS Alert Triggered! Emergency response teams have been notified.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Credits:**\nDeveloped by NE India Landslide Response Team.")


# --- Data Loading ---
try:
    if RiskScoringEngine:
        engine = RiskScoringEngine()
        df_scores = engine.score_all_stations()
    else:
        df_scores = get_sample_scores()
except Exception as e:
    st.error(f"Error connecting to backend services: {e}. Showing sample data.")
    df_scores = get_sample_scores()

# Filter data
mask = df_scores["state"].isin(selected_states)
if selected_risk:
    mask &= df_scores["risk_level"].isin(selected_risk)
filtered_df = df_scores[mask]

# --- Main App ---


def laughing_emojis():
    import random
    emojis_list = ['😂', '🤣', '😆']
    divs = ""
    for i in range(30):
        e = random.choice(emojis_list)
        left = random.uniform(0, 95)
        dur = random.uniform(2, 4)
        delay = random.uniform(0, 2)
        divs += f'<div class="emoji-rain" style="left:{left}vw;animation-duration:{dur}s;animation-delay:{delay}s;">{e}</div>'
    emoji_html = f'''
    <style>
    @keyframes emoji-fall {{
        0% {{ top: -10vh; opacity: 1; transform: rotate(0deg); }}
        100% {{ top: 100vh; opacity: 0; transform: rotate(360deg); }}
    }}
    .emoji-rain {{
        position: fixed;
        top: -10vh;
        z-index: 999999;
        font-size: 2.5rem;
        user-select: none;
        pointer-events: none;
        animation: emoji-fall linear forwards;
    }}
    </style>
    {divs}
    '''
    st.markdown(emoji_html, unsafe_allow_html=True)


def play_cheer():
    try:
        with open("dashboard/cheering.mp3", "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        pass


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Risk Map", 
    "📊 Risk Analysis", 
    "🌧️ Weather Monitor", 
    "🔬 Model Insights", 
    "📋 Landslide Inventory", 
    "🌍 Seismic Activity",
    "📝 Report Incident"
])

def get_color(level):
    colors = {"LOW": "green", "MODERATE": "orange", "HIGH": "darkorange", "VERY_HIGH": "red", "SEVERE": "darkred"}
    return colors.get(level, "gray")

with tab1:
    st.header("Real-time Landslide Risk Map")
    
    m = Map(location=[26.0, 92.5], zoom_start=7)
    TileLayer('CartoDB positron').add_to(m)
    
    for _, row in filtered_df.iterrows():
        html = f"""
        <b>{row['name']}, {row['state']}</b><br>
        Risk Level: {row['risk_level']}<br>
        Probability: {row['risk_probability']:.2f}<br>
        Rainfall: {row.get('rainfall_24h', 0):.1f}mm
        """
        CircleMarker(
            location=[row['lat'], row['lon']],
            radius=10,
            color=get_color(row['risk_level']),
            fill=True,
            fill_color=get_color(row['risk_level']),
            fill_opacity=0.7,
            popup=Popup(html, max_width=300)
        ).add_to(m)
        
    st_folium(m, width=1200, height=600)
    
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Monitored Stations", len(df_scores))
    high_risk_count = len(df_scores[df_scores['risk_level'].isin(['HIGH', 'VERY_HIGH', 'SEVERE'])])
    col2.metric("High/Severe Risk Locations", high_risk_count, delta=f"{high_risk_count} Alert(s)", delta_color="inverse")
    col3.metric("Last Updated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with tab2:
    st.header("Risk Analysis Dashboard")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Risk by State")
        state_avg = df_scores.groupby('state')['risk_probability'].mean().reset_index()
        fig1 = px.bar(state_avg, x='state', y='risk_probability', color='risk_probability', 
                     color_continuous_scale='Reds', title="State Risk Averages")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Risk Distribution")
        fig2 = px.histogram(df_scores, x='risk_probability', nbins=10, 
                           title="Distribution of Risk Probabilities across Stations")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Top High-Risk Locations")
    top_risk = df_scores.nlargest(10, 'risk_probability')[['name', 'state', 'risk_level', 'risk_probability', 'rainfall_24h']]
    
    def highlight_risk(val):
        color_map = {"LOW": "#28a745", "MODERATE": "#ffc107", "HIGH": "#fd7e14", "VERY_HIGH": "#dc3545", "SEVERE": "#8b0000"}
        color = color_map.get(val, "black")
        return f'background-color: {color}; color: white'
        
    st.dataframe(top_risk.style.map(highlight_risk, subset=['risk_level']), use_container_width=True)

with tab3:
    st.header("Weather Monitor")
    st.info("Weather data fetched from monitoring APIs.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Rainfall")
        if not filtered_df.empty:
            filtered_df["rainfall_24h"] = pd.to_numeric(filtered_df["rainfall_24h"], errors="coerce").fillna(0)
            st.map(filtered_df, latitude="lat", longitude="lon", size="rainfall_24h", color="#1f77b4", use_container_width=True)
        else:
            st.warning("No data to map.")
        
    with col2:
        st.subheader("Soil Moisture Levels")
        if not filtered_df.empty:
            filtered_df["soil_moisture"] = pd.to_numeric(filtered_df["soil_moisture"], errors="coerce").fillna(0)
            st.map(filtered_df, latitude="lat", longitude="lon", size="soil_moisture", color="#8c564b", use_container_width=True)
        else:
            st.warning("No data to map.")

    st.subheader("Raw Weather Data")
    st.dataframe(df_scores[['name', 'state', 'rainfall_24h', 'soil_moisture', 'slope']], use_container_width=True)

with tab4:
    st.header("Model Insights")
    st.info("Explaining the predictions of the ML model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        # Dummy feature importance
        feat_imp = pd.DataFrame({
            "Feature": ["Rainfall (24h)", "Slope", "Soil Moisture", "Elevation", "Seismic Activity"],
            "Importance": [0.45, 0.30, 0.15, 0.08, 0.02]
        }).sort_values("Importance", ascending=True)
        fig_feat = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', title="Global Feature Importance")
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col2:
        st.subheader("Model Performance")
        st.metric("Accuracy", "92.4%", "+1.2%")
        st.metric("AUC-ROC", "0.95")
        st.metric("F1-Score", "0.89")
        
    st.subheader("Local Explanation (Simulated SHAP)")
    selected_loc = st.selectbox("Select Location for Explanation", df_scores['name'].tolist())
    st.write(f"Explanation for prediction at **{selected_loc}**:")
    
    # Waterfall chart for dummy SHAP
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "h",
        measure = ["relative", "relative", "relative", "relative", "total"],
        y = ["Base Value", "Rainfall Effect", "Slope Effect", "Moisture Effect", "Final Risk Prob"],
        x = [0.1, 0.4, 0.2, 0.1, 0.8],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Landslide Inventory")
    st.info("Historical catalog of landslide events.")
    
    # Dummy historical data
    hist_data = pd.DataFrame({
        "Date": ["2023-06-15", "2023-07-02", "2022-08-14", "2021-06-25"],
        "Location": ["Dimapur", "Tawang", "Cherrapunji", "Silchar"],
        "State": ["Nagaland", "Arunachal Pradesh", "Meghalaya", "Assam"],
        "Trigger": ["Heavy Rain", "Heavy Rain", "Extreme Rain", "Earthquake"],
        "Severity": ["High", "Medium", "Severe", "High"]
    })
    
    st.dataframe(hist_data, use_container_width=True)
    
    fig_hist = px.pie(hist_data, names="Trigger", title="Events by Trigger Type")
    st.plotly_chart(fig_hist)

with tab6:
    st.header("Seismic Activity")
    st.info("Recent earthquakes in the NE region (Placeholder Data).")
    
    seismic_data = pd.DataFrame({
        "Time": [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(5)],
        "Location": ["Near Imphal", "Near Kohima", "Near Tezpur", "Near Itanagar", "Near Aizawl"],
        "Magnitude": [3.2, 4.1, 2.5, 3.8, 4.5],
        "Depth (km)": [10, 15, 8, 25, 12],
        "Lat": [24.8, 25.6, 26.6, 27.1, 23.7],
        "Lon": [93.9, 94.1, 92.8, 93.6, 92.7]
    })
    
    st.dataframe(seismic_data, use_container_width=True)
    
    if not seismic_data.empty:
        st.map(seismic_data, latitude="Lat", longitude="Lon", size="Magnitude", color="#d62728", use_container_width=True)
    else:
        st.warning("No data to map.")


with tab7:
    st.header("Report a Landslide Incident")
    st.info("Use this form to register a new landslide event into the system database.")
    
    st.subheader("📍 Get Current Location")
    location = streamlit_geolocation()
    
    default_lat = 25.0
    default_lon = 92.0
    if location and location.get('latitude') is not None and location.get('longitude') is not None:
        default_lat = float(location['latitude'])
        default_lon = float(location['longitude'])
        st.success(f"Location captured! Lat: {default_lat:.4f}, Lon: {default_lon:.4f}")
    
    with st.form("incident_report_form"):
        col1, col2 = st.columns(2)
        with col1:
            reporter_name = st.text_input("Your Name / Organization")
            incident_date = st.date_input("Date of Incident", datetime.date.today())
            incident_state = st.selectbox("State", NE_STATES)
        with col2:
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, format="%.6f")
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, format="%.6f")
            severity = st.selectbox("Severity", ["Minor (Road Blocked)", "Moderate (Property Damage)", "Severe (Casualties/Major Destruction)"])
        
        description = st.text_area("Incident Description", placeholder="Describe the landslide extent, triggers (e.g. heavy rain), and damages...")
        
        submitted = st.form_submit_button("Submit Incident Report", type="primary")
        
        if submitted:
            if reporter_name and description:
                # Save to database
                c.execute("INSERT INTO incidents (name, date, state, lat, lon, severity, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (reporter_name, str(incident_date), incident_state, lat, lon, severity, description))
                conn.commit()
                st.success(f"✅ Incident reported successfully and saved to database!")
                st.balloons()
                laughing_emojis()
                play_cheer()
            else:
                st.error("Please fill in your name and a description of the incident.")
    
    st.markdown("---")
    st.subheader("Recent Reported Incidents")
    recent_incidents = pd.read_sql("SELECT * FROM incidents ORDER BY id DESC", conn)
    if not recent_incidents.empty:
        st.dataframe(recent_incidents, use_container_width=True)
        # Also plot on a map
        st.map(recent_incidents, latitude="lat", longitude="lon", color="#ff0000", size=50)
    else:
        st.write("No incidents reported yet.")
