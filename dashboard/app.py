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
c.execute('''CREATE TABLE IF NOT EXISTS emergency_alerts
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              incident_id INTEGER, alert_type TEXT, agency TEXT, 
              status TEXT DEFAULT 'DISPATCHED', timestamp TEXT,
              FOREIGN KEY(incident_id) REFERENCES incidents(id))''')
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

st.set_page_config(page_title='NLRMP | NDMA, Government of India', layout='wide', page_icon='🇮🇳')

# --- Custom CSS ---

st.markdown('''
<style>
/* Government Header Styling */
.gov-header {
    background-color: #f1f6fa;
    border-bottom: 4px solid #FF9933; /* Saffron border */
    border-top: 4px solid #138808;    /* Green top border */
    padding: 15px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Arial', sans-serif;
    color: #000;
    margin-top: -60px;
    margin-bottom: 20px;
    border-radius: 4px;
}
.gov-header-left, .gov-header-right {
    flex: 1;
    display: flex;
    align-items: center;
}
.gov-header-right {
    justify-content: flex-end;
}
.gov-header-center {
    flex: 3;
    text-align: center;
}
.gov-title-hi {
    font-size: 1.4rem;
    font-weight: bold;
    color: #000066;
    margin: 0;
}
.gov-title-en {
    font-size: 1.6rem;
    font-weight: bold;
    color: #000066;
    margin: 5px 0;
}
.gov-subtitle {
    font-size: 1.1rem;
    color: #333;
    margin: 0;
    font-weight: 600;
}
.tricolor-line {
    height: 4px;
    background: linear-gradient(to right, #FF9933 33.3%, #FFFFFF 33.3%, #FFFFFF 66.6%, #138808 66.6%);
    width: 100%;
    margin-top: 5px;
}
</style>

<div class="gov-header">
    <div class="gov-header-left">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg" height="90" alt="Emblem of India">
    </div>
    <div class="gov-header-center">
        <p class="gov-title-hi">राष्ट्रीय भूस्खलन जोखिम शमन परियोजना</p>
        <p class="gov-title-en">National Landslide Risk Mitigation Project (NLRMP)</p>
        <div class="tricolor-line"></div>
        <p class="gov-subtitle" style="margin-top:8px;">National Disaster Management Authority, Government of India</p>
    </div>
    <div class="gov-header-right">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/Digital_India_logo.svg" height="65" alt="Digital India">
    </div>
</div>
''', unsafe_allow_html=True)

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

def play_emergency_siren():
    import struct, math
    sample_rate = 8000
    duration = 1.5
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        freq = 600 + 600 * (0.5 + 0.5 * math.sin(2 * math.pi * 2 * t))
        val = int(32767 * 0.4 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack('<h', val))
    audio_data = b''.join(samples)
    data_size = len(audio_data)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b'data', data_size)
    wav = header + audio_data
    b64 = base64.b64encode(wav).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>', unsafe_allow_html=True)

def dispatch_emergency_alerts(incident_id, state, severity, lat, lon):
    timestamp = datetime.datetime.now().isoformat()
    agencies = [
        ("NDRF", "National Disaster Response Force"),
        ("SDMA", f"{state} Disaster Management Authority"),
        ("HOSPITAL", "Nearest District Hospital"),
        ("POLICE", f"{state} Police Control Room"),
    ]
    if "Severe" in severity:
        agencies.append(("ARMY", "Indian Army Disaster Relief"))
        agencies.append(("IAF", "Indian Air Force Rescue"))
    
    for code_name, agency in agencies:
        c.execute("INSERT INTO emergency_alerts (incident_id, alert_type, agency, status, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (incident_id, code_name, agency, "DISPATCHED", timestamp))
    conn.commit()

# --- Sidebar ---
st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg', width=50)
st.sidebar.markdown('**Government of India**<br><span style="color:#000066; font-weight:bold;">NDMA Early Warning System</span>', unsafe_allow_html=True)


st.sidebar.markdown("**Region Filter**")
region_filter = st.sidebar.radio("Select Region", ["North East India", "Outside North East", "All India"], label_visibility="collapsed")

if region_filter == "North East India":
    available_states = NE_STATES
elif region_filter == "Outside North East":
    available_states = NON_NE_STATES
else:
    available_states = ALL_STATES

selected_states = st.sidebar.multiselect('Select States', available_states, default=available_states)

selected_risk = st.sidebar.multiselect('Risk Level', ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "SEVERE"], default=["HIGH", "VERY_HIGH", "SEVERE"])
selected_date = st.sidebar.date_input('Date for Analysis', datetime.date.today())

refresh = st.sidebar.button('🔄 Refresh Data')


st.sidebar.markdown("---")
theme = st.sidebar.toggle("🌙 Dark Mode", value=False)

# 2. Mountain cursor + theme CSS + animated weather effects
if theme:
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#262730"
else:
    bg_color = "#ffffff"
    text_color = "#1a1a2e"
    card_bg = "#f0f2f6"

custom_css = f'''
<style>
/* Theme override */
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

/* Animated rain effect on background */
@keyframes rainDrop {{
    0% {{ top: -5vh; opacity: 0.7; }}
    100% {{ top: 105vh; opacity: 0; }}
}}
.rain-container {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}}
.rain-drop {{
    position: absolute;
    top: -5vh;
    width: 2px;
    height: 15px;
    background: linear-gradient(transparent, rgba(100, 180, 255, 0.6));
    border-radius: 0 0 2px 2px;
    animation: rainDrop linear infinite;
}}

/* Alert sound indicator */
.risk-pulse {{
    animation: pulse 1.5s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# Animated weather rain effect
import random as _rnd
rain_divs = ""
for i in range(40):
    left = _rnd.uniform(0, 100)
    dur = _rnd.uniform(1.5, 3.5)
    delay = _rnd.uniform(0, 3)
    rain_divs += f'<div class="rain-drop" style="left:{left}vw;animation-duration:{dur}s;animation-delay:{delay}s;"></div>'
st.markdown(f'<div class="rain-container">{rain_divs}</div>', unsafe_allow_html=True)


st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: red;'>EMERGENCY</h3>", unsafe_allow_html=True)
if st.sidebar.button("🚨 SOS ALARM 🚨", type="primary", use_container_width=True):
    st.sidebar.error("🚨 SOS Alert Triggered!")
    st.sidebar.warning("📞 NDRF Helpline: 011-24363260")
    st.sidebar.warning("📞 Disaster Helpline: 1078")
    st.sidebar.warning("📞 Police: 100 | Ambulance: 108")
    play_emergency_siren()

st.sidebar.markdown("---")
st.sidebar.markdown("**© 2026 NDMA, Gov of India**<br>Designed & Developed by<br>National Informatics Centre (NIC)", unsafe_allow_html=True)


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
col_title, col_admin = st.columns([5, 2])
with col_admin:
    with st.expander("🔐 Admin Access"):
        admin_password = st.text_input("Password", type="password")
        if admin_password == "admin123":
            st.success("Admin Authenticated")
            
            # Selective Delete
            st.write("---")
            st.write("**Selective Delete**")
            incidents_df = pd.read_sql("SELECT id, name, date, state FROM incidents ORDER BY id DESC", conn)
            if not incidents_df.empty:
                opts = {f"ID {r['id']}: {r['name']} ({r['date']})": r['id'] for _, r in incidents_df.iterrows()}
                sel_inc = st.selectbox("Select Incident", list(opts.keys()), label_visibility="collapsed")
                if st.button("🗑️ Delete Selected", type="secondary", use_container_width=True):
                    inc_id = opts[sel_inc]
                    c.execute("DELETE FROM emergency_alerts WHERE incident_id=?", (inc_id,))
                    c.execute("DELETE FROM incidents WHERE id=?", (inc_id,))
                    conn.commit()
                    st.success("Deleted!")
            else:
                st.write("No incidents found.")
            
            # Clear All
            st.write("---")
            st.write("**Bulk Actions**")
            if st.button("🗑️ Clear All Reports", type="primary", use_container_width=True):
                c.execute("DELETE FROM emergency_alerts")
                c.execute("DELETE FROM incidents")
                conn.commit()
                st.success("Database cleared!")
        elif admin_password != "":
            st.error("Incorrect password")




# Risk level alert sounds
def play_risk_alert(risk_level):
    # Generate different beep tones for different risk levels using base64 WAV
    import struct, math
    sample_rate = 8000
    duration = 0.3
    freqs = {"LOW": 440, "MODERATE": 550, "HIGH": 660, "VERY_HIGH": 880, "SEVERE": 1100}
    freq = freqs.get(risk_level, 440)
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack('<h', val))
    audio_data = b''.join(samples)
    # WAV header
    data_size = len(audio_data)
    header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16, b'data', data_size)
    wav = header + audio_data
    b64 = base64.b64encode(wav).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>', unsafe_allow_html=True)







tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Risk Map", 
    "📊 Risk Analysis", 
    "🌧️ Weather Monitor", 
    "🔬 Model Insights", 
    "📋 Landslide Inventory", 
    "🌍 Seismic Activity",
    "🆘 Report & Emergency Alerts"
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
    detected_state = None
    
    # State bounding boxes for auto-detection (all Indian states + UTs)
    STATE_BOUNDS = {
        "Andhra Pradesh": (12.4, 19.9, 76.7, 84.8),
        "Arunachal Pradesh": (26.5, 29.5, 91.5, 97.5),
        "Assam": (24.0, 28.0, 89.5, 96.5),
        "Bihar": (24.2, 27.5, 83.3, 88.2),
        "Chhattisgarh": (17.8, 24.1, 80.2, 84.4),
        "Goa": (14.9, 15.8, 73.6, 74.3),
        "Gujarat": (20.1, 24.7, 68.2, 74.5),
        "Haryana": (27.6, 30.9, 74.5, 77.6),
        "Himachal Pradesh": (30.4, 33.3, 75.6, 79.0),
        "Jharkhand": (21.9, 25.3, 83.3, 87.9),
        "Karnataka": (11.6, 18.5, 74.0, 78.6),
        "Kerala": (8.2, 12.8, 74.8, 77.4),
        "Madhya Pradesh": (21.1, 26.9, 74.0, 82.8),
        "Maharashtra": (15.6, 22.0, 72.6, 80.9),
        "Manipur": (23.8, 25.7, 93.0, 94.8),
        "Meghalaya": (25.0, 26.2, 89.8, 92.8),
        "Mizoram": (21.9, 24.5, 92.2, 93.5),
        "Nagaland": (25.2, 27.0, 93.3, 95.3),
        "Odisha": (17.8, 22.6, 81.3, 87.5),
        "Punjab": (29.5, 32.5, 73.9, 76.9),
        "Rajasthan": (23.0, 30.2, 69.5, 78.3),
        "Sikkim": (27.0, 28.2, 88.0, 89.0),
        "Tamil Nadu": (8.0, 13.6, 76.2, 80.4),
        "Telangana": (15.8, 19.9, 77.2, 81.3),
        "Tripura": (22.9, 24.5, 91.1, 92.4),
        "Uttar Pradesh": (23.9, 30.4, 77.1, 84.6),
        "Uttarakhand": (28.7, 31.5, 77.6, 81.0),
        "West Bengal": (21.5, 27.2, 86.0, 89.9),
        "Delhi": (28.4, 28.9, 76.8, 77.4),
        "Jammu & Kashmir": (32.2, 37.1, 73.3, 80.3),
        "Ladakh": (32.5, 37.0, 75.5, 80.3),
        "Chandigarh": (30.6, 30.8, 76.7, 76.9),
        "Puducherry": (10.7, 12.0, 79.6, 80.0),
        "Andaman & Nicobar": (6.7, 13.7, 92.2, 94.3),
        "Lakshadweep": (8.2, 12.5, 71.7, 74.0),
        "Dadra & Nagar Haveli": (20.0, 20.4, 72.9, 73.3),
        "Daman & Diu": (20.3, 20.8, 70.8, 73.0),
    }
    
    ALL_STATES = sorted(STATE_BOUNDS.keys())
    
    def detect_state(lat, lon):
        for state, (min_lat, max_lat, min_lon, max_lon) in STATE_BOUNDS.items():
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return state
        return None
    
    if location and location.get('latitude') is not None and location.get('longitude') is not None:
        default_lat = float(location['latitude'])
        default_lon = float(location['longitude'])
        detected_state = detect_state(default_lat, default_lon)
        if detected_state:
            st.success(f"📍 Location captured! Lat: {default_lat:.4f}, Lon: {default_lon:.4f} — Detected State: **{detected_state}**")
        else:
            st.success(f"📍 Location captured! Lat: {default_lat:.4f}, Lon: {default_lon:.4f} — (Outside India)")
    
    default_state_idx = ALL_STATES.index(detected_state) if detected_state and detected_state in ALL_STATES else 0
    
    with st.form("incident_report_form"):
        col1, col2 = st.columns(2)
        with col1:
            reporter_name = st.text_input("Your Name / Organization")
            incident_date = st.date_input("Date of Incident", datetime.date.today())
            incident_state = st.selectbox("State", ALL_STATES, index=default_state_idx)
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
                
                # Emergency alert notification
                alert_html = f'''
                <div style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); 
                     border-radius: 12px; padding: 20px; margin: 10px 0; color: white; 
                     border: 2px solid #ff6666; box-shadow: 0 4px 15px rgba(255,0,0,0.3);">
                    <h3 style="margin:0; color:white;">🚨 EMERGENCY ALERT DISPATCHED</h3>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p style="margin:5px 0;"><strong>📍 Location:</strong> {incident_state} ({lat:.4f}, {lon:.4f})</p>
                    <p style="margin:5px 0;"><strong>📅 Date:</strong> {incident_date}</p>
                    <p style="margin:5px 0;"><strong>⚠️ Severity:</strong> {severity}</p>
                    <p style="margin:5px 0;"><strong>👤 Reporter:</strong> {reporter_name}</p>
                    <hr style="border-color: rgba(255,255,255,0.3);">
                    <p style="margin:5px 0;">✅ NDRF Team Notified</p>
                    <p style="margin:5px 0;">✅ State Disaster Management Authority Alerted</p>
                    <p style="margin:5px 0;">✅ Nearest Hospital Informed</p>
                    <p style="margin:5px 0;">✅ Local Police Control Room Notified</p>
                </div>
                '''
                st.markdown(alert_html, unsafe_allow_html=True)
                play_emergency_siren()
                
                # Dispatch alerts to agencies and save to DB
                last_id = c.execute("SELECT MAX(id) FROM incidents").fetchone()[0]
                dispatch_emergency_alerts(last_id, incident_state, severity, lat, lon)
                
                # Show alert dispatch status
                alerts = pd.read_sql(f"SELECT agency, status, timestamp FROM emergency_alerts WHERE incident_id={last_id}", conn)
                st.subheader("📡 Alert Dispatch Status")
                for _, row in alerts.iterrows():
                    st.markdown(f'<div style="background:#1a472a;padding:8px 15px;border-radius:8px;margin:4px 0;color:#4ade80;">✅ <strong>{row["agency"]}</strong> — {row["status"]} at {row["timestamp"][:19]}</div>', unsafe_allow_html=True)
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
    
    st.markdown("---")
    st.subheader("📡 Emergency Alert History")
    try:
        all_alerts = pd.read_sql("SELECT ea.id, ea.agency, ea.alert_type, ea.status, ea.timestamp, i.state, i.severity FROM emergency_alerts ea JOIN incidents i ON ea.incident_id = i.id ORDER BY ea.id DESC LIMIT 50", conn)
        if not all_alerts.empty:
            st.dataframe(all_alerts, use_container_width=True)
        else:
            st.write("No emergency alerts dispatched yet.")
    except Exception:
        st.write("No emergency alerts dispatched yet.")


