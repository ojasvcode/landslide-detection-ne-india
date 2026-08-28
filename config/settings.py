import os
from enum import Enum

NE_INDIA_BBOX = {
    'min_lat': 21.0,
    'max_lat': 29.5,
    'min_lon': 87.0,
    'max_lon': 98.0
}

GRID_RESOLUTION = 0.1


NE_STATES = [
    'Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 
    'Mizoram', 'Nagaland', 'Sikkim', 'Tripura'
]

ALL_STATES = [
    "Andaman & Nicobar", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", 
    "Chandigarh", "Chhattisgarh", "Dadra & Nagar Haveli", "Daman & Diu", "Delhi", 
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", 
    "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", 
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", 
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
    "Uttarakhand", "West Bengal"
]

NON_NE_STATES = [s for s in ALL_STATES if s not in NE_STATES]

API_ENDPOINTS = {
    'open_meteo': 'https://api.open-meteo.com/v1/forecast',
    'open_meteo_historical': 'https://archive-api.open-meteo.com/v1/archive',
    'usgs_earthquake': 'https://earthquake.usgs.gov/fdsnws/event/1/query',
    'bhuvan_wms': 'https://bhuvan-vec1.nrsc.gov.in/bhuvan/gwc/service/wms/'
}

RISK_THRESHOLDS = {
    'LOW': 0.2,
    'MODERATE': 0.4,
    'HIGH': 0.6,
    'VERY_HIGH': 0.8,
    'SEVERE': 1.0
}

class RiskLevel(Enum):
    LOW = 'LOW'
    MODERATE = 'MODERATE'
    HIGH = 'HIGH'
    VERY_HIGH = 'VERY_HIGH'
    SEVERE = 'SEVERE'

MONITORING_LOCATIONS = [
    {'name': 'Guwahati', 'lat': 26.14, 'lon': 91.73, 'state': 'Assam'},
    {'name': 'Shillong', 'lat': 25.57, 'lon': 91.88, 'state': 'Meghalaya'},
    {'name': 'Gangtok', 'lat': 27.33, 'lon': 88.62, 'state': 'Sikkim'},
    {'name': 'Itanagar', 'lat': 27.08, 'lon': 93.61, 'state': 'Arunachal Pradesh'},
    {'name': 'Imphal', 'lat': 24.82, 'lon': 93.95, 'state': 'Manipur'},
    {'name': 'Aizawl', 'lat': 23.73, 'lon': 92.72, 'state': 'Mizoram'},
    {'name': 'Kohima', 'lat': 25.67, 'lon': 94.10, 'state': 'Nagaland'},
    {'name': 'Agartala', 'lat': 23.83, 'lon': 91.28, 'state': 'Tripura'},
    {'name': 'Tawang', 'lat': 27.58, 'lon': 91.86, 'state': 'Arunachal Pradesh'},
    {'name': 'Bomdila', 'lat': 27.26, 'lon': 92.42, 'state': 'Arunachal Pradesh'},
    {'name': 'Pasighat', 'lat': 28.06, 'lon': 95.33, 'state': 'Arunachal Pradesh'},
    {'name': 'Ziro', 'lat': 27.53, 'lon': 93.83, 'state': 'Arunachal Pradesh'},
    {'name': 'Dibrugarh', 'lat': 27.47, 'lon': 94.91, 'state': 'Assam'},
    {'name': 'Tezpur', 'lat': 26.63, 'lon': 92.78, 'state': 'Assam'},
    {'name': 'Silchar', 'lat': 24.82, 'lon': 92.79, 'state': 'Assam'},
    {'name': 'Haflong', 'lat': 25.18, 'lon': 93.01, 'state': 'Assam'},
    {'name': 'Jowai', 'lat': 25.45, 'lon': 92.20, 'state': 'Meghalaya'},
    {'name': 'Cherrapunji', 'lat': 25.27, 'lon': 91.73, 'state': 'Meghalaya'},
    {'name': 'Tura', 'lat': 25.52, 'lon': 90.22, 'state': 'Meghalaya'},
    {'name': 'Nongstoin', 'lat': 25.52, 'lon': 91.27, 'state': 'Meghalaya'},
    {'name': 'Churachandpur', 'lat': 24.33, 'lon': 93.68, 'state': 'Manipur'},
    {'name': 'Ukhrul', 'lat': 25.12, 'lon': 94.37, 'state': 'Manipur'},
    {'name': 'Senapati', 'lat': 25.27, 'lon': 94.02, 'state': 'Manipur'},
    {'name': 'Champhai', 'lat': 23.47, 'lon': 93.33, 'state': 'Mizoram'},
    {'name': 'Lunglei', 'lat': 22.88, 'lon': 92.73, 'state': 'Mizoram'},
    {'name': 'Serchhip', 'lat': 23.30, 'lon': 92.83, 'state': 'Mizoram'},
    {'name': 'Mokokchung', 'lat': 26.33, 'lon': 94.53, 'state': 'Nagaland'},
    {'name': 'Dimapur', 'lat': 25.90, 'lon': 93.73, 'state': 'Nagaland'},
    {'name': 'Mon', 'lat': 26.75, 'lon': 95.10, 'state': 'Nagaland'},
    {'name': 'Wokha', 'lat': 26.10, 'lon': 94.27, 'state': 'Nagaland'},
    {'name': 'Namchi', 'lat': 27.17, 'lon': 88.35, 'state': 'Sikkim'},
    {'name': 'Ravangla', 'lat': 27.30, 'lon': 88.36, 'state': 'Sikkim'},
    {'name': 'Pelling', 'lat': 27.30, 'lon': 88.24, 'state': 'Sikkim'},
    {'name': 'Mangan', 'lat': 27.50, 'lon': 88.53, 'state': 'Sikkim'},
    {'name': 'Dharmanagar', 'lat': 24.37, 'lon': 92.17, 'state': 'Tripura'},
    {'name': 'Kailashahar', 'lat': 24.32, 'lon': 92.01, 'state': 'Tripura'},
    {'name': 'Ambassa', 'lat': 23.92, 'lon': 91.85, 'state': 'Tripura'},
    {'name': 'Udaipur-Tripura', 'lat': 23.53, 'lon': 91.48, 'state': 'Tripura'},
    {'name': 'Jorhat', 'lat': 26.75, 'lon': 94.22, 'state': 'Assam'},
    {'name': 'Nagaon', 'lat': 26.33, 'lon': 92.68, 'state': 'Assam'},
    {'name': 'Nalbari', 'lat': 26.43, 'lon': 91.43, 'state': 'Assam'},
    {'name': 'Bongaigaon', 'lat': 26.47, 'lon': 90.56, 'state': 'Assam'},
    {'name': 'Goalpara', 'lat': 26.17, 'lon': 90.62, 'state': 'Assam'},
    {'name': 'Diphu', 'lat': 25.83, 'lon': 93.42, 'state': 'Assam'},
    {'name': 'North Lakhimpur', 'lat': 27.23, 'lon': 94.10, 'state': 'Assam'},
    {'name': 'Doom Dooma', 'lat': 27.57, 'lon': 95.57, 'state': 'Assam'},
    {'name': 'Lumding', 'lat': 25.75, 'lon': 93.17, 'state': 'Assam'},
    {'name': 'Dima Hasao', 'lat': 25.18, 'lon': 93.01, 'state': 'Assam'}
]

MODEL_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'test_size': 0.15,
    'random_state': 42
}

CACHE_DIR = {
    'raw': 'data/raw',
    'processed': 'data/processed'
}

DEFAULT_LOOKBACK_DAYS = 30
