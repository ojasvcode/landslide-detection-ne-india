"""
Earthquake API Client for fetching seismic data from USGS.
"""
import os
import json
import time
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

class USGSEarthquakeClient:
    """Client for fetching earthquake data from USGS."""

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    # NE India bounding box
    MIN_LAT, MAX_LAT = 20.0, 30.0
    MIN_LON, MAX_LON = 86.0, 99.0

    def __init__(self, cache_dir: str = 'data/raw/earthquakes'):
        """
        Initialize the USGS Earthquake client.

        Args:
            cache_dir (str): Directory to store cached responses.
        """
        self.cache_dir = cache_dir
        self.session = requests.Session()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with retry logic and simple caching."""
        # Simple cache key
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        safe_param_str = "".join(c if c.isalnum() else "_" for c in param_str)
        cache_path = os.path.join(self.cache_dir, f"usgs_{hash(safe_param_str)}.json")

        # Check cache (1 hour TTL)
        if os.path.exists(cache_path):
            if time.time() - os.path.getmtime(cache_path) < 3600:
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                    
                return data
            except requests.exceptions.RequestException as e:
                logger.error(f"USGS request failed (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return {}

    def _parse_geojson(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse USGS GeoJSON response into a DataFrame."""
        features = data.get("features", [])
        records = []
        
        for feature in features:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])
            
            # coords: [longitude, latitude, depth]
            records.append({
                "event_id": feature.get("id"),
                "datetime": pd.to_datetime(props.get("time"), unit="ms") if props.get("time") else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "longitude": coords[0] if len(coords) > 0 else None,
                "depth_km": coords[2] if len(coords) > 2 else None,
                "magnitude": props.get("mag"),
                "place": props.get("place")
            })
            
        return pd.DataFrame(records)

    def fetch_recent_earthquakes(self, days: int = 30, min_magnitude: float = 2.5) -> pd.DataFrame:
        """
        Fetch recent earthquakes within the NE India bounding box.
        
        Args:
            days (int): Number of past days to query.
            min_magnitude (float): Minimum magnitude threshold.

        Returns:
            pd.DataFrame: DataFrame containing earthquake records.
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        return self.fetch_earthquakes_daterange(
            start_date=start_time.strftime("%Y-%m-%d"),
            end_date=end_time.strftime("%Y-%m-%d"),
            min_magnitude=min_magnitude
        )

    def fetch_earthquakes_daterange(self, start_date: str, end_date: str, min_magnitude: float = 2.0) -> pd.DataFrame:
        """
        Fetch earthquakes for a specific date range within NE India.

        Args:
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            min_magnitude (float): Minimum magnitude threshold.

        Returns:
            pd.DataFrame: DataFrame containing earthquake records.
        """
        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": end_date,
            "minmagnitude": min_magnitude,
            "minlatitude": self.MIN_LAT,
            "maxlatitude": self.MAX_LAT,
            "minlongitude": self.MIN_LON,
            "maxlongitude": self.MAX_LON
        }
        
        data = self._make_request(params)
        return self._parse_geojson(data)

    def calculate_seismic_proximity(self, target_lat: float, target_lon: float, earthquakes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate proximity metrics to earthquakes for a specific location.

        Args:
            target_lat (float): Target latitude.
            target_lon (float): Target longitude.
            earthquakes_df (pd.DataFrame): DataFrame of earthquakes.

        Returns:
            Dict[str, Any]: Calculated seismic features.
        """
        if earthquakes_df.empty:
            return {
                "nearest_eq_distance_km": float('inf'),
                "nearest_eq_magnitude": 0.0,
                "eq_count_100km": 0,
                "max_magnitude_200km": 0.0
            }

        distances = []
        for _, row in earthquakes_df.iterrows():
            dist = haversine_distance(target_lat, target_lon, row["latitude"], row["longitude"])
            distances.append((dist, row["magnitude"]))

        distances.sort(key=lambda x: x[0])
        
        nearest = distances[0]
        count_100km = sum(1 for d, m in distances if d <= 100.0)
        mags_200km = [m for d, m in distances if d <= 200.0 and pd.notna(m)]
        max_mag_200km = max(mags_200km) if mags_200km else 0.0
        
        return {
            "nearest_eq_distance_km": nearest[0],
            "nearest_eq_magnitude": nearest[1],
            "eq_count_100km": count_100km,
            "max_magnitude_200km": max_mag_200km
        }

    def get_seismic_features(self, lat: float, lon: float, days: int = 90) -> Dict[str, Any]:
        """
        Convenience method to fetch recent quakes and compute features.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            days (int): Lookback period in days.

        Returns:
            Dict[str, Any]: Seismic features for the location.
        """
        eq_df = self.fetch_recent_earthquakes(days=days, min_magnitude=2.0)
        return self.calculate_seismic_proximity(lat, lon, eq_df)
