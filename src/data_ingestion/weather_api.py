"""
Weather API Client for fetching meteorological data from Open-Meteo.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

class OpenMeteoClient:
    """Client for fetching weather data from Open-Meteo API."""

    BASE_URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
    BASE_URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, cache_dir: str = 'data/raw/weather'):
        """
        Initialize the OpenMeteo client.

        Args:
            cache_dir (str): Directory to store cached JSON responses.
        """
        self.cache_dir = cache_dir
        self.session = requests.Session()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str, params: Dict[str, Any]) -> str:
        """Generate a deterministic cache file path based on url and params."""
        # Simple hash of url and params for caching
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        safe_param_str = "".join(c if c.isalnum() else "_" for c in param_str)
        filename = f"cache_{hash(url)}_{safe_param_str}.json"
        return os.path.join(self.cache_dir, filename)

    def _make_request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to make API requests with retry logic and caching.

        Args:
            url (str): API endpoint.
            params (Dict[str, Any]): Query parameters.

        Returns:
            Dict[str, Any]: Parsed JSON response.
        """
        cache_path = self._get_cache_path(url, params)
        
        # Check cache (30 minutes TTL)
        if os.path.exists(cache_path):
            file_mod_time = os.path.getmtime(cache_path)
            if time.time() - file_mod_time < 1800:
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_path}: {e}")

        # Retry logic with exponential backoff
        max_retries = 3
        backoff_factor = 2

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Save to cache
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(data, f)
                except Exception as e:
                    logger.warning(f"Failed to write cache file {cache_path}: {e}")

                return data
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff_factor ** attempt)
        
        return {}

    def fetch_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get current weather conditions.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            Dict[str, Any]: Current weather metrics.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,rain,relative_humidity_2m,wind_speed_10m",
            "hourly": "precipitation_probability,soil_moisture_0_to_7cm",
            "timezone": "auto"
        }
        
        data = self._make_request(self.BASE_URL_FORECAST, params)
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        
        # Extract the first valid hourly metrics as approximations for current
        precip_prob = hourly.get("precipitation_probability", [0])[0] if hourly.get("precipitation_probability") else 0
        soil_moisture = hourly.get("soil_moisture_0_to_7cm", [0.0])[0] if hourly.get("soil_moisture_0_to_7cm") else 0.0
        
        return {
            "temperature": current.get("temperature_2m", 0.0),
            "rainfall": current.get("rain", 0.0),
            "precipitation_probability": precip_prob,
            "soil_moisture": soil_moisture,
            "wind_speed": current.get("wind_speed_10m", 0.0),
            "relative_humidity": current.get("relative_humidity_2m", 0.0)
        }

    def fetch_hourly_forecast(self, lat: float, lon: float, days: int = 7) -> pd.DataFrame:
        """
        Get hourly weather forecast.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            days (int): Number of days to forecast.

        Returns:
            pd.DataFrame: Hourly forecast data.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,rain,precipitation_probability,soil_moisture_0_to_7cm",
            "forecast_days": days,
            "timezone": "auto"
        }
        
        data = self._make_request(self.BASE_URL_FORECAST, params)
        hourly = data.get("hourly", {})
        
        if not hourly:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly.get("time", [])),
            "temperature": hourly.get("temperature_2m", []),
            "rain": hourly.get("rain", []),
            "precipitation_probability": hourly.get("precipitation_probability", []),
            "soil_moisture": hourly.get("soil_moisture_0_to_7cm", [])
        })
        
        return df

    def fetch_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical daily weather data from the archive API.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Historical daily data.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "rain_sum,temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration",
            "timezone": "auto"
        }
        
        data = self._make_request(self.BASE_URL_ARCHIVE, params)
        daily = data.get("daily", {})
        
        if not daily:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "rain_sum": daily.get("rain_sum", []),
            "temperature_max": daily.get("temperature_2m_max", []),
            "temperature_min": daily.get("temperature_2m_min", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "et0_fao_evapotranspiration": daily.get("et0_fao_evapotranspiration", [])
        })
        
        return df

    def fetch_batch_current(self, locations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Fetch current weather for multiple locations.

        Args:
            locations (List[Dict[str, Any]]): List of dicts with 'name', 'lat', 'lon'.

        Returns:
            pd.DataFrame: Weather data for all locations.
        """
        results = []
        for loc in locations:
            try:
                weather = self.fetch_current_weather(loc['lat'], loc['lon'])
                row = {"name": loc.get("name", "Unknown"), "lat": loc['lat'], "lon": loc['lon']}
                row.update(weather)
                results.append(row)
            except Exception as e:
                logger.error(f"Failed to fetch weather for {loc}: {e}")
                
        return pd.DataFrame(results)
