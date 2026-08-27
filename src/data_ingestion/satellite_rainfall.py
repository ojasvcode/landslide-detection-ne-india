"""
Satellite Rainfall Processor for analyzing IMERG and historical rainfall data.
"""
import os
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SatelliteRainfallProcessor:
    """Processor for NASA GPM IMERG and other satellite rainfall data."""

    # Historical annual normals (in mm) based on long-term station data
    # (Cherrapunji/Mawsynram are exceptionally high, others proportional)
    NE_INDIA_RAINFALL_NORMALS = {
        "Cherrapunji": 11777.0,
        "Mawsynram": 11871.0,
        "Guwahati": 1600.0,
        "Shillong": 2100.0,
        "Imphal": 1400.0,
        "Agartala": 2200.0,
        "Aizawl": 2000.0,
        "Kohima": 1800.0,
        "Itanagar": 3000.0,
        "Gangtok": 3500.0,
        "Regional_Avg": 2500.0
    }

    def __init__(self, data_dir: str = 'data/raw/satellite'):
        """
        Initialize the Satellite Rainfall Processor.

        Args:
            data_dir (str): Directory for satellite data.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def parse_imerg_hdf5(self, filepath: str) -> pd.DataFrame:
        """
        Parse a NASA GPM IMERG HDF5 file and extract precipitation grid.
        Filters to NE India bounding box (Lat 21.0-29.5, Lon 87.0-98.0).

        Args:
            filepath (str): Path to the HDF5 file.

        Returns:
            pd.DataFrame: DataFrame containing lat, lon, precipitation_mm.
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py is not installed. Returning empty DataFrame.")
            return pd.DataFrame(columns=["lat", "lon", "precipitation_mm"])

        if not os.path.exists(filepath):
            logger.error(f"File {filepath} not found.")
            return pd.DataFrame(columns=["lat", "lon", "precipitation_mm"])

        records = []
        try:
            with h5py.File(filepath, 'r') as f:
                # IMERG typically stores data under Grid/precipitationCal
                precip = f['Grid']['precipitationCal'][:]
                lon_arr = f['Grid']['lon'][:]
                lat_arr = f['Grid']['lat'][:]

                # IMERG data shape is often [time, lon, lat]
                if len(precip.shape) == 3:
                    precip = precip[0]

                # Filter to NE India bbox
                lat_mask = (lat_arr >= 21.0) & (lat_arr <= 29.5)
                lon_mask = (lon_arr >= 87.0) & (lon_arr <= 98.0)

                valid_lats = lat_arr[lat_mask]
                valid_lons = lon_arr[lon_mask]
                
                lat_idx = np.where(lat_mask)[0]
                lon_idx = np.where(lon_mask)[0]

                for i, l_lon in zip(lon_idx, valid_lons):
                    for j, l_lat in zip(lat_idx, valid_lats):
                        val = precip[i, j]
                        if val >= 0:  # Filter out fill values
                            records.append({
                                "lat": l_lat,
                                "lon": l_lon,
                                "precipitation_mm": val
                            })
        except Exception as e:
            logger.error(f"Error parsing IMERG file: {e}")

        return pd.DataFrame(records)

    def compute_monthly_stats(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily precipitation data into monthly statistics.

        Args:
            daily_data (pd.DataFrame): DataFrame containing 'date' and 'precipitation_mm'.

        Returns:
            pd.DataFrame: Monthly aggregated statistics.
        """
        if daily_data.empty or 'date' not in daily_data.columns:
            return pd.DataFrame()

        df = daily_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')

        monthly = df.groupby('month').agg(
            mean_precip=('precipitation_mm', 'mean'),
            max_precip=('precipitation_mm', 'max'),
            total_precip=('precipitation_mm', 'sum'),
            rainy_days=('precipitation_mm', lambda x: (x > 0.1).sum())
        ).reset_index()

        return monthly

    def compute_seasonal_anomaly(self, current_rainfall: float, historical_mean: float) -> float:
        """
        Compute rainfall deviation from normal as a percentage.

        Args:
            current_rainfall (float): Current rainfall total.
            historical_mean (float): Historical mean rainfall total.

        Returns:
            float: Percentage anomaly (positive means excess, negative means deficit).
        """
        if historical_mean <= 0:
            return 0.0
        return ((current_rainfall - historical_mean) / historical_mean) * 100.0

    def get_historical_rainfall_stats(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Return long-term rainfall statistics for a location. 
        Uses regional heuristics based on coordinates.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            Dict[str, Any]: Historical rainfall statistics.
        """
        # Hardcoded approximations for NE India geography
        if 25.0 < lat < 25.5 and 91.5 < lon < 92.0:
            # Proximity to Mawsynram / Cherrapunji in Meghalaya
            annual_normal = (self.NE_INDIA_RAINFALL_NORMALS["Cherrapunji"] + 
                             self.NE_INDIA_RAINFALL_NORMALS["Mawsynram"]) / 2
        elif 27.0 < lat < 28.0 and 88.0 < lon < 89.0:
            # Sikkim region
            annual_normal = self.NE_INDIA_RAINFALL_NORMALS["Gangtok"]
        elif 26.5 < lat < 28.0 and 92.0 < lon < 96.0:
            # Arunachal region
            annual_normal = self.NE_INDIA_RAINFALL_NORMALS["Itanagar"]
        else:
            annual_normal = self.NE_INDIA_RAINFALL_NORMALS["Regional_Avg"]

        # Typical monsoon pattern: ~80% of rain falls in Jun-Sep
        monsoon_normal = annual_normal * 0.8
        
        return {
            "annual_normal_mm": annual_normal,
            "monsoon_normal_mm": monsoon_normal,
            "daily_mean_mm": annual_normal / 365.25
        }
