import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DummyWeatherClient:
    """Mock Weather Client in case one is not provided."""
    def get_historical_daily(self, lat: float, lon: float, days: int, end_date: datetime) -> pd.Series:
        dates = pd.date_range(end=end_date, periods=days)
        return pd.Series(np.random.gamma(2, 2, days), index=dates)
    
    def get_historical_hourly(self, lat: float, lon: float, hours: int, end_date: datetime) -> pd.Series:
        dates = pd.date_range(end=end_date, periods=hours, freq='H')
        return pd.Series(np.random.exponential(1.5, hours), index=dates)
        
    def get_monthly_normal(self, lat: float, lon: float, month: int) -> float:
        return np.random.uniform(50, 400)
    
    def get_soil_moisture(self, lat: float, lon: float, date: datetime) -> float:
        return np.random.uniform(10, 90)


class RainfallFeatureExtractor:
    """Extracts rainfall-based features for landslide detection."""

    def __init__(self, weather_client: Optional[Any] = None):
        """
        Initialize RainfallFeatureExtractor.
        
        Args:
            weather_client: Optional instance of OpenMeteoClient or similar.
        """
        self.weather_client = weather_client if weather_client is not None else DummyWeatherClient()
        logger.info("RainfallFeatureExtractor initialized.")

    def compute_antecedent_rainfall(self, daily_rainfall_series: pd.Series, windows: List[int] = [1, 3, 7, 15, 30]) -> Dict[str, float]:
        """
        Computes cumulative rainfall for various windows from a daily series.
        
        Args:
            daily_rainfall_series: Series of daily rainfall ending on the target date.
            windows: List of integer day windows.
            
        Returns:
            Dict mapping window strings to total rainfall.
        """
        results = {}
        for w in windows:
            if len(daily_rainfall_series) >= w:
                results[f'rainfall_{w}day' if w > 1 else 'rainfall_24h'] = float(daily_rainfall_series.iloc[-w:].sum())
            else:
                results[f'rainfall_{w}day' if w > 1 else 'rainfall_24h'] = float(daily_rainfall_series.sum())
        return results

    def compute_consecutive_wet_days(self, daily_rainfall_series: pd.Series, threshold_mm: float = 2.5) -> int:
        """
        Counts consecutive days with rain > threshold ending at the last day.
        
        Args:
            daily_rainfall_series: Series of daily rainfall.
            threshold_mm: Rainfall amount to be considered a 'wet day'.
            
        Returns:
            Number of consecutive wet days.
        """
        wet_days = (daily_rainfall_series >= threshold_mm).astype(int)
        
        # Count backwards from the last day
        consecutive = 0
        for val in reversed(wet_days.values):
            if val == 1:
                consecutive += 1
            else:
                break
        return consecutive

    def compute_rainfall_intensity(self, hourly_rainfall_series: pd.Series) -> float:
        """
        Computes peak hourly rainfall intensity.
        
        Args:
            hourly_rainfall_series: Series of hourly rainfall amounts.
            
        Returns:
            Peak intensity in mm/hr.
        """
        return float(hourly_rainfall_series.max()) if not hourly_rainfall_series.empty else 0.0

    def compute_rainfall_anomaly(self, current_monthly_total: float, lat: float, lon: float, date: datetime) -> float:
        """
        Percentage deviation from historical normal.
        
        Args:
            current_monthly_total: Accumulated rainfall for the current month.
            lat: Latitude.
            lon: Longitude.
            date: Date to get the month from.
            
        Returns:
            Percentage deviation (0 = normal, >0 = above normal).
        """
        normal = self.weather_client.get_monthly_normal(lat, lon, date.month)
        if normal == 0:
            return 0.0
        return ((current_monthly_total - normal) / normal) * 100.0

    def get_critical_rainfall_threshold(self, slope_deg: float) -> float:
        """
        Returns critical 24h rainfall threshold (in mm) that could trigger a landslide based on slope.
        Empirical relationship: Threshold decreases as slope increases.
        
        Args:
            slope_deg: Slope in degrees.
            
        Returns:
            Critical rainfall amount in mm.
        """
        # Basic empirical formula (e.g., Threshold = 150 - 2.5 * slope)
        # Capped to reasonable minimums and maximums
        threshold = 150.0 - (2.5 * slope_deg)
        return float(np.clip(threshold, 30.0, 250.0))

    def extract_features(self, lat: float, lon: float, reference_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Extracts all rainfall features for a point at a specific date.
        
        Args:
            lat: Latitude.
            lon: Longitude.
            reference_date: Target date for calculation (defaults to now).
            
        Returns:
            Dict of rainfall features.
        """
        if reference_date is None:
            reference_date = datetime.now()
            
        try:
            daily_series = self.weather_client.get_historical_daily(lat, lon, 30, reference_date)
            hourly_series = self.weather_client.get_historical_hourly(lat, lon, 24, reference_date)
            
            # Antecedent rainfalls
            antecedent = self.compute_antecedent_rainfall(daily_series)
            
            # Consecutive wet days
            cwd = self.compute_consecutive_wet_days(daily_series)
            
            # Peak intensity
            intensity = self.compute_rainfall_intensity(hourly_series)
            
            # Anomaly
            month_total = daily_series.iloc[-reference_date.day:].sum()
            anomaly = self.compute_rainfall_anomaly(month_total, lat, lon, reference_date)
            
            # Soil moisture
            soil_moisture = self.weather_client.get_soil_moisture(lat, lon, reference_date)
            
            features = {
                **antecedent,
                'rainfall_intensity_max': intensity,
                'consecutive_wet_days': cwd,
                'soil_moisture': soil_moisture,
                'rainfall_anomaly_pct': anomaly
            }
            return features
            
        except Exception as e:
            logger.error(f"Error extracting rainfall features at ({lat}, {lon}): {e}")
            raise

    def extract_batch(self, locations_df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Batch extraction for multiple locations.
        
        Args:
            locations_df: DataFrame with 'lat' and 'lon'.
            reference_date: Reference date.
            
        Returns:
            DataFrame with rainfall features added.
        """
        if 'lat' not in locations_df.columns or 'lon' not in locations_df.columns:
            raise ValueError("Input DataFrame must contain 'lat' and 'lon' columns.")
            
        features_list = []
        for _, row in locations_df.iterrows():
            features = self.extract_features(row['lat'], row['lon'], reference_date)
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        return pd.concat([locations_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
