import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DummyDEMProcessor:
    """Mock DEM Processor in case one is not provided."""
    def get_elevation(self, lat: float, lon: float) -> float:
        return np.random.uniform(100, 4000)
    def get_slope(self, lat: float, lon: float) -> float:
        return np.random.uniform(0, 60)
    def get_aspect(self, lat: float, lon: float) -> float:
        return np.random.uniform(0, 360)
    def get_curvature(self, lat: float, lon: float) -> float:
        return np.random.uniform(-1, 1)
    def get_twi(self, lat: float, lon: float) -> float:
        return np.random.uniform(5, 15)
    def get_relative_relief(self, lat: float, lon: float) -> float:
        return np.random.uniform(50, 500)


class TerrainFeatureExtractor:
    """Extracts terrain-based features for landslide detection."""

    def __init__(self, dem_processor: Optional[Any] = None):
        """
        Initialize the TerrainFeatureExtractor.
        
        Args:
            dem_processor: Optional instance to process DEM data.
        """
        self.dem_processor = dem_processor if dem_processor is not None else DummyDEMProcessor()
        logger.info("TerrainFeatureExtractor initialized.")

    def classify_slope(self, slope_deg: float) -> str:
        """
        Classifies slope into categories.
        
        Args:
            slope_deg: Slope in degrees.
            
        Returns:
            Slope class string.
        """
        if slope_deg < 5:
            return 'flat'
        elif slope_deg <= 15:
            return 'gentle'
        elif slope_deg <= 25:
            return 'moderate'
        elif slope_deg <= 35:
            return 'steep'
        else:
            return 'very_steep'

    def classify_aspect(self, aspect_deg: float) -> str:
        """
        Classifies aspect into 8 cardinal directions.
        
        Args:
            aspect_deg: Aspect in degrees (0-360).
            
        Returns:
            Aspect class string.
        """
        if aspect_deg < 0 or aspect_deg > 360:
            return 'flat'
        
        if (0 <= aspect_deg <= 22.5) or (337.5 < aspect_deg <= 360):
            return 'N'
        elif 22.5 < aspect_deg <= 67.5:
            return 'NE'
        elif 67.5 < aspect_deg <= 112.5:
            return 'E'
        elif 112.5 < aspect_deg <= 157.5:
            return 'SE'
        elif 157.5 < aspect_deg <= 202.5:
            return 'S'
        elif 202.5 < aspect_deg <= 247.5:
            return 'SW'
        elif 247.5 < aspect_deg <= 292.5:
            return 'W'
        elif 292.5 < aspect_deg <= 337.5:
            return 'NW'
        return 'flat'

    def compute_slope_stability_index(self, slope: float, soil_moisture: float, rainfall_intensity: float) -> float:
        """
        Computes a simple proxy for the Factor of Safety.
        Lower values indicate less stable slopes.
        
        Args:
            slope: Slope in degrees.
            soil_moisture: Soil moisture percentage (0-100).
            rainfall_intensity: Rainfall intensity in mm/hr.
            
        Returns:
            Stability index.
        """
        # A pseudo-empirical relationship for proxy Factor of Safety (FoS)
        # Assumes base stability of 2.0, degraded by slope, moisture, and rainfall.
        base_stability = 2.0
        slope_penalty = math.tan(math.radians(slope)) if slope < 80 else 5.0
        moisture_penalty = soil_moisture / 100.0
        rainfall_penalty = (rainfall_intensity / 50.0)  # Assuming 50mm/hr is extreme

        stability = base_stability - slope_penalty - (moisture_penalty * 0.5) - rainfall_penalty
        return max(0.1, stability)  # Floor at 0.1

    def extract_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Extracts all terrain features for a specific coordinate point.
        
        Args:
            lat: Latitude.
            lon: Longitude.
            
        Returns:
            Dictionary containing elevation, slope, aspect, curvature, twi, 
            relative_relief, slope_class, and aspect_class.
        """
        try:
            elevation = self.dem_processor.get_elevation(lat, lon)
            slope = self.dem_processor.get_slope(lat, lon)
            aspect = self.dem_processor.get_aspect(lat, lon)
            curvature = self.dem_processor.get_curvature(lat, lon)
            twi = self.dem_processor.get_twi(lat, lon)
            relative_relief = self.dem_processor.get_relative_relief(lat, lon)

            features = {
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'curvature': curvature,
                'twi': twi,
                'relative_relief': relative_relief,
                'slope_class': self.classify_slope(slope),
                'aspect_class': self.classify_aspect(aspect)
            }
            return features
        except Exception as e:
            logger.error(f"Error extracting terrain features at ({lat}, {lon}): {e}")
            raise

    def extract_batch(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts terrain features for a batch of locations.
        
        Args:
            locations_df: DataFrame containing 'lat' and 'lon' columns.
            
        Returns:
            DataFrame with extracted terrain features appended.
        """
        if 'lat' not in locations_df.columns or 'lon' not in locations_df.columns:
            raise ValueError("Input DataFrame must contain 'lat' and 'lon' columns.")
        
        features_list = []
        for _, row in locations_df.iterrows():
            features = self.extract_features(row['lat'], row['lon'])
            features_list.append(features)
            
        features_df = pd.DataFrame(features_list)
        return pd.concat([locations_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
