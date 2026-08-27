import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime

from .terrain_features import TerrainFeatureExtractor
from .rainfall_features import RainfallFeatureExtractor

logger = logging.getLogger(__name__)

class CompositeFeatureBuilder:
    """Builds the composite feature matrix combining terrain and rainfall."""

    FEATURE_COLUMNS = [
        'elevation', 'slope', 'aspect', 'curvature', 'twi', 'relative_relief',
        'rainfall_24h', 'rainfall_3day', 'rainfall_7day', 'rainfall_15day', 'rainfall_30day',
        'rainfall_intensity_max', 'consecutive_wet_days', 'soil_moisture', 'rainfall_anomaly_pct',
        'slope_x_rainfall', 'moisture_x_slope', 'trigger_index', 'vulnerability_score'
    ]

    def __init__(self, terrain_extractor: Optional[TerrainFeatureExtractor] = None, 
                 rainfall_extractor: Optional[RainfallFeatureExtractor] = None):
        """
        Initialize with extractors.
        """
        self.terrain_extractor = terrain_extractor if terrain_extractor is not None else TerrainFeatureExtractor()
        self.rainfall_extractor = rainfall_extractor if rainfall_extractor is not None else RainfallFeatureExtractor()
        logger.info("CompositeFeatureBuilder initialized.")

    def compute_landslide_trigger_index(self, rainfall_intensity: float, antecedent_rain_7d: float, 
                                        soil_moisture: float, slope: float) -> float:
        """
        Computes a combined trigger index (0-1 scale).
        Higher value = more likely to trigger.
        """
        # Normalize features against arbitrary heavy maximums
        norm_rain_int = min(rainfall_intensity / 50.0, 1.0)
        norm_ant_rain = min(antecedent_rain_7d / 300.0, 1.0)
        norm_moisture = min(soil_moisture / 100.0, 1.0)
        norm_slope = min(slope / 60.0, 1.0)
        
        # Weighted combination
        index = (norm_rain_int * 0.4) + (norm_ant_rain * 0.3) + (norm_moisture * 0.2) + (norm_slope * 0.1)
        return min(index, 1.0)

    def compute_vulnerability_score(self, slope: float, elevation: float, curvature: float, land_cover_factor: float = 1.0) -> float:
        """
        Computes static vulnerability score (0-1) based on terrain.
        """
        norm_slope = min(slope / 60.0, 1.0)
        # Higher elevations tend to have steeper gradients, but extremely high might be rocky/stable
        norm_elevation = np.clip((elevation - 500) / 3000.0, 0.0, 1.0) 
        norm_curvature = np.clip(abs(curvature), 0.0, 1.0)
        
        score = (norm_slope * 0.6) + (norm_elevation * 0.2) + (norm_curvature * 0.1) + (land_cover_factor * 0.1)
        return min(score, 1.0)

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds computed interaction features to the dataframe.
        """
        df_out = df.copy()
        
        # Cross features
        df_out['slope_x_rainfall'] = df_out['slope'] * df_out['rainfall_24h']
        df_out['moisture_x_slope'] = df_out['soil_moisture'] * df_out['slope']
        
        # Computed indices
        df_out['trigger_index'] = df_out.apply(
            lambda row: self.compute_landslide_trigger_index(
                row['rainfall_intensity_max'], 
                row['rainfall_7day'], 
                row['soil_moisture'], 
                row['slope']
            ), axis=1
        )
        
        df_out['vulnerability_score'] = df_out.apply(
            lambda row: self.compute_vulnerability_score(
                row['slope'], 
                row['elevation'], 
                row['curvature']
            ), axis=1
        )
        
        return df_out

    def build_feature_matrix(self, locations_df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Builds complete feature matrix for all locations.
        """
        logger.info(f"Building feature matrix for {len(locations_df)} locations.")
        
        # Extract base features
        terrain_df = self.terrain_extractor.extract_batch(locations_df)
        
        # Only keep lat/lon to avoid duplicating columns before merge
        rain_locs = locations_df[['lat', 'lon']].copy()
        rain_df = self.rainfall_extractor.extract_batch(rain_locs, reference_date)
        
        # Merge them (they are aligned since we didn't drop rows, but merge on index for safety)
        combined_df = pd.merge(terrain_df, rain_df.drop(columns=['lat', 'lon']), left_index=True, right_index=True)
        
        # Add interactions
        final_df = self.add_interaction_features(combined_df)
        
        return final_df

    def get_feature_names(self) -> List[str]:
        """
        Returns ordered list of all feature column names used by the model.
        """
        return self.FEATURE_COLUMNS.copy()
