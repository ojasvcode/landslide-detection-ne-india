import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

try:
    from config.settings import RISK_THRESHOLDS, NE_INDIA_BBOX
except ImportError:
    RISK_THRESHOLDS = {'LOW': 0.0, 'MODERATE': 0.25, 'HIGH': 0.5, 'VERY_HIGH': 0.75, 'SEVERE': 0.9}
    NE_INDIA_BBOX = {"lat_min": 21.0, "lat_max": 29.5, "lon_min": 87.0, "lon_max": 98.0}

from src.feature_engineering.composite_features import CompositeFeatureBuilder

logger = logging.getLogger(__name__)

class LandslidePredictor:
    """Uses a trained model to make landslide risk predictions."""

    def __init__(self, model_path: Optional[str] = None, model: Optional[Any] = None, scaler: Optional[Any] = None):
        if model_path:
            self.model = joblib.load(model_path)
            # Infer scaler path
            scaler_path = model_path.replace('.joblib', '_scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            
            features_path = model_path.replace('.joblib', '_features.joblib')
            self.feature_names = joblib.load(features_path)
            logger.info(f"Loaded model and scaler from {model_path}")
        elif model and scaler:
            self.model = model
            self.scaler = scaler
            self.feature_names = CompositeFeatureBuilder.FEATURE_COLUMNS
            logger.info("Initialized with provided model and scaler.")
        else:
            raise ValueError("Must provide either model_path or (model and scaler).")

    def get_risk_level(self, probability: float) -> str:
        """Converts probability to risk level string based on thresholds."""
        if probability >= RISK_THRESHOLDS.get('SEVERE', 0.9):
            return 'SEVERE'
        elif probability >= RISK_THRESHOLDS.get('VERY_HIGH', 0.75):
            return 'VERY_HIGH'
        elif probability >= RISK_THRESHOLDS.get('HIGH', 0.5):
            return 'HIGH'
        elif probability >= RISK_THRESHOLDS.get('MODERATE', 0.25):
            return 'MODERATE'
        else:
            return 'LOW'

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predicts risk for a single location."""
        # Convert dict to array following feature order
        feature_values = [features.get(col, 0.0) for col in self.feature_names]
        feature_array = np.array(feature_values).reshape(1, -1)
        
        # Scale and predict
        scaled_features = self.scaler.transform(feature_array)
        prob = float(self.model.predict_proba(scaled_features)[0, 1])
        
        return {
            'risk_probability': prob,
            'risk_level': self.get_risk_level(prob),
            'feature_values': features
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Predicts risk for multiple locations."""
        missing_cols = set(self.feature_names) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
            
        X = features_df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        
        results_df = features_df.copy()
        results_df['risk_probability'] = probs
        results_df['risk_level'] = results_df['risk_probability'].apply(self.get_risk_level)
        
        return results_df

    def predict_grid(self, bbox: Dict[str, float], resolution: float = 0.1) -> pd.DataFrame:
        """Predicts over entire grid within bbox."""
        lats = np.arange(bbox['lat_min'], bbox['lat_max'] + resolution, resolution)
        lons = np.arange(bbox['lon_min'], bbox['lon_max'] + resolution, resolution)
        
        # Create grid points
        grid_lats, grid_lons = np.meshgrid(lats, lons)
        locations_df = pd.DataFrame({
            'lat': grid_lats.flatten(),
            'lon': grid_lons.flatten()
        })
        
        logger.info(f"Generated {len(locations_df)} grid points for prediction.")
        
        # In a real implementation, you would extract features for these grid points here using CompositeFeatureBuilder
        # For completeness of this method per requirements, we will simulate the extraction if we are just demonstrating
        # but the request indicates I should rely on the user to pass a dataframe if they just want pure predictions.
        # However, the prompt specifically asks for predict_grid to return a DataFrame with lat, lon, prob, level.
        # Let's instantiate a feature builder and use it.
        builder = CompositeFeatureBuilder()
        features_df = builder.build_feature_matrix(locations_df)
        
        return self.predict_batch(features_df)[['lat', 'lon', 'risk_probability', 'risk_level']]

    def get_top_risk_locations(self, predictions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Returns top N highest risk locations."""
        return predictions_df.sort_values(by='risk_probability', ascending=False).head(n)
