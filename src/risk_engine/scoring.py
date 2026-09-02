import pandas as pd
import numpy as np
import datetime
import logging
from typing import List, Dict, Any, Optional
import os

# Each optional import gets its OWN try/except, so one module that's
# expected to be missing (e.g. src.config, which this repo never actually
# has) doesn't take down unrelated modules like our trained predictors.
# The original single shared try/except meant any one failure silently
# disabled everything below it, including working code.
try:
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
        {"name": "Gangtok", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065},
    ]
    NE_STATES = ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"]

try:
    from src.data_ingestion.weather_api import OpenMeteoClient
except ImportError:
    OpenMeteoClient = None

try:
    from src.data_ingestion.dem_processor import DEMProcessor
except ImportError:
    DEMProcessor = None

try:
    from src.data_ingestion.seismic_api import USGSClient
except ImportError:
    USGSClient = None

try:
    from src.models.predictor import LandslidePredictor
except ImportError:
    LandslidePredictor = None

try:
    from src.models.rainfall_trigger import RainfallTrigger
except ImportError:
    RainfallTrigger = None

logger = logging.getLogger(__name__)

class RiskScoringEngine:
    """
    Engine to compute real-time landslide risk scores for monitoring locations.
    Uses trained machine learning models if available, otherwise falls back to a rule-based system.
    """
    def __init__(self, model_path: Optional[str] = None, rainfall_model_path: Optional[str] = None):
        """
        Initialize the Risk Scoring Engine.

        Args:
            model_path: Path to the trained ML model directory (Model A - terrain susceptibility).
            rainfall_model_path: Path to the trained rainfall threshold model (Model B).
        """
        self.weather_client = OpenMeteoClient() if OpenMeteoClient else None
        self.dem_processor = DEMProcessor() if DEMProcessor else None
        self.seismic_client = USGSClient() if USGSClient else None

        self.model_available = False
        self.predictor = None

        if model_path and os.path.exists(model_path) and LandslidePredictor:
            try:
                self.predictor = LandslidePredictor(model_dir=model_path)
                self.model_available = True
                logger.info(f"Successfully loaded ML model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")

        if not self.model_available:
            logger.warning("No ML model loaded. Using rule-based fallback scoring.")

        self.rainfall_trigger = None
        if rainfall_model_path and os.path.exists(rainfall_model_path) and RainfallTrigger:
            try:
                self.rainfall_trigger = RainfallTrigger(rainfall_model_path)
                logger.info(f"Successfully loaded rainfall trigger model from {rainfall_model_path}")
            except Exception as e:
                logger.error(f"Failed to load rainfall trigger model from {rainfall_model_path}: {e}")

    def score_location(self, lat: float, lon: float, name: str = "Unknown", state: str = "Unknown") -> dict:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        features = {
            "lat": lat,
            "lon": lon,
            "slope": 20.0,
            "elevation": 1000.0,
            "rainfall_24h": 0.0,
            "soil_moisture": 0.3,
            "seismic_activity": 0
        }
        weather_summary = "Weather data unavailable"

        try:
            if self.rainfall_trigger:
                rain_result = self.rainfall_trigger.check(lat, lon)
                features["rainfall_24h"] = rain_result["rainfall_24h_mm"]
                features["rainfall_trigger_probability"] = rain_result["trigger_probability"]
                weather_summary = (
                    f"Rain(24h): {rain_result['rainfall_24h_mm']:.1f}mm, "
                    f"Trigger prob: {rain_result['trigger_probability']:.2f}"
                )
            elif self.weather_client:
                weather = self.weather_client.fetch_current_weather(lat, lon)
                features["rainfall_24h"] = weather.get("precipitation_mm", 0.0)
                features["soil_moisture"] = weather.get("soil_moisture_0_to_7cm_m3_m3", 0.3)
                weather_summary = f"Temp: {weather.get('temperature_2m_c', 'N/A')}\u00b0C, Rain: {features['rainfall_24h']}mm"
        except Exception as e:
            logger.error(f"Error fetching rainfall for {name}: {e}")

        try:
            if self.dem_processor:
                terrain = self.dem_processor.get_terrain_features(lat, lon)
                features["slope"] = terrain.get("slope", 20.0)
                features["elevation"] = terrain.get("elevation", 1000.0)
        except Exception as e:
            logger.error(f"Error fetching terrain for {name}: {e}")

        try:
            if self.seismic_client:
                end_time = datetime.datetime.now(datetime.timezone.utc)
                start_time = end_time - datetime.timedelta(days=30)
                quakes = self.seismic_client.get_earthquakes(start_time, end_time, lat, lon, max_radius_km=100)
                features["seismic_activity"] = 1 if not quakes.empty else 0
        except Exception as e:
            logger.error(f"Error fetching seismic data for {name}: {e}")

        if self.model_available and self.predictor:
            try:
                feature_df = pd.DataFrame([features])
                predictions = self.predictor.predict(feature_df)
                risk_probability = float(predictions["probability"].iloc[0])
                risk_level = predictions["risk_level"].iloc[0]
            except Exception as e:
                logger.error(f"Model prediction failed for {name}: {e}. Falling back to rule-based.")
                risk_probability = self.rule_based_risk(features)
                risk_level = self._assign_risk_level(risk_probability)
        else:
            risk_probability = self.rule_based_risk(features)
            risk_level = self._assign_risk_level(risk_probability)

        return {
            "name": name,
            "lat": lat,
            "lon": lon,
            "state": state,
            "risk_probability": risk_probability,
            "risk_level": risk_level,
            "contributing_factors": features,
            "timestamp": timestamp,
            "weather_summary": weather_summary
        }

    def _assign_risk_level(self, probability: float) -> str:
        if probability < 0.2:
            return "LOW"
        elif probability < 0.4:
            return "MODERATE"
        elif probability < 0.7:
            return "HIGH"
        elif probability < 0.9:
            return "VERY_HIGH"
        else:
            return "SEVERE"

    def rule_based_risk(self, features: dict) -> float:
        slope = features.get("slope", 0)
        rainfall_24h = features.get("rainfall_24h", 0)
        soil_moisture = features.get("soil_moisture", 0)
        seismic_activity = features.get("seismic_activity", 0)

        slope_risk = min(slope / 45.0, 1.0)
        rain_risk = min(rainfall_24h / 100.0, 1.0)
        moisture_risk = min(soil_moisture / 0.5, 1.0)
        seismic_risk = 0.3 if seismic_activity > 0 else 0.0

        combined = 0.30 * slope_risk + 0.35 * rain_risk + 0.20 * moisture_risk + 0.15 * seismic_risk
        return min(combined, 1.0)

    def score_all_stations(self) -> pd.DataFrame:
        scores = []
        for loc in MONITORING_LOCATIONS:
            score = self.score_location(loc["lat"], loc["lon"], loc["name"], loc["state"])
            flat_score = {**score, **score.pop("contributing_factors")}
            scores.append(flat_score)
        return pd.DataFrame(scores)

    def score_state(self, state_name: str) -> pd.DataFrame:
        scores = []
        state_locations = [loc for loc in MONITORING_LOCATIONS if loc["state"].lower() == state_name.lower()]
        for loc in state_locations:
            score = self.score_location(loc["lat"], loc["lon"], loc["name"], loc["state"])
            flat_score = {**score, **score.pop("contributing_factors")}
            scores.append(flat_score)
        return pd.DataFrame(scores)

    def get_risk_summary(self, scores_df: pd.DataFrame) -> dict:
        if scores_df.empty:
            return {}
        summary = {
            "total_stations": len(scores_df),
            "by_risk_level": scores_df["risk_level"].value_counts().to_dict(),
            "highest_risk_locations": scores_df.nlargest(5, "risk_probability")[["name", "state", "risk_probability", "risk_level"]].to_dict(orient="records"),
        }
        state_avg = scores_df.groupby("state")["risk_probability"].mean().to_dict()
        summary["state_averages"] = state_avg
        return summary

    def generate_alerts(self, scores_df: pd.DataFrame, threshold: str = 'HIGH') -> List[dict]:
        if scores_df.empty:
            return []
        risk_hierarchy = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "VERY_HIGH": 3, "SEVERE": 4}
        thresh_val = risk_hierarchy.get(threshold.upper(), 2)
        alerts = []
        for _, row in scores_df.iterrows():
            level_val = risk_hierarchy.get(row["risk_level"].upper(), 0)
            if level_val >= thresh_val:
                factors = {
                    "Rainfall": row.get("rainfall_24h", 0) / 100.0,
                    "Slope": row.get("slope", 0) / 45.0,
                    "Soil Moisture": row.get("soil_moisture", 0) / 0.5
                }
                primary_trigger = max(factors, key=factors.get) if factors else "Unknown"
                alerts.append({
                    "location": row["name"],
                    "state": row["state"],
                    "risk_level": row["risk_level"],
                    "risk_probability": row["risk_probability"],
                    "primary_trigger": primary_trigger,
                    "recommended_action": "Evacuate immediately" if row["risk_level"] in ["VERY_HIGH", "SEVERE"] else "Stay alert and monitor warnings",
                    "timestamp": row["timestamp"]
                })
        return alerts

    def get_state_risk_summary(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        if scores_df.empty:
            return pd.DataFrame()
        summary = []
        for state, group in scores_df.groupby("state"):
            high_risk_count = len(group[group["risk_level"].isin(["HIGH", "VERY_HIGH", "SEVERE"])])
            avg_rain = group["rainfall_24h"].mean() / 100.0 if "rainfall_24h" in group else 0
            avg_slope = group["slope"].mean() / 45.0 if "slope" in group else 0
            dominant_trigger = "Rainfall" if avg_rain > avg_slope else "Slope"
            summary.append({
                "state": state,
                "avg_risk": group["risk_probability"].mean(),
                "max_risk": group["risk_probability"].max(),
                "stations_at_high_risk": high_risk_count,
                "dominant_trigger": dominant_trigger
            })
        return pd.DataFrame(summary)
