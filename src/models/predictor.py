"""
src/models/predictor.py (multi-state version)

Extends the original single-region LandslidePredictor to handle terrain
rasters split across multiple states (from the NER-wide expansion) - each
state has its own raster set with its own correct UTM zone, so this picks
the right state's rasters for a given point based on which one's bounds
actually contain it, rather than assuming one fixed region.

Expects a model_dir containing:
    model_dir/ner_lsm_model.pkl  (or lsm_model.pkl for the single-region model)
    model_dir/features/<state>/elevation.tif, slope.tif, aspect.tif, curvature.tif
    (one subfolder per state - matches ner_states_features/ layout)

Falls back to the original single-folder layout (model_dir/features/*.tif
directly, no state subfolders) automatically if that's what's present -
so this works for BOTH the original Meghalaya+Assam model and the new
NER-wide model without needing two separate predictor files.
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


class LandslidePredictor:
    def __init__(self, model_dir: str):
        # Support either filename - the NER-wide model or the original.
        model_path = os.path.join(model_dir, "ner_lsm_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "lsm_model.pkl")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["features"]

        features_root = os.path.join(model_dir, "features")
        state_dirs = [d for d in glob.glob(os.path.join(features_root, "*")) if os.path.isdir(d)]

        # self.regions is a list of {"transformer", "rasters"} - one entry
        # per state (or a single entry if using the original flat layout).
        self.regions = []
        if state_dirs:
            for state_dir in state_dirs:
                self._load_region(state_dir)
        else:
            # Original flat layout: features_root itself has the .tif files.
            self._load_region(features_root)

        if not self.regions:
            raise RuntimeError(f"No terrain feature rasters found under {features_root}")

    def _load_region(self, region_dir):
        rasters = {}
        for name in self.feature_names:
            path = os.path.join(region_dir, f"{name}.tif")
            if not os.path.exists(path):
                return  # incomplete region, skip it rather than crash later
            rasters[name] = rasterio.open(path)

        raster_crs = next(iter(rasters.values())).crs
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        bounds = next(iter(rasters.values())).bounds

        self.regions.append({
            "name": os.path.basename(region_dir),
            "rasters": rasters,
            "transformer": transformer,
            "bounds": bounds,
        })

    def _sample_point(self, lat: float, lon: float):
        # Try each region until one actually contains this point - a point
        # in Sikkim should be sampled from Sikkim's rasters, not Assam's.
        for region in self.regions:
            x, y = region["transformer"].transform(lon, lat)
            b = region["bounds"]
            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                continue

            values = {}
            valid = True
            for name, src in region["rasters"].items():
                row, col = src.index(x, y)
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    valid = False
                    break
                value = list(src.sample([(x, y)]))[0][0]
                if np.isnan(value):
                    valid = False
                    break
                values[name] = value
            if valid:
                return values
        return None  # point outside every region's coverage

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        probabilities, levels = [], []
        for _, row in feature_df.iterrows():
            terrain = self._sample_point(row["lat"], row["lon"])
            if terrain is None:
                probabilities.append(0.0)
                levels.append("LOW")
                continue
            vector = np.array([[terrain[name] for name in self.feature_names]])
            probability = float(self.model.predict_proba(vector)[0, 1])
            probabilities.append(probability)
            levels.append(self._assign_level(probability))
        return pd.DataFrame({"probability": probabilities, "risk_level": levels})

    @staticmethod
    def _assign_level(probability: float) -> str:
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
