"""
DEM Processor for extracting topographical features like elevation, slope, and aspect.
"""
import os
import math
import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import srtm
    SRTM_AVAILABLE = True
except ImportError:
    SRTM_AVAILABLE = False
    logger.warning("srtm.py is not installed. Using fallback average elevation values.")

class DEMProcessor:
    """Processor for Digital Elevation Model data."""

    # Fallback average elevations for NE Indian states (in meters)
    STATE_AVG_ELEVATION = {
        "Arunachal Pradesh": 3000.0,
        "Assam": 100.0,
        "Manipur": 1500.0,
        "Meghalaya": 1200.0,
        "Mizoram": 1000.0,
        "Nagaland": 1400.0,
        "Sikkim": 3500.0,
        "Tripura": 50.0,
        "Default": 1000.0
    }

    def __init__(self, cache_dir: str = 'data/raw/dem'):
        """
        Initialize the DEM Processor.

        Args:
            cache_dir (str): Directory to cache DEM data.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if SRTM_AVAILABLE:
            self.elevation_data = srtm.get_data(local_cache_dir=self.cache_dir)
        else:
            self.elevation_data = None

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get elevation at a specific point. Uses SRTM if available, else fallback.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            float: Elevation in meters.
        """
        if self.elevation_data:
            try:
                elev = self.elevation_data.get_elevation(lat, lon)
                if elev is not None:
                    return float(elev)
            except Exception as e:
                logger.error(f"Failed to get elevation from SRTM: {e}")
        
        # Fallback heuristic based on latitude/longitude roughly mapping to states
        if lat > 26.5 and lon > 91.5:
            return self.STATE_AVG_ELEVATION["Arunachal Pradesh"]
        elif 25.5 < lat < 28.0 and 89.5 < lon < 96.0:
            return self.STATE_AVG_ELEVATION["Assam"]
        elif 25.0 < lat < 26.0 and 89.5 < lon < 92.5:
            return self.STATE_AVG_ELEVATION["Meghalaya"]
        elif 27.0 < lat < 28.5 and 88.0 < lon < 89.0:
            return self.STATE_AVG_ELEVATION["Sikkim"]
        else:
            return self.STATE_AVG_ELEVATION["Default"]

    def get_elevation_grid(self, center_lat: float, center_lon: float, radius_deg: float = 0.05, step: float = 0.001) -> np.ndarray:
        """
        Get an elevation grid around a point for spatial feature computation.

        Args:
            center_lat (float): Center latitude.
            center_lon (float): Center longitude.
            radius_deg (float): Radius in degrees to fetch.
            step (float): Resolution step in degrees.

        Returns:
            np.ndarray: 2D numpy array of elevations.
        """
        lats = np.arange(center_lat - radius_deg, center_lat + radius_deg + step, step)
        lons = np.arange(center_lon - radius_deg, center_lon + radius_deg + step, step)
        
        grid = np.zeros((len(lats), len(lons)))
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                grid[i, j] = self.get_elevation(lat, lon)
                
        return grid

    def compute_slope(self, elevation_grid: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
        """
        Compute slope in degrees using numpy gradient (Horn's method equivalent).

        Args:
            elevation_grid (np.ndarray): 2D elevation grid.
            cell_size_m (float): Grid cell size in meters.

        Returns:
            np.ndarray: Slope grid in degrees.
        """
        dy, dx = np.gradient(elevation_grid, cell_size_m, cell_size_m)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return np.degrees(slope_rad)

    def compute_aspect(self, elevation_grid: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
        """
        Compute aspect in degrees (0=N, 90=E, 180=S, 270=W).

        Args:
            elevation_grid (np.ndarray): 2D elevation grid.
            cell_size_m (float): Grid cell size in meters.

        Returns:
            np.ndarray: Aspect grid in degrees.
        """
        dy, dx = np.gradient(elevation_grid, cell_size_m, cell_size_m)
        aspect_rad = np.arctan2(dy, -dx)
        aspect_deg = np.degrees(aspect_rad)
        return (aspect_deg + 360) % 360

    def compute_curvature(self, elevation_grid: np.ndarray, cell_size_m: float = 30.0) -> np.ndarray:
        """
        Compute profile curvature.

        Args:
            elevation_grid (np.ndarray): 2D elevation grid.
            cell_size_m (float): Grid cell size in meters.

        Returns:
            np.ndarray: Curvature grid.
        """
        dy, dx = np.gradient(elevation_grid, cell_size_m, cell_size_m)
        d2y, dy_dx = np.gradient(dy, cell_size_m, cell_size_m)
        dx_dy, d2x = np.gradient(dx, cell_size_m, cell_size_m)
        
        # Simple laplacian as proxy for curvature
        curvature = d2x + d2y
        return curvature

    def compute_twi(self, slope_grid: np.ndarray, contributing_area: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Topographic Wetness Index (TWI).

        Args:
            slope_grid (np.ndarray): 2D slope grid in degrees.
            contributing_area (Optional[np.ndarray]): Upstream area. Defaults to a constant if not provided.

        Returns:
            np.ndarray: TWI grid.
        """
        if contributing_area is None:
            # Fallback uniform area if routing is not computed
            contributing_area = np.ones_like(slope_grid) * 900.0  # e.g., 30x30m
            
        slope_rad = np.radians(slope_grid)
        # Avoid division by zero
        slope_rad = np.maximum(slope_rad, 0.001)
        twi = np.log(contributing_area / np.tan(slope_rad))
        return twi

    def get_terrain_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get all terrain features for a single point.

        Args:
            lat (float): Latitude.
            lon (float): Longitude.

        Returns:
            Dict[str, Any]: Terrain features including elevation, slope, etc.
        """
        radius = 0.002
        step = 0.001
        
        grid = self.get_elevation_grid(lat, lon, radius_deg=radius, step=step)
        
        # The center of the grid represents our point
        center_idx = grid.shape[0] // 2
        
        slope_grid = self.compute_slope(grid)
        aspect_grid = self.compute_aspect(grid)
        curv_grid = self.compute_curvature(grid)
        twi_grid = self.compute_twi(slope_grid)
        
        elev = grid[center_idx, center_idx]
        slope = slope_grid[center_idx, center_idx]
        aspect = aspect_grid[center_idx, center_idx]
        curvature = curv_grid[center_idx, center_idx]
        twi = twi_grid[center_idx, center_idx]
        
        relative_relief = np.max(grid) - np.min(grid)
        
        return {
            "elevation": elev,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "twi": twi,
            "relative_relief": relative_relief
        }

    def get_batch_terrain_features(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute terrain features for multiple locations.

        Args:
            locations_df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.

        Returns:
            pd.DataFrame: Original DataFrame merged with computed terrain features.
        """
        features_list = []
        for _, row in locations_df.iterrows():
            feats = self.get_terrain_features(row['latitude'], row['longitude'])
            features_list.append(feats)
            
        feats_df = pd.DataFrame(features_list)
        return pd.concat([locations_df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
