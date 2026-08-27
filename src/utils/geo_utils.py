"""
Geospatial utilities for Landslide Detection System.
"""
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def generate_grid(bbox: dict, resolution: float) -> List[Tuple[float, float]]:
    """
    Create a grid of latitude and longitude points within a bounding box.
    """
    lats = np.arange(bbox['min_lat'], bbox['max_lat'] + resolution, resolution)
    lons = np.arange(bbox['min_lon'], bbox['max_lon'] + resolution, resolution)
    
    grid = []
    for lat in lats:
        for lon in lons:
            grid.append((float(lat), float(lon)))
            
    return grid

def point_in_ne_india(lat: float, lon: float) -> bool:
    """
    Check if a given latitude and longitude point is within the NE India bounding box.
    """
    min_lat, max_lat = 21.0, 29.5
    min_lon, max_lon = 87.0, 98.0
    
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

def get_state_for_point(lat: float, lon: float) -> Optional[str]:
    """
    Roughly assign a state to a point based on simple bounding boxes.
    Note: This is a rough approximation.
    """
    state_bboxes = {
        'Sikkim': (27.0, 28.1, 88.0, 89.0),
        'Arunachal Pradesh': (26.5, 29.5, 91.5, 97.5),
        'Assam': (24.0, 28.0, 89.5, 96.0),
        'Meghalaya': (25.0, 26.1, 89.7, 92.8),
        'Nagaland': (25.2, 27.0, 93.3, 95.3),
        'Manipur': (23.8, 25.7, 93.0, 94.8),
        'Mizoram': (21.9, 24.5, 92.2, 93.4),
        'Tripura': (22.9, 24.5, 91.1, 92.3)
    }
    
    for state, bbox in state_bboxes.items():
        if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
            return state
            
    return None

def create_grid_dataframe(bbox: dict, resolution: float) -> pd.DataFrame:
    """
    Create a DataFrame containing a grid of points with grid_id.
    """
    grid = generate_grid(bbox, resolution)
    
    df = pd.DataFrame(grid, columns=['lat', 'lon'])
    df['grid_id'] = [f"GRID_{i:05d}" for i in range(len(df))]
    
    return df

def nearest_point(target_lat: float, target_lon: float, points_df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Find the nearest point in points_df to the target coordinates.
    points_df must have 'lat' and 'lon' columns.
    Returns (nearest_lat, nearest_lon, distance_in_km).
    """
    if points_df.empty:
        raise ValueError("points_df cannot be empty")
        
    distances = points_df.apply(
        lambda row: haversine_distance(target_lat, target_lon, row['lat'], row['lon']),
        axis=1
    )
    
    min_idx = distances.idxmin()
    nearest_row = points_df.loc[min_idx]
    
    return nearest_row['lat'], nearest_row['lon'], distances[min_idx]
