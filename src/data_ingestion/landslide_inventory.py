"""
Landslide Inventory Loader for handling known landslide event datasets.
"""
import os
import random
import math
import logging
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class LandslideInventoryLoader:
    """Loader for historical landslide inventory data."""

    # NE India Bounding Box
    BBOX = {
        "min_lat": 21.0,
        "max_lat": 29.5,
        "min_lon": 87.0,
        "max_lon": 98.0
    }

    def __init__(self, data_dir: str = 'data/sample'):
        """
        Initialize the Landslide Inventory Loader.

        Args:
            data_dir (str): Directory containing inventory datasets.
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_nasa_catalog(self, filepath: str = None) -> pd.DataFrame:
        """
        Load the NASA Global Landslide Catalog and filter it to NE India.

        Args:
            filepath (str, optional): Path to GLC CSV file.

        Returns:
            pd.DataFrame: Filtered and standardized landslide catalog.
        """
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'nasa_glc.csv')
            
        if not os.path.exists(filepath):
            logger.warning(f"NASA GLC file not found at {filepath}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["latitude", "longitude", "date", "state", "trigger"])
            
        try:
            df = pd.read_csv(filepath)
            
            # Filter to bbox
            mask = (
                (df['latitude'] >= self.BBOX['min_lat']) & 
                (df['latitude'] <= self.BBOX['max_lat']) & 
                (df['longitude'] >= self.BBOX['min_lon']) & 
                (df['longitude'] <= self.BBOX['max_lon'])
            )
            df = df[mask].copy()
            
            # Standardize columns
            if 'event_date' in df.columns:
                df['date'] = pd.to_datetime(df['event_date'])
            else:
                df['date'] = pd.to_datetime('today')
                
            df['state'] = df.get('adminname1', 'Unknown')
            df['trigger'] = df.get('landslide_trigger', 'Unknown')
            df['label'] = 1  # Positive sample
            
            return df[['latitude', 'longitude', 'date', 'state', 'trigger', 'label']]
            
        except Exception as e:
            logger.error(f"Error loading NASA catalog: {e}")
            return pd.DataFrame(columns=["latitude", "longitude", "date", "state", "trigger", "label"])

    def load_sample_catalog(self) -> pd.DataFrame:
        """
        Load a sample catalog from the data directory.

        Returns:
            pd.DataFrame: Landslide catalog.
        """
        filepath = os.path.join(self.data_dir, 'ne_india_landslide_catalog.csv')
        if not os.path.exists(filepath):
            # Create a dummy dataframe for demonstration if file doesn't exist
            logger.info("Sample catalog not found, generating dummy data.")
            dummy_data = {
                "latitude": [25.57, 27.33, 26.14],
                "longitude": [91.89, 88.61, 91.73],
                "date": pd.to_datetime(["2021-06-15", "2022-07-20", "2023-08-10"]),
                "state": ["Meghalaya", "Sikkim", "Assam"],
                "trigger": ["Downpour", "Continuous rain", "Downpour"],
                "label": [1, 1, 1]
            }
            return pd.DataFrame(dummy_data)
        
        return pd.read_csv(filepath)

    def generate_negative_samples(self, landslide_df: pd.DataFrame, n_multiplier: int = 2, min_distance_km: float = 5.0) -> pd.DataFrame:
        """
        Generate non-landslide (negative) sample points within the NE India bbox.

        Args:
            landslide_df (pd.DataFrame): DataFrame of positive samples.
            n_multiplier (int): Ratio of negative to positive samples.
            min_distance_km (float): Minimum distance from any positive sample.

        Returns:
            pd.DataFrame: Combined DataFrame of positive and negative samples.
        """
        n_positives = len(landslide_df)
        if n_positives == 0:
            return landslide_df
            
        n_negatives = n_positives * n_multiplier
        negatives = []
        
        positive_coords = landslide_df[['latitude', 'longitude']].values.tolist()
        
        attempts = 0
        max_attempts = n_negatives * 10
        
        while len(negatives) < n_negatives and attempts < max_attempts:
            attempts += 1
            rand_lat = random.uniform(self.BBOX['min_lat'], self.BBOX['max_lat'])
            rand_lon = random.uniform(self.BBOX['min_lon'], self.BBOX['max_lon'])
            
            # Check distance against all positive samples
            too_close = False
            for p_lat, p_lon in positive_coords:
                if haversine_distance(rand_lat, rand_lon, p_lat, p_lon) < min_distance_km:
                    too_close = True
                    break
                    
            if not too_close:
                negatives.append({
                    "latitude": rand_lat,
                    "longitude": rand_lon,
                    "date": pd.to_datetime('today'), # Usually use a random or matching date
                    "state": "Unknown",
                    "trigger": "None",
                    "label": 0
                })
                
        neg_df = pd.DataFrame(negatives)
        return pd.concat([landslide_df, neg_df], ignore_index=True)

    def prepare_labeled_dataset(self, catalog_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Full pipeline: load catalog, generate negative samples, and return a labeled dataset.

        Args:
            catalog_df (pd.DataFrame, optional): Pre-loaded catalog. Loads sample if None.

        Returns:
            pd.DataFrame: Complete labeled dataset.
        """
        if catalog_df is None:
            catalog_df = self.load_sample_catalog()
            
        if 'label' not in catalog_df.columns:
            catalog_df['label'] = 1
            
        return self.generate_negative_samples(catalog_df, n_multiplier=2, min_distance_km=5.0)

    def get_landslide_statistics(self, catalog_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for a landslide inventory.

        Args:
            catalog_df (pd.DataFrame): The inventory DataFrame.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        if catalog_df.empty:
            return {}
            
        stats = {
            "total_events": len(catalog_df),
            "by_state": catalog_df['state'].value_counts().to_dict() if 'state' in catalog_df else {},
            "by_trigger": catalog_df['trigger'].value_counts().to_dict() if 'trigger' in catalog_df else {},
        }
        
        if 'date' in catalog_df.columns:
            years = pd.to_datetime(catalog_df['date']).dt.year
            stats["by_year"] = years.value_counts().to_dict()
            
        if 'fatalities' in catalog_df.columns:
            stats["total_fatalities"] = catalog_df['fatalities'].sum()
            
        return stats

    def filter_by_state(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """Filter inventory by state name."""
        if 'state' not in df.columns:
            return df
        return df[df['state'].str.lower() == state.lower()]

    def filter_by_daterange(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter inventory by date range."""
        if 'date' not in df.columns:
            return df
        
        mask = (pd.to_datetime(df['date']) >= pd.to_datetime(start_date)) & \
               (pd.to_datetime(df['date']) <= pd.to_datetime(end_date))
        return df[mask]
