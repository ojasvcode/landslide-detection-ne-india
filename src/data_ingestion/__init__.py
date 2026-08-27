"""
Data Ingestion Module Initialization.
"""
from .weather_api import OpenMeteoClient
from .earthquake_api import USGSEarthquakeClient
from .dem_processor import DEMProcessor
from .landslide_inventory import LandslideInventoryLoader

__all__ = [
    "OpenMeteoClient",
    "USGSEarthquakeClient",
    "DEMProcessor",
    "LandslideInventoryLoader"
]
