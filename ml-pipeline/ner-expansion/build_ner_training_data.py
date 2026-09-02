"""
build_ner_training_data.py
For each state, filters the same global NASA COOLR shapefile (already
downloaded) to that state's bbox, samples terrain features at real
landslide points + synthetic negatives, and combines everything into one
NER-wide training CSV.

Reuses the exact sampling logic from build_training_dataset.py - one
source of truth for how points get labeled and sampled.

Usage:
    python build_ner_training_data.py \
        --shapefile ../data/landslide_catalog_raw/global_landslide_catalog_NASA.shp \
        --features_dir ../data/ner_states_features \
        --out ../data/ner_training_data.csv \
        --n_negatives_per_state 300
"""
import argparse
import os
import sys
import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model-a-lsm"))
from build_training_dataset import load_inventory_as_points, generate_negative_points, sample_raster_at_points

# Same bboxes as download_dem_states.py - kept in sync manually since these
# are small, stable reference values.
STATE_BBOXES = {
    "arunachal_pradesh": {"south": 26.6, "north": 29.5, "west": 91.6, "east": 97.5},
    "assam":              {"south": 24.1, "north": 28.2, "west": 89.7, "east": 96.1},
    "manipur":            {"south": 23.8, "north": 25.7, "west": 93.0, "east": 94.8},
    "meghalaya":          {"south": 25.0, "north": 26.3, "west": 89.7, "east": 92.8},
    "mizoram":            {"south": 21.9, "north": 24.5, "west": 92.2, "east": 93.5},
    "nagaland":           {"south": 25.2, "north": 27.0, "west": 93.3, "east": 95.8},
    "sikkim":             {"south": 27.0, "north": 28.2, "west": 88.0, "east": 88.9},
    "tripura":            {"south": 22.9, "north": 24.5, "west": 91.1, "east": 92.3},
}

FEATURE_FILES = {
    "elevation": "elevation.tif",
    "slope": "slope.tif",
    "aspect": "aspect.tif",
    "curvature": "curvature.tif",
}


def process_state(state, bbox, global_gdf, features_dir, n_negatives):
    state_features_dir = os.path.join(features_dir, state)
    ref_raster = os.path.join(state_features_dir, "elevation.tif")
    if not os.path.exists(ref_raster):
        print(f"  {state}: no features found, skipping (run batch_terrain_features.py first)")
        return None

    import rasterio
    with rasterio.open(ref_raster) as ref:
        dst_crs = ref.crs
        bounds = ref.bounds

    filtered = global_gdf[
        (global_gdf["longitude"] >= bbox["west"]) & (global_gdf["longitude"] <= bbox["east"]) &
        (global_gdf["latitude"] >= bbox["south"]) & (global_gdf["latitude"] <= bbox["north"])
    ]
    if len(filtered) == 0:
        print(f"  {state}: 0 positive points in bbox, skipping")
        return None

    positive_gdf = gpd.GeoDataFrame(
        filtered, geometry=gpd.points_from_xy(filtered.longitude, filtered.latitude), crs="EPSG:4326"
    ).to_crs(dst_crs)

    negative_points = generate_negative_points(bounds, positive_gdf, n_negatives)

    all_points = list(positive_gdf.geometry) + negative_points
    labels = [1] * len(positive_gdf) + [0] * len(negative_points)

    rows = {"label": labels, "state": [state] * len(all_points)}
    for feature_name, filename in FEATURE_FILES.items():
        raster_path = os.path.join(state_features_dir, filename)
        rows[feature_name] = sample_raster_at_points(raster_path, all_points)

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.dropna()
    print(f"  {state}: {len(positive_gdf)} positive, {len(negative_points)} negative "
          f"({before - len(df)} dropped outside coverage)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_negatives_per_state", type=int, default=300)
    args = parser.parse_args()

    global_gdf = gpd.read_file(args.shapefile)

    all_dfs = []
    print(f"Processing {len(STATE_BBOXES)} states...")
    for state, bbox in STATE_BBOXES.items():
        df = process_state(state, bbox, global_gdf, args.features_dir, args.n_negatives_per_state)
        if df is not None:
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"\nCombined NER-wide training data: {len(combined)} rows")
    print(combined["label"].value_counts())
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
