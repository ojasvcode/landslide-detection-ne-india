"""
build_ner_landslide_events.py
Extracts real landslide events (with dates) across all 8 North East India
states from the same NASA COOLR shapefile, into one combined CSV ready for
fetch_rainfall_history.py - which is already location-agnostic, so no
changes needed there or in train_rainfall_threshold.py.

Usage:
    python build_ner_landslide_events.py \
        --shapefile ../data/landslide_catalog_raw/global_landslide_catalog_NASA.shp \
        --out ../data/ner_landslide_events.csv
"""
import argparse
import geopandas as gpd
import pandas as pd

# Same bboxes as download_dem_states.py / build_ner_training_data.py -
# kept in sync manually since these are small, stable reference values.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gdf = gpd.read_file(args.shapefile)
    gdf["event_date"] = pd.to_datetime(gdf["event_date"], errors="coerce")

    all_events = []
    print(f"Processing {len(STATE_BBOXES)} states...")
    for state, bbox in STATE_BBOXES.items():
        filtered = gdf[
            (gdf["longitude"] >= bbox["west"]) & (gdf["longitude"] <= bbox["east"]) &
            (gdf["latitude"] >= bbox["south"]) & (gdf["latitude"] <= bbox["north"])
        ].dropna(subset=["event_date"])
        print(f"  {state}: {len(filtered)} events with valid dates")
        all_events.append(filtered[["latitude", "longitude", "event_date"]])

    combined = pd.concat(all_events, ignore_index=True)
    # A landslide near a state border could technically fall inside more
    # than one rectangular bbox - drop exact duplicate lat/lon/date rows so
    # it isn't double-counted in training.
    before = len(combined)
    combined = combined.drop_duplicates(subset=["latitude", "longitude", "event_date"])
    print(f"Dropped {before - len(combined)} duplicate events from bbox overlap")

    combined["event_date"] = combined["event_date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(args.out, index=False)
    print(f"\nTotal NER-wide events: {len(combined)}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
