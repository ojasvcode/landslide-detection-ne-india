"""
extract_landslide_events.py
Re-filters the NASA COOLR shapefile (already downloaded for Model A) to the
same bbox, but this time keeps event_date - needed to look up what rainfall
actually preceded each real landslide.

Usage:
    python extract_landslide_events.py \
        --shapefile ../data/landslide_catalog_raw/global_landslide_catalog_NASA.shp \
        --out landslide_events.csv
"""
import argparse
import geopandas as gpd
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_lon", type=float, default=89.7)
    parser.add_argument("--max_lon", type=float, default=92.8)
    parser.add_argument("--min_lat", type=float, default=25.0)
    parser.add_argument("--max_lat", type=float, default=26.3)
    args = parser.parse_args()

    gdf = gpd.read_file(args.shapefile)

    filtered = gdf[
        (gdf["longitude"] >= args.min_lon) & (gdf["longitude"] <= args.max_lon) &
        (gdf["latitude"] >= args.min_lat) & (gdf["latitude"] <= args.max_lat)
    ].copy()

    # event_date comes in as a mixed/inconsistent string format from NASA's
    # export - coerce to real dates and drop anything that fails to parse,
    # since an event we can't date can't be matched to rainfall history.
    filtered["event_date"] = pd.to_datetime(filtered["event_date"], errors="coerce")
    before = len(filtered)
    filtered = filtered.dropna(subset=["event_date"])
    print(f"Points in bbox: {before}, with valid event_date: {len(filtered)}")

    out_df = filtered[["latitude", "longitude", "event_date"]].copy()
    out_df["event_date"] = out_df["event_date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
