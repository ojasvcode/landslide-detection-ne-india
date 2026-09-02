"""
download_dem_states.py
Downloads a DEM per North East India state via OpenTopography's REST API -
scripted, so no manual portal clicking needed for each of the 8 states.

Approximate state bounding boxes (rectangles, so each includes a bit of
neighboring territory - harmless here since we're modeling terrain, not
enforcing administrative boundaries).

Usage:
    python download_dem_states.py --api_key YOUR_KEY_HERE --outdir ../data/ner_states
"""
import argparse
import os
import time
import requests

API_URL = "https://portal.opentopography.org/API/globaldem"

# Approximate rectangular bounding boxes per state (south, north, west, east).
# Meghalaya is intentionally the already-covered bbox from earlier work, kept
# here for consistency in case you want to regenerate it identically.
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


def is_valid_geotiff(path):
    """Actually reads the ENTIRE band, not just a small corner sample - the
    corrupted tiles we hit were deep inside the file (e.g. row 5120), so a
    small origin-only sample missed them completely. Reading the full band
    touches every tile and will surface a read error anywhere in the file."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            src.read(1)
        return True
    except Exception:
        return False


def download_state_dem(state, bbox, api_key, outdir, max_retries=2):
    out_path = os.path.join(outdir, f"dem_{state}.tif")
    if os.path.exists(out_path):
        if is_valid_geotiff(out_path):
            print(f"  {state}: already downloaded and valid, skipping")
            return out_path
        else:
            print(f"  {state}: existing file is corrupted, re-downloading")
            os.remove(out_path)

    params = {
        "demtype": "SRTMGL1",
        "south": bbox["south"],
        "north": bbox["north"],
        "west": bbox["west"],
        "east": bbox["east"],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    for attempt in range(1, max_retries + 1):
        print(f"  {state}: requesting {bbox} (attempt {attempt}/{max_retries})...")
        try:
            resp = requests.get(API_URL, params=params, timeout=180)
        except requests.exceptions.RequestException as e:
            print(f"  {state}: attempt {attempt} FAILED (network error) - {e}")
            continue

        if resp.status_code != 200 or resp.headers.get("Content-Type", "").startswith("application/json"):
            print(f"  {state}: attempt {attempt} FAILED - {resp.status_code} - {resp.text[:300]}")
            continue

        with open(out_path, "wb") as f:
            f.write(resp.content)

        if is_valid_geotiff(out_path):
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  {state}: saved and verified {size_mb:.1f} MB to {out_path}")
            return out_path
        else:
            print(f"  {state}: attempt {attempt} downloaded but file is corrupted, retrying")
            os.remove(out_path)

    print(f"  {state}: FAILED after {max_retries} attempts")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--states", nargs="*", default=list(STATE_BBOXES.keys()),
                         help="Subset of states to download (default: all 8)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Downloading DEMs for {len(args.states)} state(s)...")
    results = {}
    for state in args.states:
        if state not in STATE_BBOXES:
            print(f"  {state}: unknown state, skipping (valid: {list(STATE_BBOXES.keys())})")
            continue
        try:
            path = download_state_dem(state, STATE_BBOXES[state], args.api_key, args.outdir)
        except Exception as e:
            print(f"  {state}: FAILED (unexpected error) - {e}")
            path = None
        results[state] = path
        time.sleep(2)  # be polite to the API between requests

    print("\nSummary:")
    for state, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {state}: {status}")


if __name__ == "__main__":
    main()
