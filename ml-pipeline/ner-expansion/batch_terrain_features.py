"""
batch_terrain_features.py
Runs terrain_features.py as a SEPARATE PROCESS per state DEM, rather than
importing and calling its functions repeatedly inside one long-running
process. Each state's large GDAL reprojection gets a clean memory slate,
matching the exact invocation pattern already proven to work standalone -
avoids GDAL's C-level memory not being fully released between large
operations when run back-to-back in a single Python process.

Usage:
    python batch_terrain_features.py \
        --dem_dir ../data/ner_states \
        --outdir ../data/ner_states_features
"""
import argparse
import glob
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    script_path = os.path.join(os.path.dirname(__file__), "..", "model-a-lsm", "terrain_features.py")
    script_path = os.path.abspath(script_path)

    dem_files = sorted(glob.glob(os.path.join(args.dem_dir, "dem_*.tif")))
    print(f"Found {len(dem_files)} state DEM(s)")

    results = {}
    for dem_path in dem_files:
        state = os.path.basename(dem_path).replace("dem_", "").replace(".tif", "")
        state_outdir = os.path.join(args.outdir, state)
        os.makedirs(state_outdir, exist_ok=True)

        print(f"\nProcessing {state} (subprocess)...")
        result = subprocess.run(
            [sys.executable, script_path, "--dem", os.path.abspath(dem_path), "--outdir", os.path.abspath(state_outdir)],
            capture_output=True, text=True,
        )

        if result.returncode == 0:
            # Print the child process's own output (includes the slope range line)
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
            results[state] = "OK"
        else:
            print(f"  {state}: FAILED")
            print(f"  --- stderr ---")
            print("  " + result.stderr.strip().replace("\n", "\n  "))
            results[state] = "FAILED"

    print("\nSummary:")
    for state, status in results.items():
        print(f"  {state}: {status}")


if __name__ == "__main__":
    main()
