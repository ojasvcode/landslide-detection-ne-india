"""
fetch_rainfall_history.py
For each real landslide event, fetches actual daily rainfall (Open-Meteo
Historical Weather API, free, no API key) for the 15 days ending on the
event date. For negative (non-triggering) examples, uses the same points
but dates far from any known event.

Usage:
    python fetch_rainfall_history.py \
        --events landslide_events.csv \
        --out rainfall_training_data.csv \
        --negatives_per_positive 2
"""
import argparse
import time
import random
import requests
import pandas as pd
from datetime import datetime, timedelta

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LOOKBACK_DAYS = 15


def fetch_daily_rainfall(lat, lon, end_date, lookback_days=LOOKBACK_DAYS):
    """Returns a list of daily precipitation values (mm) for the
    lookback_days ending on end_date (inclusive), oldest first."""
    start = end_date - timedelta(days=lookback_days - 1)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "precipitation_sum",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    values = data.get("daily", {}).get("precipitation_sum", [])
    # Open-Meteo returns null for missing days - treat as 0 rather than
    # dropping the whole event, since a single gap day shouldn't discard
    # an otherwise valid real event.
    return [v if v is not None else 0.0 for v in values]


def compute_features(daily_rain):
    """Turns a daily rainfall series into I-D threshold features."""
    if not daily_rain:
        return None
    cum_1d = daily_rain[-1]
    cum_3d = sum(daily_rain[-3:])
    cum_7d = sum(daily_rain[-7:])
    cum_15d = sum(daily_rain)

    # Duration = length of the current consecutive-rainy-day streak ending
    # on the target date - the "D" in the classic Intensity-Duration
    # threshold (Caine 1980 and successors).
    duration = 0
    for v in reversed(daily_rain):
        if v > 1.0:  # >1mm counts as a "rain day", filters out drizzle noise
            duration += 1
        else:
            break
    duration = max(duration, 1)  # avoid divide-by-zero in log(D) later

    # Storm intensity MUST be measured over the actual duration window, not
    # a fixed period - otherwise duration and intensity end up correlated
    # through total rainfall rather than showing the true inverse I-D
    # relationship (short bursts vs long gentle rain both trigger slides).
    cum_over_duration = sum(daily_rain[-duration:])
    storm_intensity_mm_day = cum_over_duration / duration

    return {
        "cum_1d": cum_1d,
        "cum_3d": cum_3d,
        "cum_7d": cum_7d,
        "cum_15d": cum_15d,
        "duration_days": duration,
        "storm_intensity_mm_day": storm_intensity_mm_day,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--negatives_per_positive", type=int, default=2)
    args = parser.parse_args()

    events = pd.read_csv(args.events, parse_dates=["event_date"])
    rng = random.Random(42)
    rows = []

    print(f"Fetching rainfall for {len(events)} positive events...")
    for i, row in events.iterrows():
        try:
            daily = fetch_daily_rainfall(row["latitude"], row["longitude"], row["event_date"])
            features = compute_features(daily)
            if features:
                features.update({"latitude": row["latitude"], "longitude": row["longitude"], "label": 1})
                rows.append(features)
        except Exception as e:
            print(f"  [{i+1}/{len(events)}] failed for real event: {e}")
        print(f"  [{i+1}/{len(events)}] positive events done", flush=True)
        time.sleep(0.3)  # be polite to the free public API

    print(f"Fetching rainfall for {args.negatives_per_positive}x negative (non-trigger) samples...")
    today = datetime.now()
    n_negatives = len(events) * args.negatives_per_positive
    for j in range(n_negatives):
        src = events.iloc[j % len(events)]
        # Pick a random date at least 45 days from this point's real event -
        # far enough that it's very unlikely to be part of the same trigger
        # window, giving us a genuine "nothing happened" rainfall sample.
        while True:
            days_offset = rng.randint(60, 3650)
            sign = rng.choice([-1, 1])
            candidate = src["event_date"] + timedelta(days=sign * days_offset)
            if abs((candidate - src["event_date"]).days) >= 45 and candidate < today:
                break
        try:
            daily = fetch_daily_rainfall(src["latitude"], src["longitude"], candidate)
            features = compute_features(daily)
            if features:
                features.update({"latitude": src["latitude"], "longitude": src["longitude"], "label": 0})
                rows.append(features)
        except Exception as e:
            print(f"  [{j+1}/{n_negatives}] failed for negative sample: {e}")
        print(f"  [{j+1}/{n_negatives}] negative samples done", flush=True)
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
