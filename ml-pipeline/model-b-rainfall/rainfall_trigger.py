"""
rainfall_trigger.py
Live scoring: given a lat/lon, fetches the last 15 days of real rainfall
(Open-Meteo forecast API's past_days parameter, which includes actual
observed data, not just forecasts) and evaluates it against the trained
threshold model - both the classic I-D curve check and the logistic
trigger probability.

Usage as a script (for testing):
    python rainfall_trigger.py --model rainfall_threshold_model.pkl --lat 25.28 --lon 91.73

Usage as a module (for backend integration):
    from rainfall_trigger import RainfallTrigger
    trigger = RainfallTrigger("rainfall_threshold_model.pkl")
    result = trigger.check(lat=25.28, lon=91.73)
"""
import argparse
import numpy as np
import requests
import joblib

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class RainfallTrigger:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["features"]
        self.id_curve = bundle["id_curve"]

    def _fetch_recent_rainfall(self, lat, lon):
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 15,
            "forecast_days": 1,
            "timezone": "auto",
        }
        resp = requests.get(FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("daily", {}).get("precipitation_sum", [])
        return [v if v is not None else 0.0 for v in values]

    def _compute_features(self, daily_rain):
        cum_1d = daily_rain[-1]
        cum_3d = sum(daily_rain[-3:])
        cum_7d = sum(daily_rain[-7:])
        cum_15d = sum(daily_rain)

        duration = 0
        for v in reversed(daily_rain):
            if v > 1.0:
                duration += 1
            else:
                break
        duration = max(duration, 1)

        cum_over_duration = sum(daily_rain[-duration:])
        storm_intensity_mm_day = cum_over_duration / duration

        return {
            "cum_1d": cum_1d, "cum_3d": cum_3d, "cum_7d": cum_7d, "cum_15d": cum_15d,
            "duration_days": duration, "storm_intensity_mm_day": storm_intensity_mm_day,
        }

    def check(self, lat: float, lon: float) -> dict:
        daily_rain = self._fetch_recent_rainfall(lat, lon)
        features = self._compute_features(daily_rain)

        vector = np.array([[features[name] for name in self.feature_names]])
        trigger_probability = float(self.model.predict_proba(vector)[0, 1])

        # Classic I-D threshold check: is current intensity above the
        # curve for the current duration? This is the literal geoscience
        # threshold check, independent of the logistic model.
        threshold_intensity = self.id_curve["a"] * (features["duration_days"] ** -self.id_curve["b"])
        exceeds_id_threshold = features["storm_intensity_mm_day"] >= threshold_intensity

        return {
            "latitude": lat,
            "longitude": lon,
            "rainfall_24h_mm": features["cum_1d"],
            "rainfall_72h_mm": features["cum_3d"],
            "rainfall_7d_mm": features["cum_7d"],
            "duration_days": features["duration_days"],
            "trigger_probability": round(trigger_probability, 4),
            "exceeds_id_threshold": bool(exceeds_id_threshold),
            "id_threshold_curve_intensity_mm_day": round(threshold_intensity, 2),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    trigger = RainfallTrigger(args.model)
    result = trigger.check(args.lat, args.lon)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
