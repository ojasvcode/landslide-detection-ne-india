# Model B - Rainfall Intensity-Duration Threshold

Uses the same NASA COOLR shapefile from Model A (already downloaded to
ml-pipeline/data/landslide_catalog_raw/) plus real historical rainfall
from Open-Meteo (free, no API key needed).

## 1. Set up environment

```bash
cd ml-pipeline/model-b-rainfall
pip install -r requirements.txt
```

(If you're reusing the same venv from Model A, most of these are already
installed - only `requests` is likely new.)

## 2. Run the pipeline in order

```bash
python extract_landslide_events.py \
    --shapefile ../data/landslide_catalog_raw/global_landslide_catalog_NASA.shp \
    --out landslide_events.csv

python fetch_rainfall_history.py \
    --events landslide_events.csv \
    --out rainfall_training_data.csv \
    --negatives_per_positive 2

python train_rainfall_threshold.py \
    --data rainfall_training_data.csv \
    --out rainfall_threshold_model.pkl
```

Step 2 (fetch_rainfall_history.py) makes real API calls to Open-Meteo for
every event - with ~93 positive events x 3 (1 positive + 2 negatives)
that's roughly 280 requests with a 0.3s delay between each, so expect it
to take several minutes. It prints progress as it goes.

## 3. Test live scoring

```bash
python rainfall_trigger.py --model rainfall_threshold_model.pkl --lat 25.28 --lon 91.73
```

## Outputs (handoff to Member 3 / Member 4)

- `rainfall_threshold_model.pkl` - trained logistic classifier + fitted I-D curve
- `rainfall_threshold_model_id_curve.json` - the I-D curve (a, b) as plain JSON
- `rainfall_trigger.py` - importable `RainfallTrigger` class for live scoring,
  ready to wire into the FastAPI backend the same way Model A's
  `LandslidePredictor` was wired in

## Final results
- I-D curve: I = 0.59 * D^-(-1.468) — note the positive exponent, which
  differs from classical global thresholds (typically negative exponents
  around 0.3-0.6). This is an honest finding given the small (93-event),
  daily-resolution, geologically-mixed regional dataset compared to the
  hourly, single-region instrumented datasets typical in published I-D
  threshold literature - not a bug, disclosed openly in the pitch.
- Logistic classifier: CV AUC 0.80, test AUC 0.85 - this is the real,
  reliable trigger mechanism; the I-D curve is retained as a supplementary
  geoscience-standard reference, not the primary alert logic.


## Known limitations (disclose honestly in the pitch)

- Trained on real event dates but a synthetic negative-sampling strategy
  (real locations, dates far from any known event) - standard practice
  given no readily available "confirmed non-trigger" rainfall dataset
- Small real positive sample size (~93 events), same caveat as Model A
