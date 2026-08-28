from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.risk_engine.scoring import RiskScoringEngine

app = FastAPI(
    title="Landslide Detection API",
    description="Real-time risk scoring API for North Eastern India",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RiskScoringEngine()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Landslide Detection API is running"}

@app.get("/api/risk/all")
def get_all_risk():
    """Get real-time landslide risk for all 48 monitoring stations."""
    try:
        scores_df = engine.score_all_stations()
        if scores_df.empty:
            return {"status": "success", "data": []}
            
        scores_df = scores_df.sort_values(by="risk_probability", ascending=False)
        return {"status": "success", "data": scores_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk/state/{state_name}")
def get_state_risk(state_name: str):
    """Get real-time landslide risk for a specific state (e.g., 'Assam')."""
    try:
        scores_df = engine.score_state(state_name)
        if scores_df.empty:
            return {"status": "success", "data": [], "message": f"No data or stations found for {state_name}"}
            
        scores_df = scores_df.sort_values(by="risk_probability", ascending=False)
        return {"status": "success", "state": state_name, "data": scores_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/geojson")
def get_risk_geojson():
    """Get real-time landslide risk formatted as GeoJSON for direct map integration."""
    try:
        scores_df = engine.score_all_stations()
        features = []
        for _, row in scores_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["lon"], row["lat"]]
                },
                "properties": {
                    "name": row["name"],
                    "state": row["state"],
                    "risk_level": row["risk_level"],
                    "risk_probability": round(row["risk_probability"], 3),
                    "rainfall_24h": row.get("rainfall_24h", 0.0),
                    "slope": row.get("slope", 0.0)
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        return geojson
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
