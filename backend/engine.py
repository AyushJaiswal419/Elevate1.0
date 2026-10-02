from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    'rain_1h', 'rain_6h', 'rain_24h', 'rain_72h', 'forecast_rain_24h',
    'river_discharge_pct', 'river_rise_6h_pct', 'soil_moisture_pct',
    'elevation_m', 'slope_pct', 'river_distance_km', 'drainage_density',
    'population_density', 'vulnerable_population_pct'
]


def _make_training_data(n=9000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.gamma(1.6, 3.0, n),
        rng.gamma(1.7, 7.0, n),
        rng.gamma(1.8, 15.0, n),
        rng.gamma(1.7, 28.0, n),
        rng.gamma(1.8, 18.0, n),
        rng.beta(2.5, 1.6, n) * 100,
        rng.normal(0, 15, n).clip(-40, 80),
        rng.beta(3.0, 2.0, n) * 100,
        rng.normal(1580, 450, n).clip(900, 3200),
        rng.gamma(1.8, 2.0, n).clip(0, 25),
        rng.gamma(2.0, 1.4, n).clip(.05, 12),
        rng.gamma(2.0, .9, n).clip(.1, 8),
        rng.lognormal(7.5, .8, n).clip(100, 25000),
        rng.beta(2.0, 6.0, n) * 100,
    ])
    # Synthetic labels: a monotonic hydrometeorological risk surface, not real event labels.
    z = (
        0.018*x[:,1] + 0.010*x[:,2] + 0.005*x[:,3] + 0.014*x[:,4]
        + 0.018*x[:,5] + 0.020*np.maximum(x[:,6], 0) + 0.012*x[:,7]
        - 0.0007*(x[:,8]-1500) - 0.025*x[:,9] - 0.11*x[:,10]
        + 0.07*x[:,11] + 0.000015*x[:,12] + 0.010*x[:,13] - 5.2
    )
    p = 1/(1+np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    return x, y


def train_model() -> RandomForestClassifier:
    x, y = _make_training_data()
    model = RandomForestClassifier(
        n_estimators=220, max_depth=11, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=42, n_jobs=-1
    )
    model.fit(x, y)
    return model

MODEL = train_model()


def predict_flood(features: Dict[str, float]) -> Dict[str, float]:
    row = np.array([[float(features[k]) for k in FEATURES]])
    p = float(MODEL.predict_proba(row)[0, 1])
    return {'probability': round(p*100, 1), 'horizon_hours': 24}


def severity(probability: float, vulnerability: float, sos: int = 0) -> Dict[str, float | str]:
    score = probability * vulnerability + min(10, sos * 2)
    if score >= 60: level = 'CRITICAL'
    elif score >= 40: level = 'HIGH'
    elif score >= 20: level = 'MODERATE'
    else: level = 'LOW'
    return {'score': round(score,1), 'level': level}


def haversine_km(a, b):
    lat1, lon1 = a; lat2, lon2 = b
    r=6371.0
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(h))


def cyclone_risk(center, storm):
    d = haversine_km(center, (storm['lat'], storm['lon']))
    wind_factor = min(1, max(0, (storm['wind_kmh']-50)/140))
    pressure_factor = min(1, max(0, (1015-storm['pressure_hpa'])/80))
    proximity = math.exp(-d/180)
    p = 100*(0.55*proximity*wind_factor + 0.45*proximity*pressure_factor)
    return {'probability': round(min(99.9,p),1), 'distance_km': round(d,1)}


def fingerprint(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + '|' + payload).encode()).hexdigest()
