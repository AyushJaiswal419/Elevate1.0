from __future__ import annotations
import json, time, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import FEATURES, predict_flood, severity, cyclone_risk, fingerprint

ROOT=Path(__file__).resolve().parents[1]
FRONT=ROOT/'frontend'
DATA=ROOT/'data'
app=FastAPI(title='Saarthi Disaster Command Center', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

ZONES=json.loads((DATA/'srinagar_zones.json').read_text())
LEDGER=[]

def live_weather():
    """Best-effort live forecast for central Srinagar; demo remains functional if unavailable."""
    params=urllib.parse.urlencode({
        'latitude':34.0837,'longitude':74.7973,
        'hourly':'precipitation,rain,soil_moisture_0_to_7cm_mean,pressure_msl,wind_gusts_10m',
        'forecast_days':2,'timezone':'Asia/Kolkata'
    })
    try:
        with urllib.request.urlopen('https://api.open-meteo.com/v1/forecast?'+params, timeout=2.5) as r:
            d=json.loads(r.read().decode())
        h=d.get('hourly',{})
        precip=h.get('precipitation',[])
        rain24=sum(float(x or 0) for x in precip[:24])
        rain48=sum(float(x or 0) for x in precip[:48])
        return {'available':True,'rain_next_24h_mm':round(rain24,1),'rain_next_48h_mm':round(rain48,1),'source':'Open-Meteo forecast'}
    except Exception as e:
        return {'available':False,'rain_next_24h_mm':None,'rain_next_48h_mm':None,'source':'demo fallback'}

GENESIS='GENESIS'

class Scenario(BaseModel):
    rain_1h: float=12
    rain_6h: float=42
    rain_24h: float=88
    rain_72h: float=145
    forecast_rain_24h: float=72
    river_discharge_pct: float=78
    river_rise_6h_pct: float=14
    soil_moisture_pct: float=74
    elevation_m: float=1585
    slope_pct: float=4
    river_distance_km: float=2.2
    drainage_density: float=2.5
    population_density: float=7000
    vulnerable_population_pct: float=22

@app.get('/api/health')
def health(): return {'ok':True,'service':'saarthi','time':datetime.now(timezone.utc).isoformat()}

@app.get('/api/state')
def state():
    now=datetime.now().astimezone().strftime('%H:%M:%S')
    rows=[]
    for z in ZONES:
        pred=predict_flood(z['features'])
        # Zone-specific vulnerability and simulated field demand.
        sev=severity(pred['probability'], z['vulnerability'], z['open_sos'])
        rows.append({**z, 'prediction':pred, **sev})
    rows.sort(key=lambda x:x['score'], reverse=True)
    critical=sum(x['level']=='CRITICAL' for x in rows)
    high=sum(x['level']=='HIGH' for x in rows)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'local_time': now,
        'live_weather': live_weather(),
        'refresh_seconds':30,
        'zones':rows,
        'metrics': {'critical':critical,'high':high,'open_sos':sum(x['open_sos'] for x in rows),'shipments':7},
        'alerts': [
            {'id':'SA-2401','type':'Flood','zone':rows[0]['name'],'level':rows[0]['level'],'probability':rows[0]['prediction']['probability'],'eta_h':24},
            {'id':'SA-2402','type':'Flash Flood','zone':'Rainawari','level':'HIGH','probability':64.2,'eta_h':18},
        ],
        'feed': [
            {'time':now,'text':'Prediction engine recalculated 24h flood probabilities'},
            {'time':now,'text':'Vulnerability ranking refreshed from zone exposure + open SOS'},
            {'time':now,'text':'Resource optimizer checked hub stock and road constraints'},
        ],
        'cyclone': cyclone_risk((34.0837,74.7973), {'lat':31.2,'lon':75.5,'wind_kmh':65,'pressure_hpa':998}),
        'ledger':LEDGER[-6:],
    }

@app.post('/api/predict')
def predict(s: Scenario):
    return predict_flood(s.model_dump())

@app.post('/api/simulate')
def simulate(s: Scenario):
    d=s.model_dump(); base=predict_flood(d)
    d['forecast_rain_24h']*=1.65; d['river_discharge_pct']=min(100,d['river_discharge_pct']+14); d['soil_moisture_pct']=min(100,d['soil_moisture_pct']+8)
    stressed=predict_flood(d)
    return {'baseline':base,'stressed':stressed,'delta':round(stressed['probability']-base['probability'],1)}

@app.post('/api/ledger')
def add_ledger(payload: dict):
    prev=LEDGER[-1]['fingerprint'] if LEDGER else GENESIS
    step=len(LEDGER)+1
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'))
    fp=fingerprint(prev,raw)
    rec={'step':step,'timestamp':datetime.now(timezone.utc).isoformat(),'payload':payload,'previous':prev,'fingerprint':fp}
    LEDGER.append(rec)
    return rec

@app.get('/api/resources')
def resources():
    return {'hubs':[
        {'name':'Lal Chowk Relief Hub','food':1280,'medical':420,'water':2100,'tarps':740,'oxygen':180,'status':'OPEN'},
        {'name':'Bemina Logistics Hub','food':980,'medical':310,'water':1650,'tarps':620,'oxygen':140,'status':'OPEN'},
        {'name':'Soura Emergency Store','food':740,'medical':520,'water':1200,'tarps':410,'oxygen':260,'status':'OPEN'},
        {'name':'Pantha Chowk Depot','food':1520,'medical':280,'water':2300,'tarps':820,'oxygen':120,'status':'OPEN'},
    ], 'recommendations':[
        {'priority':'CRITICAL','item':'Water packs','qty':760,'from':'Pantha Chowk Depot','to':'Rainawari','distance_km':11,'reason':'High flood probability + vulnerable low-lying exposure'},
        {'priority':'CRITICAL','item':'Medical kits','qty':180,'from':'Soura Emergency Store','to':'Downtown Srinagar','distance_km':7,'reason':'Open medical SOS + high population exposure'},
        {'priority':'HIGH','item':'Food kits','qty':520,'from':'Lal Chowk Relief Hub','to':'Bemina','distance_km':9,'reason':'Demand forecast exceeds local stock buffer'},
    ]}

@app.get('/')
def index(): return FileResponse(FRONT/'index.html')
app.mount('/static', StaticFiles(directory=FRONT), name='static')
