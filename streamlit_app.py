from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from folium.plugins import Fullscreen, MiniMap
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "srinagar_zones.json"
FAVICON = BASE / "assets" / "saarthi_favicon.png"

st.set_page_config(
    page_title="SAARTHI | India Disaster Command",
    page_icon=str(FAVICON),
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Theme / visual system
# The operator can switch between Saarthi Light and Saarthi Dark from the
# sidebar. The native Streamlit app menu remains available for normal actions
# (for example rerun/print), but its Settings item is hidden so theme changes
# are controlled only by the Saarthi sidebar control.
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Dark"

THEMES = {
    "Dark": {
        "bg":"#050A10",
        "panel":"#0A121C",
        "panel2":"#0D1B29",
        "line":"#193447",
        "sidebar":"#07111B",
        "text":"#F4FBFF",
        "muted":"#8EA7B7",
        "accent":"#18DDF2",
        "accent2":"#1E73FF",
        "accent3":"#35F3A4",
        # Severity colours stay semantic and are NOT replaced by brand colours.
        "critical":"#C83E3E",
        "high":"#F56616",
        "moderate":"#F58A4B",
        "low":"#2F8F5B",
        "info":"#2D6CDF",
        "shadow":"0 10px 30px rgba(0,0,0,.28)",
        "map":"OpenStreetMap",
        "alert":"rgba(24,221,242,.08)"
    },
    "Light": {
        "bg":"#F6F7F8",
        "panel":"#FFFFFF",
        "panel2":"#EEF3F6",
        "line":"#DDE2E6",
        "sidebar":"#E9EEF2",
        "text":"#111315",
        "muted":"#69737D",
        "accent":"#0BBED6",
        "accent2":"#1E73FF",
        "accent3":"#17B979",
        # Severity colours stay semantic and are NOT replaced by brand colours.
        "critical":"#C83E3E",
        "high":"#F56616",
        "moderate":"#F58A4B",
        "low":"#2F8F5B",
        "info":"#2D6CDF",
        "shadow":"0 8px 24px rgba(15,23,42,.08)",
        "map":"OpenStreetMap",
        "alert":"rgba(11,190,214,.08)"
    },
}
C = THEMES[st.session_state.ui_theme]
THEME = st.session_state.ui_theme.lower()

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {{
  --s-bg:{C['bg']}; --s-panel:{C['panel']}; --s-sidebar:{C.get('sidebar', C['panel'])}; --s-panel2:{C['panel2']}; --s-line:{C['line']};
  --s-text:{C['text']}; --s-muted:{C['muted']}; --s-accent:{C['accent']}; --s-accent2:{C['accent2']};
  --s-critical:{C['critical']}; --s-high:{C['high']}; --s-moderate:{C['moderate']}; --s-low:{C['low']}; --s-info:{C['info']}; --s-alert:{C.get('alert', C['bg'])}; --s-shadow:{C['shadow']};
}}
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
.stApp {{ background:var(--s-bg); color:var(--s-text); transition:background 350ms ease; }}
[data-testid="stAppViewContainer"] {{ background:var(--s-bg); }}
[data-testid="stHeader"] {{ background:transparent; }}
/* Keep Streamlit's native header and three-dot menu visible. Theme switching is handled by Saarthi's sidebar. */
[data-testid="stToolbar"] {{ visibility:visible !important; display:flex !important; }}
[data-testid="stHeader"] {{ background:transparent !important; height:auto !important; min-height:2.75rem !important; }}
[data-testid="stAppViewContainer"] {{ padding-top:0 !important; }}

.block-container {{ max-width:1600px; padding:1.25rem 2rem 2.5rem; }}
[data-testid="stSidebar"] {{ background:var(--s-sidebar); border-right:1px solid var(--s-line); box-shadow:8px 0 28px rgba(15,23,42,.08); }}
[data-testid="stSidebar"] * {{ color:var(--s-text); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ width:100% !important; box-sizing:border-box !important; font-size:1rem !important; font-weight:750 !important; padding:13px 14px !important; margin:4px 0 !important; border:1px solid transparent; border-radius:7px; transition:.18s ease; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:rgba(255,255,255,.32); border-color:var(--s-line); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background:var(--s-panel); border-color:var(--s-accent); box-shadow:0 3px 12px rgba(15,23,42,.08); width:100% !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{ width:100% !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {{ width:100% !important; }}
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span, [data-testid="stAppViewContainer"] label, [data-testid="stAppViewContainer"] small, [data-testid="stAppViewContainer"] li {{ color:var(--s-text); }}
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"], [data-testid="stAppViewContainer"] .stCaption {{ color:var(--s-muted) !important; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; color:var(--s-text) !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:var(--s-muted) !important; font-weight:500; }}
[data-testid="stMetric"] {{ background:var(--s-panel); border:1px solid var(--s-line); border-radius:16px; padding:15px 17px; box-shadow:{C['shadow']}; }}
[data-testid="stMetricLabel"] {{ color:var(--s-muted) !important; }}
[data-testid="stMetricValue"] {{ color:var(--s-text) !important; font-family:'Space Grotesk',sans-serif; }}
.stButton > button {{ border:1px solid var(--s-line); background:var(--s-panel); color:var(--s-text); border-radius:7px; font-weight:800; box-shadow:0 3px 10px rgba(15,23,42,.06); background-image:none !important; }}
.stButton > button:hover {{ border-color:var(--s-accent); color:var(--s-accent); background:var(--s-panel2); }}
[data-testid="stFormSubmitButton"] > button {{ background:linear-gradient(90deg,#35F3A4,#18DDF2,#1E73FF) !important; color:#fff !important; border-color:var(--s-accent) !important; background-image:none !important; }}
[data-testid="stFormSubmitButton"] > button:hover {{ filter:brightness(.94); color:#fff !important; }}
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {{ background:var(--s-panel) !important; color:var(--s-text) !important; border-color:var(--s-line) !important; }}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button:hover, [data-testid="stSidebar"] .stButton > button:hover {{ background:var(--s-panel2) !important; color:var(--s-accent) !important; border-color:var(--s-accent) !important; }}
[data-testid="stTextInput"] input {{ background:{C['panel2']} !important; color:{C['text']} !important; -webkit-text-fill-color:{C['text']} !important; border:1px solid {C['line']} !important; border-radius:7px !important; font-weight:650 !important; }}
[data-testid="stTextInput"] input::placeholder {{ color:{C['muted']} !important; opacity:1 !important; }}
[data-testid="stTextInput"] input:focus {{ border-color:{C['accent']} !important; box-shadow:0 0 0 2px rgba(24,221,242,.18) !important; }}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background:var(--s-panel); border-color:var(--s-line); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--s-line); border-radius:7px; overflow:hidden; background:var(--s-panel) !important; }}
[data-testid="stDataFrame"] iframe {{ background:var(--s-panel) !important; }}
[data-testid="stDataFrame"] [role="grid"] {{ background:var(--s-panel) !important; }}
[data-testid="stDataFrame"] [role="columnheader"] {{ background:var(--s-panel2) !important; color:var(--s-text) !important; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--s-line); border-radius:7px; background:var(--s-panel); }}
.saarthi-table {{ width:100%; border-collapse:collapse; font-size:.79rem; color:var(--s-text); background:var(--s-panel); }}
.saarthi-table th {{ background:var(--s-panel2); color:var(--s-text); text-align:left; padding:10px 11px; border-bottom:2px solid var(--s-line); font-weight:800; white-space:nowrap; }}
.saarthi-table td {{ background:var(--s-panel); color:var(--s-text); padding:9px 11px; border-bottom:1px solid var(--s-line); white-space:nowrap; }}
.saarthi-table tr:hover td {{ background:var(--s-panel2); }}
div[data-testid="stTabs"] {{ border-bottom:1px solid var(--s-line); }}
div[data-testid="stTabs"] button {{ color:var(--s-text); font-weight:700; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color:var(--s-accent); }}
hr {{ border-color:var(--s-line); }}
.hero {{ background:var(--s-panel); border:1px solid var(--s-line); border-radius:7px; padding:24px 26px; box-shadow:{C['shadow']}; }}
.brand {{ display:flex; align-items:center; gap:13px; }}
.brand .saarthi-logo {{ filter:drop-shadow(0 0 12px rgba(24,221,242,.12)); }}
.saarthi-logo {{ width:150px; height:120px; object-fit:contain; object-position:left center; display:block; }}
.brand-mark {{ width:42px; height:42px; border-radius:13px; display:grid; place-items:center; background:var(--s-accent); color:#fff; font-size:21px; box-shadow:0 8px 24px rgba(24,221,242,.22); }}
.brand-name {{ font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; letter-spacing:-.04em; }}
.eyebrow {{ color:var(--s-accent); text-transform:uppercase; font-size:.68rem; font-weight:800; letter-spacing:.16em; }}
.sub {{ color:var(--s-muted); font-size:.9rem; margin-top:4px; font-weight:500; }}
.live {{ display:inline-flex; align-items:center; gap:7px; font-size:.75rem; color:var(--s-low); font-weight:800; }}
.pulse {{ width:8px; height:8px; border-radius:50%; background:var(--s-low); box-shadow:0 0 12px var(--s-low); animation:pulse 1.7s infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(.78)}} }}
.section-head {{ display:flex; align-items:end; justify-content:space-between; gap:15px; margin:24px 0 10px; }}
.section-title {{ font-family:'Space Grotesk'; font-size:1.35rem; font-weight:700; }}
.section-note {{ color:var(--s-muted); font-size:.78rem; font-weight:500; }}
.card {{ background:var(--s-panel); border:1px solid var(--s-line); border-radius:7px; padding:17px; box-shadow:{C['shadow']}; }}
.card-tight {{ padding:13px 15px; }}
.kpi-label {{ color:var(--s-muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }}
.kpi-value {{ font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin-top:3px; }}
.kpi-foot {{ color:var(--s-muted); font-size:.72rem; margin-top:2px; }}
.badge {{ display:inline-flex; align-items:center; justify-content:center; min-width:64px; padding:4px 7px; border-radius:6px; font-size:.61rem; font-weight:900; letter-spacing:.08em; color:#fff !important; border:1px solid rgba(255,255,255,.55); box-shadow:0 4px 12px rgba(15,23,42,.14); }}
.compact-badge {{ min-width:44px; padding:2px 5px; font-size:.56rem; line-height:1.2; border-radius:5px; box-shadow:none; border-width:1px; }}
.critical {{ background:#C83E3E !important; color:#fff !important; border-color:#b91c1c !important; }}
.high {{ background:#F56616 !important; color:#fff !important; border-color:#c2410c !important; }}
.moderate {{ background:#F58A4B !important; color:#fff !important; border-color:#ea580c !important; }}
.low {{ background:#2F8F5B !important; color:#fff !important; border-color:#15803d !important; }}
.queue-card {{ border:1px solid var(--s-line); border-radius:17px; padding:14px; background:var(--s-panel); box-shadow:var(--s-shadow); }}
.queue-item {{ border:1px solid var(--s-line); border-radius:7px; padding:13px; margin-top:9px; background:var(--s-panel2); box-shadow:0 3px 10px rgba(15,23,42,.05); }}
.queue-item:last-child {{ border-bottom:1px solid var(--s-line); }}
.queue-row {{ display:flex; align-items:center; gap:8px; }}
.queue-number {{ width:34px; height:34px; display:grid; place-items:center; border-radius:8px; background:var(--s-bg); border:1px solid var(--s-line); font-weight:900; font-size:1rem; }}
.queue-name {{ font-weight:800; margin:5px 0 2px; }}
.muted {{ color:var(--s-muted); }}
.small {{ font-size:.76rem; color:var(--s-muted); font-weight:600; }}
.plotly .legendtext, .plotly .legendtitle, .plotly .g-legend text {{ fill:var(--s-text) !important; color:var(--s-text) !important; }}
.map-mini {{ border:1px solid var(--s-line); border-radius:7px; overflow:hidden; background:var(--s-panel); box-shadow:{C['shadow']}; padding:4px; }}
.chart-shell {{ border:1px solid var(--s-line); border-radius:16px; padding:8px 8px 2px; background:var(--s-panel); box-shadow:0 8px 24px rgba(15,23,42,.06); }}
div[data-testid="stPlotlyChart"] {{ border:1px solid var(--s-line); border-radius:14px; overflow:hidden; background:var(--s-panel); }}
.threat-card {{ border:1px solid var(--s-line); border-radius:7px; padding:22px 24px; margin-bottom:16px; box-shadow:var(--s-shadow); }}
.threat-city {{ font-family:'Space Grotesk'; font-size:1.75rem; font-weight:800; margin:6px 0; }}
.threat-main {{ color:var(--s-text); font-size:.95rem; margin:6px 0; }}
.threat-action {{ margin-top:12px; font-size:.92rem; }}
.recommend-card {{ background:var(--s-panel); border:1px solid var(--s-line); border-radius:7px; padding:17px; box-shadow:var(--s-shadow); min-height:180px; }}
.recommend-title {{ font-family:'Space Grotesk'; font-size:1.05rem; font-weight:800; margin-bottom:3px; }}
.action-row {{ display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-top:1px solid var(--s-line); font-size:.82rem; }}
.action-num {{ min-width:24px; height:24px; display:grid; place-items:center; border-radius:8px; background:var(--s-accent); color:#fff; font-weight:900; }}
.reason-row {{ padding:8px 0; border-top:1px solid var(--s-line); font-size:.8rem; font-weight:600; }}
.movement-card {{ background:var(--s-panel); border:1px solid var(--s-line); border-radius:7px; padding:13px 15px; margin:8px 0; }}
.movement-line {{ display:flex; align-items:center; gap:12px; font-size:.84rem; }}
.movement-arrow {{ color:var(--s-accent); font-weight:900; }}
.movement-progress {{ margin-top:10px; }}
.movement-progress-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; font-size:.75rem; font-weight:800; }}
.movement-progress-track {{ height:9px; background:#27323a; border:1px solid var(--s-line); border-radius:5px; overflow:hidden; margin-top:6px; }}
.movement-progress-fill {{ height:100%; background:var(--s-accent); border-radius:4px; }}
.movement-progress-fill.stable {{ background:var(--s-low); }}
.movement-progress-fill.info {{ background:var(--s-info); }}
.source-pill {{ display:inline-flex; padding:5px 8px; border:1px solid var(--s-line); border-radius:6px; color:var(--s-text); font-size:.68rem; font-weight:700; margin-right:5px; }}
.warning-strip {{ border:1px solid rgba(245,158,11,.35); background:rgba(245,158,11,.08); border-radius:13px; padding:11px 13px; color:var(--s-text); font-size:.8rem; }}
.success-strip {{ border:1px solid rgba(34,197,94,.28); background:rgba(34,197,94,.07); border-radius:13px; padding:11px 13px; color:var(--s-text); font-size:.8rem; }}
.nav-title {{ font-family:'Space Grotesk'; font-size:1.1rem; font-weight:700; }}
.nav-caption {{ color:var(--s-muted); font-size:.72rem; line-height:1.4; }}
/* Full-theme widget overrides */
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea{{background:var(--s-panel)!important;color:var(--s-text)!important;-webkit-text-fill-color:var(--s-text)!important;border-color:var(--s-line)!important}}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{{background:var(--s-panel)!important;border-color:var(--s-line)!important}}
[data-baseweb="select"] *,[data-baseweb="input"] *,[data-baseweb="textarea"] *{{color:var(--s-text)!important;-webkit-text-fill-color:var(--s-text)!important}}
div[role="listbox"],div[role="option"]{{background:var(--s-panel)!important;color:var(--s-text)!important}}
div[role="option"]:hover,div[role="option"][aria-selected="true"]{{background:var(--s-panel2)!important;color:var(--s-text)!important}}
[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"],.stApp{{background:var(--s-bg)!important;color:var(--s-text)!important}}
[data-testid="stSidebar"]{{background:var(--s-sidebar)!important}}
.stButton>button,[data-testid="stFormSubmitButton"]>button{{background:var(--s-panel)!important;color:var(--s-text)!important;border-color:var(--s-line)!important;background-image:none!important}}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{{background:var(--s-panel2)!important;color:var(--s-accent)!important;border-color:var(--s-accent)!important}}
[data-testid="stFormSubmitButton"]>button[kind="primary"]{{background:linear-gradient(90deg,#35F3A4,#18DDF2,#1E73FF)!important;color:#fff!important;border-color:var(--s-accent)!important}}
footer {{ visibility:hidden; }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Location intelligence: any place in India
# -----------------------------------------------------------------------------
INDIA_CITY_FALLBACKS = {
    "srinagar": (34.0837, 74.7973), "jammu": (32.7266, 74.8570),
    "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777), "pune": (18.5204, 73.8567),
    "nagpur": (21.1458, 79.0882), "nashik": (19.9975, 73.7898),
    "ahmedabad": (23.0225, 72.5714), "surat": (21.1702, 72.8311),
    "jaipur": (26.9124, 75.7873), "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319), "varanasi": (25.3176, 82.9739),
    "patna": (25.5941, 85.1376), "kolkata": (22.5726, 88.3639),
    "bhubaneswar": (20.2961, 85.8245), "guwahati": (26.1445, 91.7362),
    "ranchi": (23.3441, 85.3096), "bhopal": (23.2599, 77.4126),
    "indore": (22.7196, 75.8577), "chandigarh": (30.7333, 76.7794),
    "dehradun": (30.3165, 78.0322), "shimla": (31.1048, 77.1734),
    "amritsar": (31.6340, 74.8723), "ludhiana": (30.9010, 75.8573),
    "hyderabad": (17.3850, 78.4867), "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946), "chennai": (13.0827, 80.2707),
    "kochi": (9.9312, 76.2673), "thiruvananthapuram": (8.5241, 76.9366),
    "visakhapatnam": (17.6868, 83.2185), "vijayawada": (16.5062, 80.6480),
    "goa": (15.2993, 74.1240), "panaji": (15.4909, 73.8278),
    "raipur": (21.2514, 81.6296), "vadodara": (22.3072, 73.1812),
    "agra": (27.1767, 78.0081), "meerut": (28.9845, 77.7064),
}

@st.cache_data(ttl=86400)
def geocode_india(place: str):
    q = place.strip()
    if not q:
        return None
    key = q.lower().replace(", india", "").strip()
    if key in INDIA_CITY_FALLBACKS:
        lat, lon = INDIA_CITY_FALLBACKS[key]
        return {"name": q.title(), "display": f"{q.title()}, India", "lat": lat, "lon": lon, "source": "built-in city reference"}
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{q}, India", "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers={"User-Agent": "SAARTHI-Disaster-Command-Center/1.0"}, timeout=8,
        )
        r.raise_for_status()
        results = r.json()
        if results:
            item = results[0]
            return {"name": q.title(), "display": item.get("display_name", f"{q.title()}, India"),
                    "lat": float(item["lat"]), "lon": float(item["lon"]), "source": "OpenStreetMap geocoder"}
    except Exception:
        pass
    return None


def location_profile(lat, lon):
    """Coarse city-level baseline used when detailed local hydrology is unavailable."""
    # Indian city-level estimate: weather drives the live hazard signal; these are
    # transparent baseline assumptions, not a replacement for local agency data.
    coastal = (7.0 <= lat <= 24.5 and 68.0 <= lon <= 89.5 and (
        lon < 74.5 or lon > 79.5 or lat < 12.5 or lat > 20.0
    ))
    hill = lat >= 28 or (lat >= 24 and lon < 78) or lat < 12
    elevation = 120 if not hill else 650
    river_distance = 2.0 if not coastal else 4.0
    drainage = 2.4 if coastal else 1.7
    population_density = 7000
    vulnerable_pct = 22 if coastal else 18
    soil = 62 if coastal else 52
    discharge = 55 if coastal else 48
    return {"coastal": coastal, "hill": hill, "elevation": elevation, "river_distance": river_distance,
            "drainage": drainage, "population_density": population_density,
            "vulnerable_pct": vulnerable_pct, "soil": soil, "discharge": discharge}


def generate_city_state(location, weather):
    """Create six operational sectors around any geocoded Indian city."""
    lat, lon = location["lat"], location["lon"]
    profile = location_profile(lat, lon)
    rain24 = float(weather.get("rain_24h", 0) or 0) if weather.get("ok") else 35.0
    pop = float(weather.get("max_pop", 0) or 0) if weather.get("ok") else 35.0
    # Keep the live weather signal inside sensible model ranges.
    rain24 = float(np.clip(rain24, 0, 260))
    forecast = max(0.1, rain24)
    offsets = [(0.000,0.000,"Central"),(0.045,0.000,"North"),(-0.045,0.000,"South"),
               (0.000,0.055,"East"),(0.000,-0.055,"West"),(0.032,0.040,"North-East")]
    vuln = [0.62,0.56,0.51,0.47,0.41,0.35]
    zones=[]
    for i,(dlat,dlon,label) in enumerate(offsets):
        local_rain = max(0, forecast*(1 + [0.10,0.04,-0.03,0.07,-0.05,0.02][i]))
        f={
            "rain_1h": local_rain/8, "rain_6h": local_rain/2.2, "rain_24h": local_rain,
            "rain_72h": local_rain*1.45, "forecast_rain_24h": local_rain,
            "river_discharge_pct": profile["discharge"] + (i%3)*4,
            "river_rise_6h_pct": 8 + (i%4)*4,
            "soil_moisture_pct": profile["soil"] + (i%3)*5,
            "elevation_m": profile["elevation"] + (i%3)*35,
            "slope_pct": 2 + (i%4)*1.2, "river_distance_km": profile["river_distance"] + (i%3)*0.8,
            "drainage_density": profile["drainage"] + (i%2)*0.7,
            "population_density": profile["population_density"]*(1 + i*0.04),
            "vulnerable_population_pct": profile["vulnerable_pct"] + i*1.5,
        }
        zones.append({"name":f"{location['name']} — {label} Sector", "lat":lat+dlat, "lon":lon+dlon,
                      "population":int(70000*(1+0.08*i)), "vulnerability":vuln[i],
                      "open_sos":1 if i==0 else (1 if i in (1,3) else 0), "features":f})
    hubs=[{"name":f"{location['name']} Relief Hub","lat":lat+0.012,"lon":lon-0.018,"food":1800,"medical":420,"water":3200,"tarps":1200,"oxygen":900,"open":True},
          {"name":f"{location['name']} Emergency Depot","lat":lat-0.018,"lon":lon+0.022,"food":1400,"medical":320,"water":2400,"tarps":950,"oxygen":700,"open":True}]
    sos=[{"zone":zones[0]["name"],"need":"Rapid assessment","people":90,"status":"OPEN","lat":zones[0]["lat"],"lon":zones[0]["lon"]},
         {"zone":zones[3]["name"],"need":"Water","people":60,"status":"OPEN","lat":zones[3]["lat"],"lon":zones[3]["lon"]}]
    return zones,hubs,sos

# -----------------------------------------------------------------------------
# Data and model
# -----------------------------------------------------------------------------
@st.cache_data
def load_zones():
    with open(DATA, "r", encoding="utf-8") as f:
        return json.load(f)

FEATURES = [
    "rain_1h", "rain_6h", "rain_24h", "rain_72h", "forecast_rain_24h",
    "river_discharge_pct", "river_rise_6h_pct", "soil_moisture_pct",
    "elevation_m", "slope_pct", "river_distance_km", "drainage_density",
    "population_density", "vulnerable_population_pct",
]


def make_training_data(n=9000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.gamma(1.6, 3.0, n), rng.gamma(1.7, 7.0, n), rng.gamma(1.8, 15.0, n),
        rng.gamma(1.7, 28.0, n), rng.gamma(1.8, 18.0, n),
        rng.beta(2.5, 1.6, n) * 100, rng.normal(0, 15, n).clip(-40, 80),
        rng.beta(3.0, 2.0, n) * 100, rng.normal(1580, 450, n).clip(900, 3200),
        rng.gamma(1.8, 2.0, n).clip(0, 25), rng.gamma(2.0, 1.4, n).clip(.05, 12),
        rng.gamma(2.0, .9, n).clip(.1, 8), rng.lognormal(7.5, .8, n).clip(100, 25000),
        rng.beta(2.0, 6.0, n) * 100,
    ])
    z = (0.018*x[:,1] + 0.010*x[:,2] + 0.005*x[:,3] + 0.014*x[:,4]
         + 0.018*x[:,5] + 0.020*np.maximum(x[:,6], 0) + 0.012*x[:,7]
         - 0.0007*(x[:,8]-1500) - 0.025*x[:,9] - 0.11*x[:,10]
         + 0.07*x[:,11] + 0.000015*x[:,12] + 0.010*x[:,13] - 5.2)
    p = 1/(1+np.exp(-z))
    y = (rng.random(n) < p).astype(int)
    return x, y


@st.cache_resource
def get_model():
    x, y = make_training_data()
    model = RandomForestClassifier(
        n_estimators=220, max_depth=11, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    model.fit(x, y)
    return model


MODEL = get_model()


def predict(features):
    row = np.array([[float(features[k]) for k in FEATURES]])
    return float(MODEL.predict_proba(row)[0, 1] * 100)


def severity(prob, vulnerability, sos):
    score = prob * vulnerability + min(12, sos * 2)
    if score >= 60: level = "CRITICAL"
    elif score >= 40: level = "HIGH"
    elif score >= 20: level = "MODERATE"
    else: level = "LOW"
    return score, level


def haversine(a, b):
    lat1, lon1 = a; lat2, lon2 = b
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(h))


def cyclone_risk(center, storm):
    d = haversine(center, (storm["lat"], storm["lon"]))
    wind = min(1, max(0, (storm["wind_kmh"]-50)/140))
    pressure = min(1, max(0, (1015-storm["pressure_hpa"])/80))
    proximity = math.exp(-d/180)
    p = 100*(0.55*proximity*wind + 0.45*proximity*pressure)
    return min(99.9, p), d


@st.cache_data
def base_state():
    zones = load_zones()
    hubs = [
        {"name":"Lal Chowk Relief Hub","lat":34.0522,"lon":74.8370,"food":1250,"medical":340,"water":2600,"tarps":900,"oxygen":760,"open":True},
        {"name":"Bemina Warehouse","lat":34.0220,"lon":74.7580,"food":1900,"medical":420,"water":3100,"tarps":1200,"oxygen":980,"open":True},
        {"name":"Pantha Chowk Depot","lat":33.9990,"lon":74.8760,"food":1500,"medical":280,"water":2200,"tarps":1000,"oxygen":640,"open":True},
        {"name":"Hazratbal Relief Hub","lat":34.1190,"lon":74.8670,"food":980,"medical":230,"water":1800,"tarps":700,"oxygen":520,"open":True},
    ]
    sos = [
        {"zone":"Rainawari","need":"Water","people":120,"status":"OPEN","lat":34.030,"lon":74.832},
        {"zone":"Downtown Srinagar","need":"Medical","people":64,"status":"OPEN","lat":34.0466,"lon":74.813},
        {"zone":"Soura","need":"Shelter","people":85,"status":"IN TRANSIT","lat":34.0587,"lon":74.8422},
        {"zone":"Bemina","need":"Rescue","people":48,"status":"OPEN","lat":34.0399,"lon":74.7572},
    ]
    return zones, hubs, sos


def vulnerability_components(z):
    """Transparent 0-100 vulnerability model for operational prioritization.

    Weights: population exposure 25%, critical facilities 20%, accessibility 20%,
    resource shortage 15%, SOS pressure 10%, hazard-specific exposure 10%.
    These are Saarthi design weights for the prototype, not an official government index.
    """
    f = z["features"]
    population = float(np.clip(f["population_density"] / 18000 * 100, 0, 100))
    facilities = float(np.clip((f["population_density"] / 12000 * 65) + (f["vulnerable_population_pct"] * 0.35), 0, 100))
    accessibility = float(np.clip(100 - (f["river_distance_km"] * 9 + f["slope_pct"] * 3.5), 0, 100))
    resource_shortage = float(np.clip(35 + f["vulnerable_population_pct"] * 0.9 + max(0, 60 - f["drainage_density"] * 10), 0, 100))
    sos_pressure = float(np.clip(z.get("open_sos", 0) * 42, 0, 100))
    hazard_exposure = float(np.clip(45 + f["soil_moisture_pct"] * 0.35 + f["river_discharge_pct"] * 0.25 + max(0, 5 - f["river_distance_km"]) * 4, 0, 100))
    weighted = (0.25*population + 0.20*facilities + 0.20*accessibility + 0.15*resource_shortage + 0.10*sos_pressure + 0.10*hazard_exposure)
    return {
        "population_exposure": round(population, 1),
        "critical_facilities": round(facilities, 1),
        "accessibility": round(accessibility, 1),
        "resource_shortage": round(resource_shortage, 1),
        "sos_pressure": round(sos_pressure, 1),
        "hazard_exposure": round(hazard_exposure, 1),
        "vulnerability": round(weighted / 100, 3),
    }


def compute_zones(zones, rainfall_multiplier=1.0, discharge_delta=0, soil_delta=0, sos_boost=0):
    out=[]
    for z in zones:
        f=dict(z["features"])
        for k in ["rain_1h","rain_6h","rain_24h","rain_72h","forecast_rain_24h"]:
            f[k] *= rainfall_multiplier
        f["river_discharge_pct"] = np.clip(f["river_discharge_pct"] + discharge_delta, 0, 100)
        f["soil_moisture_pct"] = np.clip(f["soil_moisture_pct"] + soil_delta, 0, 100)
        p = predict(f)
        working = {**z, "features": f, "open_sos": z["open_sos"] + sos_boost}
        vc = vulnerability_components(working)
        sos = working["open_sos"]
        score, level = severity(p, vc["vulnerability"], sos)
        out.append({**working, **vc, "probability":round(p,1), "severity_score":round(score,1), "level":level})
    return sorted(out, key=lambda x:x["severity_score"], reverse=True)


def allocations(zone_rows, hubs):
    priorities = {"CRITICAL":1.0,"HIGH":.72,"MODERATE":.42,"LOW":.18}
    rec=[]
    for z in zone_rows:
        factor = priorities[z["level"]]
        if factor < .4: continue
        population = z["population"]
        for item, base_per_1000 in [("Food kits",34),("Medical kits",5),("Water packs",55),("Tarps",9)]:
            stock_key={"Food kits":"food","Medical kits":"medical","Water packs":"water","Tarps":"tarps"}[item]
            qty=max(1, round(population/1000*base_per_1000*factor))
            available=[h for h in hubs if h["open"] and h[stock_key] > 0]
            if not available: continue
            src=min(available, key=lambda h:h[stock_key])
            qty=min(qty, src[stock_key])
            rec.append({"Priority":z["level"],"Item":item,"Qty":qty,"From":src["name"],"To":z["name"],"Distance km":round(haversine((src["lat"],src["lon"]),(z["lat"],z["lon"])),1),"Score":z["severity_score"]})
    columns = ["Priority", "Item", "Qty", "From", "To", "Distance km", "Score"]
    return pd.DataFrame(rec, columns=columns)


def generate_hospitals(location, df):
    """Create a small operational hospital view tied to the selected city."""
    lat, lon = location["lat"], location["lon"]
    top = df.head(3).reset_index(drop=True)
    names = [f"{location['name']} General", f"{location['name']} City Hospital", f"{location['name']} Emergency Centre", f"{location['name']} District Hospital"]
    hospitals = []
    for i, name in enumerate(names):
        row = top.iloc[i % len(top)]
        beds = [120, 95, 80, 110][i]
        icu = max(0, [12, 7, 5, 9][i] - int(row["severity_score"] // 22))
        oxygen = max(12, 92 - int(row["probability"] * .42) - i * 6)
        status = "CRITICAL" if oxygen < 30 or icu <= 1 else ("ATTENTION" if oxygen < 55 or icu <= 3 else "READY")
        hospitals.append({"Hospital": name, "Beds": beds, "ICU free": icu, "Oxygen": f"{oxygen}%", "Status": status, "zone": row["name"], "lat": lat + [0.018,-0.012,0.028,-0.025][i], "lon": lon + [0.026,0.032,-0.020,-0.035][i]})
    return pd.DataFrame(hospitals)


def build_recommendation(df, hubs, hospitals):
    top = df.iloc[0]
    critical = df[df["level"] == "CRITICAL"]
    target = critical.iloc[0] if not critical.empty else top
    hospital = hospitals.sort_values(["ICU free", "Oxygen"]).iloc[0]
    src = hubs[0]
    if len(hubs) > 1:
        src = max(hubs, key=lambda h: h["oxygen"] + h["medical"] / 2)
    actions = [
        f"Pre-position {min(20, src['oxygen'])} oxygen cylinders from {src['name']} toward {hospital['Hospital']}",
        f"Prepare {max(2, int(target['population'] / 30000))} ambulance/rescue teams for {target['name']}",
        f"Move {min(100, src['medical'])} medical kits toward {target['name']}",
        f"Monitor {target['name']} for road accessibility and new SOS pressure",
    ]
    summary = (
        "Pre-position medical and rescue supplies" if target["level"] in ("CRITICAL", "HIGH")
        else "Maintain readiness and stage supplies near the highest-risk sector"
    )
    return summary, actions, target, hospital, src


def resource_movements(df, hubs, hospitals):
    target = df.iloc[0]
    hospital = hospitals.iloc[0]
    src = hubs[0]
    rows = [
        {"From": src["name"], "To": hospital["Hospital"], "Resource": "Oxygen cylinders", "Qty": min(20, src["oxygen"]), "ETA": "18 min", "Status": "READY"},
        {"From": hubs[-1]["name"], "To": target["name"], "Resource": "Emergency kits", "Qty": min(100, hubs[-1]["medical"]), "ETA": "24 min", "Status": "STAGED"},
    ]
    return pd.DataFrame(rows)


def decision_reasons(target, hospital, source):
    reasons = [
        f"{target['probability']:.0f}% 24h hazard probability",
        f"Vulnerability index {target['vulnerability']:.2f}",
        f"{int(target['open_sos'])} open SOS signal(s) in the sector",
        f"Hospital oxygen at {hospital['Oxygen']} with {hospital['ICU free']} ICU beds free",
        f"Nearest selected relief source: {source['name']}",
        "Allocation is re-ranked when rainfall, river discharge, soil saturation or SOS pressure changes",
    ]
    return reasons

# -----------------------------------------------------------------------------
# Dynamic emergency-resource optimizer
# -----------------------------------------------------------------------------
def generate_response_assets(location, df):
    """Create a transparent scenario model for ambulances, roads, hospitals and funds."""
    lat, lon = location["lat"], location["lon"]
    ambulances = [
        {"Unit":"AMB-01","Capability":"ALS","Seats":2,"Medical kits":18,"Status":"AVAILABLE","lat":lat+0.018,"lon":lon+0.020},
        {"Unit":"AMB-02","Capability":"BLS","Seats":4,"Medical kits":12,"Status":"AVAILABLE","lat":lat-0.015,"lon":lon+0.030},
        {"Unit":"AMB-03","Capability":"ALS","Seats":2,"Medical kits":22,"Status":"AVAILABLE","lat":lat+0.028,"lon":lon-0.024},
        {"Unit":"AMB-04","Capability":"Rescue","Seats":6,"Medical kits":8,"Status":"STAGED","lat":lat-0.026,"lon":lon-0.020},
    ]
    hospitals = []
    top = df.head(4).reset_index(drop=True)
    names = [f"{location['name']} General", f"{location['name']} City Hospital", f"{location['name']} Emergency Centre", f"{location['name']} District Hospital"]
    for i, name in enumerate(names):
        row = top.iloc[i % len(top)]
        beds = [120,95,80,110][i]
        occupancy = min(98, 54 + int(row["severity_score"] * .45) + i*5)
        free = max(2, beds - round(beds*occupancy/100))
        icu_total = [14,10,8,12][i]
        icu_free = max(0, icu_total - int(row["severity_score"]//18))
        oxygen = max(15, 94 - int(row["probability"]*.48) - i*7)
        hospitals.append({"Hospital":name,"Beds":beds,"Occupancy %":occupancy,"Beds free":free,"ICU total":icu_total,"ICU free":icu_free,"Oxygen %":oxygen,"Status":"CRITICAL" if icu_free<=1 or oxygen<30 else ("ATTENTION" if icu_free<=3 or oxygen<55 else "READY"),"zone":row["name"],"lat":lat+[.018,-.012,.028,-.025][i],"lon":lon+[.026,.032,-.020,-.035][i]})
    roads=[]
    for i, row in df.iterrows():
        severity_pressure=float(row["severity_score"])
        status="BLOCKED" if severity_pressure>=65 else ("SLOW" if severity_pressure>=40 else "OPEN")
        roads.append({"Road segment":f"{row['name']} access corridor","Zone":row["name"],"Status":status,"Travel multiplier":3.0 if status=="BLOCKED" else (1.7 if status=="SLOW" else 1.0),"Reason":"Flood depth / debris risk" if status!="OPEN" else "Passable"})
    return pd.DataFrame(ambulances), pd.DataFrame(hospitals), pd.DataFrame(roads)

def optimize_response(df, hubs, ambulances, hospitals, roads, fund_limit=250000):
    """Greedy constrained optimizer: urgency + capability + capacity + route + stock + funds."""
    priority_weight={"CRITICAL":1.0,"HIGH":.78,"MODERATE":.48,"LOW":.20}
    road_map=dict(zip(roads["Zone"], roads["Travel multiplier"]))
    available_amb=ambulances[ambulances["Status"].isin(["AVAILABLE","STAGED"])].copy()
    remaining_fund=float(fund_limit)
    plans=[]
    used=set()
    for _, z in df.sort_values("severity_score",ascending=False).iterrows():
        if z["level"]=="LOW": continue
        urgency=priority_weight[z["level"]]*(0.65*z["probability"]+0.35*z["vulnerability"]*100)
        candidates=[]
        for _, h in hospitals.iterrows():
            dist=haversine((float(z["lat"]),float(z["lon"])),(float(h["lat"]),float(h["lon"])))
            capacity=(h["ICU free"]*2+h["Beds free"])/25
            candidates.append((capacity/(1+dist*0.05),h,dist))
        candidates.sort(key=lambda x:x[0],reverse=True)
        if not candidates: continue
        _, hospital, dist=candidates[0]
        road_factor=float(road_map.get(z["name"],1.0))
        amb_candidates=[]
        for _, a in available_amb.iterrows():
            if a["Unit"] in used: continue
            capability_bonus=1.25 if (z["level"]=="CRITICAL" and a["Capability"] in ["ALS","Rescue"]) else 1.0
            adist=haversine((float(a["lat"]),float(a["lon"])),(float(z["lat"]),float(z["lon"])))
            score=capability_bonus/(1+adist*0.12*road_factor)
            amb_candidates.append((score,a,adist))
        if not amb_candidates: continue
        amb_score, amb, adist=max(amb_candidates,key=lambda x:x[0])
        est_cost=round(9000 + z["population"]*.12 + dist*650*road_factor)
        if est_cost > remaining_fund: continue
        quantity=max(1,min(120,int(z["population"]*priority_weight[z["level"]]/1400)))
        medical_need=max(1,min(35,int(z["population"]/12000*priority_weight[z["level"]]*8)))
        plans.append({"Priority":z["level"],"Zone":z["name"],"Urgency score":round(urgency,1),"Ambulance":amb["Unit"],"Capability":amb["Capability"],"Hospital":hospital["Hospital"],"Beds free":int(hospital["Beds free"]),"ICU free":int(hospital["ICU free"]),"Road":roads.loc[roads["Zone"]==z["name"],"Status"].iloc[0],"Travel factor":road_factor,"Relief units":quantity,"Medical kits":medical_need,"Est. fund ₹":est_cost,"Allocation status":"ALLOCATED"})
        used.add(amb["Unit"]); remaining_fund-=est_cost
    return pd.DataFrame(plans), round(remaining_fund,2)

def privacy_view(payload, role="Command Officer"):
    """Apply minimum-necessary disclosure to auditable transaction records."""
    out=dict(payload)
    if role not in ("Command Officer","Auditor"):
        for key in ["Beneficiary name","Phone","Exact address","Personal ID"]:
            out.pop(key,None)
    if role=="Field Responder":
        out.pop("Fund amount",None)
    return out


# -----------------------------------------------------------------------------
# Weather integration: Open-Meteo only
# -----------------------------------------------------------------------------
def secret_or_env(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def _weather_result(source, rain, pop, temp=None, humidity=None, wind=None, pressure=None):
    return {
        "ok": True,
        "source": source,
        "rain_24h": round(sum(float(x or 0) for x in rain[:24]), 1),
        "max_pop": round(max((float(x or 0) for x in pop[:24]), default=0.0), 1),
        "temperature": temp,
        "humidity": humidity,
        "wind": wind,
        "pressure": pressure,
        "updated": datetime.now(timezone.utc).strftime("%H:%M UTC"),
    }


def fetch_open_meteo(lat=34.045, lon=74.82):
    """Fetch the 24-hour forecast from Open-Meteo only."""
    api_key = secret_or_env("OPEN_METEO_API_KEY") or secret_or_env("OPENMETEO_API_KEY")
    if api_key:
        base_url = secret_or_env("OPEN_METEO_BASE_URL", "https://customer-api.open-meteo.com/v1/forecast")
    else:
        base_url = secret_or_env("OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,surface_pressure",
        "forecast_hours": 24,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    if api_key:
        params["apikey"] = api_key
    try:
        r = requests.get(base_url, params=params, timeout=12)
        if r.status_code in (401, 403):
            return {"ok": False, "source": "Open-Meteo", "error": f"API authentication failed ({r.status_code}). Check OPEN_METEO_API_KEY."}
        r.raise_for_status()
        payload = r.json()
        hourly = payload.get("hourly") or {}
        rain = hourly.get("precipitation") or []
        pop = hourly.get("precipitation_probability") or []
        temp = hourly.get("temperature_2m") or []
        humidity = hourly.get("relative_humidity_2m") or []
        wind = hourly.get("wind_speed_10m") or []
        pressure = hourly.get("surface_pressure") or []
        if not rain and not pop:
            return {"ok": False, "source": "Open-Meteo", "error": "Open-Meteo returned no hourly forecast values."}
        result = _weather_result("Open-Meteo", rain, pop, temp[0] if temp else None,
                                 humidity[0] if humidity else None,
                                 wind[0] if wind else None,
                                 pressure[0] if pressure else None)
        result["timezone"] = payload.get("timezone")
        result["model"] = payload.get("generationtime_ms")
        return result
    except requests.RequestException as e:
        return {"ok": False, "source": "Open-Meteo", "error": f"Network/API error: {str(e)[:150]}"}
    except Exception as e:
        return {"ok": False, "source": "Open-Meteo", "error": str(e)[:160]}


def fetch_weather(lat, lon, source="Open-Meteo"):
    """Use Open-Meteo as the sole selectable live weather source."""
    return fetch_open_meteo(lat, lon)


# -----------------------------------------------------------------------------
# Theme-aware tables
# -----------------------------------------------------------------------------
def show_table(frame: pd.DataFrame, *, height=None, level_column=None):
    """Render a theme-locked HTML table so Streamlit cannot retain dark headers in Day Mode."""
    if frame is None or frame.empty:
        st.info("No records to display.")
        return
    display = frame.copy()
    level_colors = {"CRITICAL": C["critical"], "HIGH": C["high"], "MODERATE": C["moderate"], "LOW": C["low"]}
    html = ["<div class='table-wrap'><table class='saarthi-table'><thead><tr>"]
    html += [f"<th>{str(c)}</th>" for c in display.columns]
    html += ["</tr></thead><tbody>"]
    for _, row in display.iterrows():
        html.append("<tr>")
        for col, val in row.items():
            text = "—" if pd.isna(val) else str(val)
            style = ""
            if level_column and col == level_column:
                raw = str(val).upper()
                if raw in level_colors:
                    compact_cls = raw.lower()
                    text = f"<span class='badge compact-badge {compact_cls}'>{text}</span>"
            html.append(f"<td style='{style}'>{text}</td>")
        html.append("</tr>")
    html += ["</tbody></table></div>"]
    st.markdown("".join(html), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Map
# -----------------------------------------------------------------------------
def build_map(df, hubs, sos, basemap="OpenStreetMap", center=None):
    # Fixed street-map mode for operational consistency.
    m = folium.Map(
        location=(center or [34.0837, 74.7973]), zoom_start=12, control_scale=True,
        tiles="OpenStreetMap", prefer_canvas=True, max_zoom=19,
    )
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    colors={"CRITICAL":"#C83E3E","HIGH":"#F56616","MODERATE":"#F58A4B","LOW":"#2F8F5B"}
    for _, r in df.iterrows():
        c = colors[r.level]
        popup = f"""
        <div style='font-family:Arial;min-width:210px'>
          <b style='font-size:15px'>{r['name']}</b><br>
          <span style='color:{c};font-weight:800'>{r['level']}</span>
          <hr style='border:0;border-top:1px solid #ddd'>
          24h flood probability: <b>{r['probability']}%</b><br>
          Vulnerability index: <b>{r['vulnerability']:.2f}</b><br>
          Severity score: <b>{r['severity_score']}</b><br>
          Population: <b>{r['population']:,}</b>
        </div>"""
        folium.Circle(
            [r.lat,r.lon], radius=650 + r.severity_score*18,
            color=c, fill=True, fill_color=c, fill_opacity=.18, weight=2,
            popup=folium.Popup(popup, max_width=310), tooltip=f"{r['name']} · {r['level']}",
        ).add_to(m)
        folium.Marker(
            [r.lat,r.lon],
            icon=folium.DivIcon(html=f"<div style='width:18px;height:18px;background:{c};border:2px solid white;border-radius:50%;box-shadow:0 0 12px {c};'></div>"),
        ).add_to(m)

    for h in hubs:
        folium.Marker(
            [h["lat"],h["lon"]], tooltip=f"RELIEF HUB · {h['name']}",
            popup=f"<b>{h['name']}</b><br>Food {h['food']:,} · Water {h['water']:,} · Medical {h['medical']:,}",
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(m)
    for s in sos:
        c="#C83E3E" if s["status"]=="OPEN" else "#2D6CDF"
        folium.Marker(
            [s["lat"],s["lon"]], tooltip=f"SOS · {s['need']} · {s['people']} people",
            icon=folium.DivIcon(html=f"<div style='background:{c};color:#fff;padding:4px 7px;border-radius:8px;font-size:10px;font-weight:900;border:1px solid white'>SOS</div>"),
        ).add_to(m)
    return m

# Theme-specific Saarthi logo assets.
# Dark mode uses the supplied neon-on-black logo; Light mode uses the supplied
# neon-on-white logo. Both are operator-provided assets and are switched only
# by the Saarthi Appearance control in the sidebar.
logo_path = BASE / "assets" / ("saarthi_logo_dark.png" if st.session_state.ui_theme == "Dark" else "saarthi_logo_light.png")
try:
    import base64
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
except Exception:
    logo_b64 = ""

# -----------------------------------------------------------------------------
# Management-only command access
# No public/general-user login is exposed. All operational and previously
# public-facing features are available to management personnel in one UI.
st.session_state.user_role = "Management Personnel"

# -----------------------------------------------------------------------------
# UI state and sidebar
# -----------------------------------------------------------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "location" not in st.session_state:
    st.session_state.location = {"name":"Srinagar", "display":"Srinagar, Jammu and Kashmir, India", "lat":34.0837, "lon":74.7973, "source":"default"}
if "weather_source" not in st.session_state:
    st.session_state.weather_source = "Open-Meteo"
if "weather" not in st.session_state:
    st.session_state.weather = fetch_weather(st.session_state.location["lat"], st.session_state.location["lon"], st.session_state.weather_source)
for _k,_v in {"rainfall_multiplier":1.0,"discharge_delta":0,"soil_delta":0,"sos_boost":0,"demand_multiplier":1.0,"capacity_shock":0,"fund_limit":250000}.items():
    st.session_state.setdefault(_k,_v)

with st.sidebar:
    st.markdown(f"<div style='padding:2px 0 4px;text-align:center'><img src='data:image/png;base64,{logo_b64}' style='width:164px;height:156px;object-fit:contain;display:block;margin:0 auto'><div class='nav-caption'>NATIONAL DISASTER GRID</div></div>",unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='nav-title'>India location</div>",unsafe_allow_html=True)
    with st.form("location_form",clear_on_submit=False):
        place_input=st.text_input("City / place in India",value=st.session_state.location["name"],placeholder="e.g. Mumbai, Srinagar, Guwahati",label_visibility="collapsed")
        locate=st.form_submit_button("Analyze this location",use_container_width=True,type="primary")
    if locate:
        found=geocode_india(place_input)
        if found:
            st.session_state.location=found; st.session_state.weather=fetch_weather(found["lat"],found["lon"],st.session_state.weather_source); st.session_state.last_refresh=datetime.now(); st.rerun()
        else: st.error("Location not found. Try an Indian city, district or locality name.")
    st.caption(f"Current: {st.session_state.location['display']}")
    st.divider()
    st.markdown("<div class='nav-title'>Appearance</div>",unsafe_allow_html=True)
    selected_theme = st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.ui_theme == "Dark" else 1,
        label_visibility="collapsed",
    )
    if selected_theme != st.session_state.ui_theme:
        st.session_state.ui_theme = selected_theme
        st.rerun()
    st.divider()
    st.markdown("<div class='nav-title'>Command centre</div>",unsafe_allow_html=True)
    opts=["Overview","Early Warning","Vulnerability","Resources","Response Optimizer","Scenario Testing","Supply Ledger","Field SOS","Prediction Lab"]
    current=st.session_state.get("current_page",opts[0]); current=current if current in opts else opts[0]
    page=st.radio("Navigation",opts,index=opts.index(current),label_visibility="collapsed"); st.session_state.current_page=page
    st.divider()
    st.markdown("<div class='nav-title'>Weather data source</div>",unsafe_allow_html=True)
    weather_source="Open-Meteo"
    st.caption("Live weather feed: Open-Meteo")
    if weather_source!=st.session_state.weather_source:
        st.session_state.weather_source=weather_source; st.session_state.weather=fetch_weather(st.session_state.location["lat"],st.session_state.location["lon"],"Open-Meteo"); st.rerun()
    auto=st.toggle("Live auto-refresh",value=True)
    interval=st.select_slider("Refresh interval",options=[15,30,60,120],value=30,format_func=lambda x:f"{x} seconds")
    if st.button("Refresh data now",use_container_width=True):
        st.session_state.weather=fetch_weather(st.session_state.location["lat"],st.session_state.location["lon"],st.session_state.weather_source); st.session_state.last_refresh=datetime.now(); st.session_state.refresh_count+=1; st.rerun()
    st.divider()
    st.markdown("<div class='nav-title'>Management personnel</div>",unsafe_allow_html=True)
    st.caption("Full operational access enabled.")
if auto:
    st_autorefresh(interval=interval * 1000, key="saarthi_live_refresh")

LOCATION = st.session_state.location
weather = st.session_state.weather
zones, hubs, sos = generate_city_state(LOCATION, weather)
rows = compute_zones(zones, st.session_state.rainfall_multiplier, st.session_state.discharge_delta, st.session_state.soil_delta, st.session_state.sos_boost)
df = pd.DataFrame(rows)
alloc_df = allocations(rows, hubs)
hospitals = generate_hospitals(LOCATION, df)
recommendation, recommendation_actions, recommendation_target, recommendation_hospital, recommendation_source = build_recommendation(df, hubs, hospitals)
movement_df = resource_movements(df, hubs, hospitals)
reason_list = decision_reasons(recommendation_target, recommendation_hospital, recommendation_source)
ambulances, response_hospitals, road_df = generate_response_assets(LOCATION, df)
response_plan, funds_remaining = optimize_response(df, hubs, ambulances, response_hospitals, road_df, fund_limit=float(st.session_state.fund_limit))

# Severity is shown through explicit status components; no page-wide gradient.
focus_options=["Auto · highest severity"]+df["name"].tolist()
with st.sidebar:
    st.divider(); st.markdown("<div class='nav-title'>Area focus</div>",unsafe_allow_html=True)
    focus_choice=st.selectbox("Focus area",focus_options,label_visibility="collapsed")
focus_row=df.iloc[0] if focus_choice=="Auto · highest severity" else df.loc[df["name"]==focus_choice].iloc[0]
focus_level=str(focus_row["level"])
focus_color={"CRITICAL":"#C83E3E","HIGH":"#F56616","MODERATE":"#F58A4B","LOW":"#2F8F5B"}.get(focus_level,C["accent"])

# Apply the live Open-Meteo forecast to the headline narrative without silently replacing
# the model's zone inputs until the operator explicitly accepts it in Prediction Lab.
live_rain = weather.get("rain_24h") if weather.get("ok") else None

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
# Compact theme-specific Saarthi mark supplied by the operator.
logo_path = BASE / "assets" / ("saarthi_logo_dark.png" if st.session_state.ui_theme == "Dark" else "saarthi_logo_light.png")
try:
    import base64
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
except Exception:
    logo_b64 = ""

st.markdown(
    f"""
<div class='hero'>
  <div class='brand'>
    <img src='data:image/png;base64,{logo_b64}' class='saarthi-logo' alt='Saarthi logo'>
    <div style='height:38px;border-left:1px solid var(--s-line);margin:0 4px'></div>
    <div style='flex:1'>
      <div class='eyebrow'>Disaster Intelligence Command Centre</div>
      <div class='brand-name' style='font-size:1.35rem'>COMMAND CENTRE</div>
      <div class='sub'>Predict 24h ahead · detect hazards · rank vulnerability · move resources before impact</div>
    </div>
    <div style='text-align:right'>
      <div class='live'><span class='pulse'></span> LIVE</div>
      <div class='small'>{st.session_state.last_refresh.strftime('%H:%M IST')} · Updated {st.session_state.last_refresh.strftime('%d %b %Y')}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(f"<div class='focus-ribbon' style='margin-top:10px;border-radius:12px;padding:9px 13px;font-size:.78rem;font-weight:700;'>FOCUS · {focus_row["name"]} · {focus_level} · vulnerability {focus_row["vulnerability"]:.2f} · severity {focus_row["severity_score"]:.1f}</div>", unsafe_allow_html=True)

if live_rain is not None:
    st.markdown(f"<div class='success-strip' style='margin-top:12px'>● <b>{weather.get('source','Weather feed')} connected</b> · {LOCATION['name']} 24h forecast precipitation: <b>{live_rain:.1f} mm</b> · precipitation probability peak: <b>{weather.get('max_pop',0):.0f}%</b> · {weather.get('updated','')}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='warning-strip' style='margin-top:12px'>⚠ <b>Weather feed fallback</b> · {weather.get("error", "Configure OPEN_METEO_API_KEY in .streamlit/secrets.toml if using a customer key, or use the Open-Meteo public endpoint.")}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='source-pill'>📍 {LOCATION['display']}</div><div class='source-pill'>Lat {LOCATION['lat']:.4f}</div><div class='source-pill'>Lon {LOCATION['lon']:.4f}</div>", unsafe_allow_html=True)

# KPIs
critical = int((df.level=="CRITICAL").sum())
high = int((df.level=="HIGH").sum())
alerts = int((df.probability>=50).sum())
open_sos = sum(x["people"] for x in sos if x["status"]=="OPEN")

st.markdown(f"<div class='section-head'><div><div class='section-title'>Situation at a glance</div><div class='section-note'>{LOCATION['name']} · six operational sectors · 24-hour prediction horizon</div></div></div>", unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Critical zones", critical)
k2.metric("High zones", high)
k3.metric("24h risk alerts", alerts)
k4.metric("People in open SOS", f"{open_sos:,}")
k5.metric("Average flood risk", f"{df.probability.mean():.1f}%")

# -----------------------------------------------------------------------------
# Overview page
# -----------------------------------------------------------------------------
if page == "Overview":
    # -------------------------------------------------------------------------
    # Command-officer overview: Detect -> Decide -> Allocate -> Track
    # -------------------------------------------------------------------------
    status = str(recommendation_target["level"])
    status_color = {"CRITICAL":"#C83E3E", "HIGH":"#F56616", "MODERATE":"#F58A4B", "LOW":"#2F8F5B"}.get(status, C["accent"])
    status_bg = {"CRITICAL":"rgba(200,62,62,.12)", "HIGH":"rgba(245,102,22,.12)", "MODERATE":"rgba(245,138,75,.10)", "LOW":"rgba(47,143,91,.10)"}.get(status, "transparent")
    st.markdown(
        f"""<div class='threat-card' style='border-left:7px solid {status_color};background:linear-gradient(100deg,{status_bg},var(--s-panel) 58%);'>
        <div class='eyebrow'>CURRENT THREAT STATUS</div>
        <div class='threat-city'>🌧️ {LOCATION['name'].upper()} — {status} RISK</div>
        <div class='threat-main'>24-hour hazard probability <b>{recommendation_target['probability']:.0f}%</b> · vulnerability <b>{recommendation_target['vulnerability']:.2f}</b> · severity score <b>{recommendation_target['severity_score']:.1f}</b></div>
        <div class='threat-action'>Recommended action: <b>{recommendation}</b></div>
        <div class='small'>Last checked: {st.session_state.last_refresh.strftime('%I:%M %p')}</div>
        </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Threat level", status)
    k1.caption(f"{recommendation_target['probability']:.0f}% 24h probability")
    k2.metric("Active hubs", len([h for h in hubs if h.get('open')]))
    k2.caption(f"{sum(h['medical']+h['oxygen'] for h in hubs):,} medical/oxygen units")
    k3.metric("Resources available", f"{sum(h['food']+h['medical']+h['water']+h['tarps']+h['oxygen'] for h in hubs):,}")
    k3.caption("Current hub inventory")
    k4.metric("Hospitals monitored", len(hospitals))
    k4.caption(f"{int(hospitals['ICU free'].sum())} ICU beds currently free")

    st.markdown("<div class='section-head'><div><div class='section-title'>⚡ Saarthi recommendation</div><div class='section-note'>Decision support generated from threat, vulnerability, SOS and available resources</div></div></div>", unsafe_allow_html=True)
    rec_col, why_col = st.columns([1.35,1], gap="large")
    with rec_col:
        st.markdown(f"<div class='recommend-card'><div class='recommend-title'>{recommendation}</div><div class='small'>Immediate operational actions</div>", unsafe_allow_html=True)
        for i, action in enumerate(recommendation_actions, 1):
            st.markdown(f"<div class='action-row'><span class='action-num'>{i}</span><span>{action}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with why_col:
        st.markdown("<div class='recommend-card'><div class='recommend-title'>🧠 Why Saarthi made this decision</div><div class='small'>Explainable allocation logic</div>", unsafe_allow_html=True)
        for reason in reason_list:
            st.markdown(f"<div class='reason-row'>✓ {reason}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'><div><div class='section-title'>🏥 Hospital + 📦 hub readiness</div><div class='section-note'>Operational capacity and stock are shown together so shortages are visible before allocation</div></div></div>", unsafe_allow_html=True)
    hcol, hubcol = st.columns([1.1,1.25], gap="large")
    with hcol:
        hospital_view = hospitals[["Hospital","Beds","ICU free","Oxygen","Status"]].copy()
        hospital_view["Status"] = hospital_view["Status"].map({"READY":"🟢 READY","ATTENTION":"🟠 ATTENTION","CRITICAL":"🔴 CRITICAL"})
        show_table(hospital_view)
    with hubcol:
        stock = pd.DataFrame(hubs)[["name","food","medical","water","tarps","oxygen"]].copy()
        stock.columns=["Hub","Food kits","Medical kits","Water","Tarps","Oxygen"]
        stock["Status"] = stock.apply(lambda r: "🔴 Refill" if r["Oxygen"] < 650 or r["Medical kits"] < 300 else ("🟠 Watch" if r["Oxygen"] < 800 else "🟢 Ready"), axis=1)
        show_table(stock)

    st.markdown("<div class='section-head'><div><div class='section-title'>🚚 Resource movement</div><div class='section-note'>Recommended movement can be replaced with live routing when transport feeds are connected</div></div></div>", unsafe_allow_html=True)
    movement_total_oxygen = max(1, sum(int(h.get('oxygen', 0)) for h in hubs))
    movement_total_kits = max(1, sum(int(h.get('medical', 0)) for h in hubs))
    for _, mv in movement_df.iterrows():
        resource_key = 'oxygen' if mv['Resource'] == 'Oxygen cylinders' else 'medical'
        available = movement_total_oxygen if resource_key == 'oxygen' else movement_total_kits
        pct = min(100, max(0, int(round((float(mv['Qty']) / available) * 100))))
        fill_cls = 'info' if resource_key == 'oxygen' else 'stable'
        icon = '🫁' if resource_key == 'oxygen' else '🩺'
        st.markdown(f"<div class='movement-card'><div class='movement-line'><b>{mv['From']}</b><span class='movement-arrow'>──── 🚚 ────▶</span><b>{mv['To']}</b></div><div class='small'>{icon} {mv['Qty']} {mv['Resource']} · ETA {mv['ETA']} · <b>{mv['Status']}</b></div><div class='movement-progress'><div class='movement-progress-head'><span>{mv['Resource']} movement</span><span>{pct}% of current hub stock</span></div><div class='movement-progress-track'><div class='movement-progress-fill {fill_cls}' style='width:{pct}%'></div></div></div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'><div><div class='section-title'>🧠 Dynamic response allocation</div><div class='section-note'>Constrained by urgency, ambulance capability, hospital capacity, road condition and available funds</div></div></div>", unsafe_allow_html=True)
    show_table(response_plan, level_column="Priority")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Ambulances available", int((ambulances["Status"].isin(["AVAILABLE","STAGED"])).sum()))
    rc2.metric("Blocked / slow roads", int((road_df["Status"]!="OPEN").sum()))
    rc3.metric("Funds remaining", f"₹{funds_remaining:,.0f}")

    map_col, map_info = st.columns([1.35, 1], gap="large")
    with map_col:
        st.markdown("<div class='section-head'><div><div class='section-title'>🗺️ Operational map</div><div class='section-note'>Compact live view of sectors, hubs and SOS points</div></div></div>", unsafe_allow_html=True)
        mini_map = build_map(df, hubs, sos, basemap="OpenStreetMap", center=(LOCATION["lat"], LOCATION["lon"]))
        st.markdown("<div class='map-mini'>", unsafe_allow_html=True)
        st_folium(mini_map, width=None, height=310, returned_objects=[], key="overview_map")
        st.markdown("</div>", unsafe_allow_html=True)
    with map_info:
        st.markdown("<div class='section-head'><div><div class='section-title'>📐 Vulnerability breakdown</div><div class='section-note'>Transparent weighted index used by Saarthi</div></div></div>", unsafe_allow_html=True)
        vc = vulnerability_components(focus_row.to_dict()) if isinstance(focus_row, pd.Series) else {}
        labels = [("Population exposure", "population_exposure", 25), ("Critical facilities", "critical_facilities", 20), ("Accessibility", "accessibility", 20), ("Resource shortage", "resource_shortage", 15), ("SOS pressure", "sos_pressure", 10), ("Hazard exposure", "hazard_exposure", 10)]
        rows_html = "".join([f"<div class='reason-row'><b>{label}</b><span style='float:right'>{vc.get(key,0):.0f}/100 · {weight}%</span></div>" for label,key,weight in labels])
        st.markdown(f"<div class='card'><div class='kpi-label'>Vulnerability index</div><div class='kpi-value'>{focus_row['vulnerability']*100:.0f}/100</div>{rows_html}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'><div><div class='section-title'>📋 Command priority queue</div><div class='section-note'>Highest operational severity first</div></div></div>", unsafe_allow_html=True)
    qcols = st.columns(2)
    for idx, (_, r) in enumerate(df.head(6).iterrows(), 1):
        with qcols[(idx-1)%2]:
            cls=r.level.lower()
            st.markdown(f"<div class='queue-item'><div class='queue-row'><div class='queue-number'>{idx}</div><span class='badge {cls}'>{r.level}</span><b style='margin-left:auto'>{r.severity_score:.1f}</b></div><div class='queue-name'>{r['name']}</div><div class='small'>24h risk <b>{r.probability}%</b> · vulnerability <b>{r.vulnerability:.2f}</b> · SOS {r.open_sos}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-head'><div><div class='section-title'>Risk trajectory</div><div class='section-note'>Current model probabilities by operational sector</div></div></div>", unsafe_allow_html=True)
    chart = px.bar(df.sort_values("probability"), x="probability", y="name", orientation="h", color="level", color_discrete_map={"CRITICAL":C["critical"],"HIGH":C["high"],"MODERATE":C["moderate"],"LOW":C["low"]}, text="probability")
    chart.update_traces(texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
    chart.update_layout(height=410, margin=dict(l=145,r=55,t=18,b=45), paper_bgcolor=C["panel"], plot_bgcolor=C["panel"], font=dict(color=C["text"], family="DM Sans", size=12), legend_title_text="", xaxis=dict(title="24h hazard probability (%)", showgrid=True, gridcolor=C["line"], zeroline=False, tickfont=dict(color=C["text"]), title_font=dict(color=C["text"])), yaxis=dict(title="", showgrid=False, tickfont=dict(color=C["text"], size=12), automargin=True))
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-head'><div><div class='section-title'>System audit log</div><div class='section-note'>Every recommendation can be traced back to the current scenario state</div></div></div>", unsafe_allow_html=True)
    audit_rows = [{"Time":datetime.now().strftime('%H:%M:%S'),"Event":"ALLOCATION RECOMMENDED","Target":recommendation_target["name"],"Decision":recommendation,"Severity":status},{"Time":datetime.now().strftime('%H:%M:%S'),"Event":"HOSPITAL CHECK","Target":recommendation_hospital["Hospital"],"Decision":f"{recommendation_hospital['ICU free']} ICU free / {recommendation_hospital['Oxygen']} oxygen","Severity":recommendation_hospital["Status"]}]
    show_table(pd.DataFrame(audit_rows))

elif page == "Early Warning":
    st.markdown("<div class='section-head'><div><div class='section-title'>Early warning console</div><div class='section-note'>24-hour flood prediction + location weather intelligence</div></div></div>", unsafe_allow_html=True)
    a,b = st.columns([1.5,1], gap="large")
    with a:
        fig = px.bar(df.sort_values("probability"), x="probability", y="name", orientation="h", color="probability", color_continuous_scale=[[0,C["low"]],[.5,C["moderate"]],[1,C["critical"]]])
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=C["text"], coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown("<div class='card'><div class='section-title' style='font-size:1.05rem'>Weather feed</div>", unsafe_allow_html=True)
        if weather.get("ok"):
            st.metric("24h precipitation", f"{weather['rain_24h']:.1f} mm")
            st.metric("Peak precipitation probability", f"{weather['max_pop']:.0f}%")
            st.caption(f"Temperature {weather.get('temperature','—')} · Humidity {weather.get('humidity','—')} · Wind {weather.get('wind','—')} · Pressure {weather.get('pressure','—')}")
            st.markdown(f"<span class='source-pill'>{weather.get('source','Live weather')}</span><span class='source-pill'>24h hourly forecast</span>", unsafe_allow_html=True)
        else:
            st.info("Live weather is not configured; the model continues using its demo scenario inputs.")
        st.markdown("</div>", unsafe_allow_html=True)
    alerts_df = df[["name","probability","level","severity_score","open_sos"]].copy()
    alerts_df.columns=["Zone","24h probability %","Severity","Score","Open SOS"]
    show_table(alerts_df, level_column="Severity")
    st.markdown("<div class='warning-strip'>Cyclone detection is retained as a regional hazard module. the selected location's primary local pathway is represented by the flood/flash-flood model; cyclone signals should not be interpreted as direct local landfall risk.</div>", unsafe_allow_html=True)

elif page == "Vulnerability":
    st.markdown("<div class='section-head'><div><div class='section-title'>Severity & vulnerability</div><div class='section-note'>Where conditions are most severe and where exposure is highest</div></div></div>", unsafe_allow_html=True)
    view=df[["name","level","severity_score","probability","vulnerability","population","open_sos","population_exposure","critical_facilities","accessibility","resource_shortage","sos_pressure","hazard_exposure"]].copy()
    view.columns=["Zone","Level","Severity score","Flood probability %","Vulnerability","Population","Open SOS","Population exposure","Critical facilities","Accessibility","Resource shortage","SOS pressure","Hazard exposure"]
    show_table(view, level_column="Level")
    fig=px.scatter(df, x="vulnerability", y="probability", size="population", color="level", hover_name="name", text="name", color_discrete_map={"CRITICAL":C["critical"],"HIGH":C["high"],"MODERATE":C["moderate"],"LOW":C["low"]})
    fig.update_traces(textposition="top center")
    fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=C["text"], legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='card'><b>Prioritization formula</b><br><span class='small'>Severity = 24h hazard probability × vulnerability + open-SOS pressure. Critical and High zones are surfaced first for resource planning.</span></div>", unsafe_allow_html=True)

elif page == "Resources":
    st.markdown("<div class='section-head'><div><div class='section-title'>Resource allocation</div><div class='section-note'>Move supplies before impact; recommendations change as severity shifts</div></div></div>", unsafe_allow_html=True)
    left,right=st.columns([1.45,1],gap="large")
    with left:
        st.markdown("<div class='card'><b>Recommended pre-positioning</b><div class='small'>Priority is driven by severity, population and current hub stock.</div></div>", unsafe_allow_html=True)
        show_table(alloc_df.sort_values(["Score","Priority"],ascending=[False,True]), level_column="Priority")
    with right:
        st.markdown("<div class='card'><b>Hub stock</b></div>", unsafe_allow_html=True)
        stock=pd.DataFrame(hubs)[["name","food","medical","water","tarps","oxygen"]]
        stock.columns=["Hub","Food","Medical","Water","Tarps","Oxygen"]
        show_table(stock)
    st.markdown("<div class='success-strip'>Re-optimization is live. Increase rainfall, river discharge or SOS pressure in the sidebar and the priority list will recalculate.</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'><div><div class='section-title'>Transport constraints</div><div class='section-note'>Road conditions directly affect ambulance and supply allocation.</div></div></div>", unsafe_allow_html=True)
    show_table(road_df)

elif page == "Scenario Testing":
    st.markdown("<div class='section-head'><div><div class='section-title'>Scenario testing</div><div class='section-note'>Stress-test changing disaster conditions before committing an operational decision.</div></div></div>",unsafe_allow_html=True)
    presets={"Baseline":(1.0,0,0,0,1.0,0,250000),"Heavy rainfall":(1.5,10,8,1,1.15,10,250000),"Flash-flood SOS wave":(1.35,20,10,4,1.35,10,300000),"Road disruption":(1.2,5,5,1,1.1,5,250000),"Hospital capacity shock":(1.15,8,5,2,1.2,45,250000),"Custom":(st.session_state.rainfall_multiplier,st.session_state.discharge_delta,st.session_state.soil_delta,st.session_state.sos_boost,st.session_state.demand_multiplier,st.session_state.capacity_shock,st.session_state.fund_limit)}
    preset=st.selectbox("Scenario preset",list(presets))
    vals=presets[preset]
    a,b,c=st.columns(3)
    with a:
        rain=st.slider("Rainfall stress",0.5,2.0,float(vals[0]),0.05); river=st.slider("River discharge change",-20,30,int(vals[1]),1); soil=st.slider("Soil saturation change",-15,15,int(vals[2]),1)
    with b:
        sosb=st.slider("New SOS pressure",0,5,int(vals[3]),1); demand=st.slider("Demand surge",0.5,2.0,float(vals[4]),0.05); cap=st.slider("Hospital capacity loss",0,60,int(vals[5]),5)
    with c:
        funds=st.number_input("Available response funds (₹)",25000,5000000,int(vals[6]),25000); run=st.button("Run scenario",use_container_width=True,type="primary")
    if run or preset=="Custom":
        st.session_state.rainfall_multiplier=float(rain); st.session_state.discharge_delta=int(river); st.session_state.soil_delta=int(soil); st.session_state.sos_boost=int(sosb); st.session_state.demand_multiplier=float(demand); st.session_state.capacity_shock=int(cap); st.session_state.fund_limit=float(funds)
    scenario_rows=compute_zones(zones,float(rain),int(river),int(soil),int(sosb)); sim_df=pd.DataFrame(scenario_rows); sim_df["population"]=(sim_df["population"]*float(demand)).round().astype(int)
    sim_amb,sim_hosp,sim_roads=generate_response_assets(LOCATION,sim_df); sim_hosp["Beds free"]=(sim_hosp["Beds free"]*(1-float(cap)/100)).round().astype(int); sim_hosp["ICU free"]=(sim_hosp["ICU free"]*(1-float(cap)/100)).round().astype(int)
    sim_plan,sim_remaining=optimize_response(sim_df,hubs,sim_amb,sim_hosp,sim_roads,float(funds))
    base_top=df.iloc[0]; sim_top=sim_df.iloc[0]
    k1,k2,k3,k4=st.columns(4); k1.metric("Top risk sector",sim_top["name"],f"{sim_top['severity_score']-base_top['severity_score']:+.1f} score"); k2.metric("24h probability",f"{sim_top['probability']:.1f}%",f"{sim_top['probability']-base_top['probability']:+.1f} pp"); k3.metric("Allocated missions",len(sim_plan)); k4.metric("Funds remaining",f"₹{sim_remaining:,.0f}")
    st.markdown("### Scenario impact")
    compare=pd.DataFrame({"Zone":df["name"],"Baseline score":df["severity_score"],"Scenario score":sim_df["severity_score"]}); compare["Change"]=compare["Scenario score"]-compare["Baseline score"]; show_table(compare.sort_values("Change",ascending=False))
    st.markdown("### Recalculated allocation"); show_table(sim_plan,level_column="Priority")
    st.markdown(f"<div class='scenario-strip'><b>{preset}</b><br><span class='small'>Rain × {rain:.2f} · River {river:+d} · Soil {soil:+d} · SOS +{sosb} · Demand × {demand:.2f} · Hospital capacity loss {cap}% · Fund limit ₹{funds:,.0f}</span></div>",unsafe_allow_html=True)

elif page == "Response Optimizer":
    st.markdown("<div class='section-head'><div><div class='section-title'>Response optimizer</div><div class='section-note'>Scenario-based allocation across affected people, ambulances, hospitals, roads, scarce supplies and funds.</div></div></div>", unsafe_allow_html=True)
    f1,f2,f3=st.columns(3)
    with f1: fund_limit=st.number_input("Available response funds (₹)", min_value=25000, max_value=5000000, value=250000, step=25000)
    with f2: demand_multiplier=st.slider("Demand surge", 0.5, 2.0, 1.0, 0.05)
    with f3: capacity_shock=st.slider("Hospital capacity loss", 0, 60, 0, 5)
    sim_hosp=response_hospitals.copy()
    sim_hosp["Beds free"]=(sim_hosp["Beds free"]*(1-capacity_shock/100)).round().astype(int)
    sim_hosp["ICU free"]=(sim_hosp["ICU free"]*(1-capacity_shock/100)).round().astype(int)
    sim_df=df.copy(); sim_df["population"]=(sim_df["population"]*demand_multiplier).round().astype(int)
    sim_plan, sim_remaining=optimize_response(sim_df,hubs,ambulances,sim_hosp,road_df,float(fund_limit))
    a,b,c,d=st.columns(4)
    a.metric("Affected demand", f"{sim_df['population'].sum():,}")
    b.metric("Allocated missions", len(sim_plan))
    c.metric("Funds committed", f"₹{fund_limit-sim_remaining:,.0f}")
    d.metric("Uncommitted funds", f"₹{sim_remaining:,.0f}")
    st.markdown("### Allocation decisions")
    show_table(sim_plan, level_column="Priority")
    st.markdown("### Ambulance capability")
    show_table(ambulances[["Unit","Capability","Seats","Medical kits","Status"]])
    st.markdown("### Changing hospital capacity")
    show_table(sim_hosp[["Hospital","Beds","Occupancy %","Beds free","ICU total","ICU free","Oxygen %","Status"]])
    st.markdown("### Disrupted transport network")
    show_table(road_df)
    st.markdown("<div class='card'><b>Why this is not nearest-hospital / first-come-first-served</b><br><span class='small'>Each mission is scored using urgency, vulnerability, ambulance capability, hospital capacity, route penalty and fund availability. Re-running the scenario after rainfall, SOS, road or capacity changes can produce a different allocation.</span></div>", unsafe_allow_html=True)

elif page == "Supply Ledger":
    st.markdown("<div class='section-head'><div><div class='section-title'>Supply tracking, funds & audit</div><div class='section-note'>Append-only, chained SHA-256 records with role-based minimum-necessary disclosure.</div></div></div>", unsafe_allow_html=True)
    records=[]; prev="GENESIS"
    demo=[("SHP-0001","Water packs",900,"Bemina Warehouse","Rainawari","ALLOCATED",54000,"District Relief Fund"),("SHP-0001","Water packs",900,"Bemina Warehouse","Rainawari","DISPATCHED",54000,"District Relief Fund"),("SHP-0002","Medical kits",120,"Lal Chowk Relief Hub","Downtown Srinagar","ALLOCATED",72000,"Emergency Medical Fund")]
    for i,(sid,item,qty,src,dst,step,amount,fund) in enumerate(demo,1):
        payload=f"{sid}|{item}|{qty}|{src}|{dst}|{step}|{amount}|{fund}"
        fp=hashlib.sha256((prev+"|"+payload).encode()).hexdigest()
        records.append({"#":i,"Shipment":sid,"Step":step,"Item":item,"Qty":qty,"Fund":fund,"Amount ₹":amount,"Previous":prev[:12]+"…","Fingerprint":fp[:16]+"…"})
        prev=fp
    role=st.selectbox("Audit view",["Command Officer","Auditor","Field Responder","Public Summary"])
    ledger_df=pd.DataFrame(records)
    if role=="Public Summary":
        ledger_df=ledger_df[["#","Shipment","Step","Item","Qty","Fund","Fingerprint"]]
    elif role=="Field Responder":
        ledger_df=ledger_df[["#","Shipment","Step","Item","Qty","Fingerprint"]]
    show_table(ledger_df)
    c1,c2,c3=st.columns(3)
    c1.metric("Ledger status","VERIFIED")
    c2.metric("Records",len(records))
    c3.metric("Chain head",prev[:12]+"…")
    st.markdown("<div class='card'><b>Privacy control</b><br><span class='small'>Individual names, phone numbers, exact addresses and personal identifiers are excluded from operational views by default. Fund details are restricted to command/audit roles in this prototype. The SHA-256 chain makes subsequent tampering detectable; it is not a public blockchain.</span></div>", unsafe_allow_html=True)

elif page == "Field SOS":
    st.markdown("<div class='section-head'><div><div class='section-title'>Field requests</div><div class='section-note'>SOS pins, demand and operational status</div></div></div>", unsafe_allow_html=True)
    s_df=pd.DataFrame(sos)
    s_df.columns=["Zone","Need","People","Status","Latitude","Longitude"]
    show_table(s_df)
    st.markdown("<div class='card'><b>Operational state machine</b><br><span class='small'>OPEN → CLAIMED → IN TRANSIT → RESOLVED</span></div>", unsafe_allow_html=True)

elif page == "Prediction Lab":
    st.markdown("<div class='section-head'><div><div class='section-title'>Prediction lab</div><div class='section-note'>Stress-test the 24-hour model before committing an operational decision</div></div></div>", unsafe_allow_html=True)
    selected=st.selectbox("Select local sector",df.name.tolist())
    z=next(x for x in rows if x["name"]==selected)
    c1,c2,c3=st.columns(3)
    with c1: rain=st.slider("Next 24h forecast rain (mm)",0.0,300.0,float(z["features"]["forecast_rain_24h"]),1.0)
    with c2: river=st.slider("River discharge percentile",0.0,100.0,float(z["features"]["river_discharge_pct"]),1.0)
    with c3: soil=st.slider("Soil moisture (%)",0.0,100.0,float(z["features"]["soil_moisture_pct"]),1.0)
    f=dict(z["features"]); f["forecast_rain_24h"]=rain; f["river_discharge_pct"]=river; f["soil_moisture_pct"]=soil
    p=predict(f); score,level=severity(p,z["vulnerability"],z["open_sos"])
    a,b,c=st.columns(3)
    a.metric("24h flood probability",f"{p:.1f}%")
    b.metric("Severity",level)
    c.metric("Severity score",f"{score:.1f}")
    st.progress(min(1,p/100))
    if weather.get("ok"):
        st.markdown(f"<div class='success-strip'>{weather.get('source','Live weather')} forecast available: {weather['rain_24h']:.1f} mm expected over the next 24h at the selected location reference point. Use the slider above to test a zone-specific stress case.</div>", unsafe_allow_html=True)
    st.markdown("### Model inputs")
    st.json({"rain_24h_forecast_mm":round(rain,1),"river_discharge_percentile":round(river,1),"soil_moisture_pct":round(soil,1),"vulnerability":z["vulnerability"],"open_sos":z["open_sos"]})

st.divider()
st.caption("SAARTHI · hackathon prototype · India-wide location focus · Open-Meteo · dynamic response optimizer · constrained resource/fund allocation · tamper-evident audit · Not for operational emergency decisions")
