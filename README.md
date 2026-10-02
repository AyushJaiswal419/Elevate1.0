# SAARTHI — India-wide Disaster Command Center

Streamlit hackathon prototype for location-based disaster intelligence across India.
Made by the team CODE BLOODED by
->Ayush Jaiswal (AyushJaiswal419),
->Madhuraa Chavan (maddyy2410),
->Achintya Singh, 
->Bhuvi Bhimani,

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

## Check any place in India
Use **India location** in the left sidebar, enter a city/place (for example Mumbai, Guwahati, Chennai, Srinagar), and click **Analyze this location**.

SAARTHI geocodes the location, fetches the 24-hour forecast for those coordinates through the configured Open-Meteo credentials, and rebuilds the dashboard around six local operational sectors. The dashboard then recalculates 24h flood probability, vulnerability, severity, priority queue, map, SOS and resource recommendations.

### Live weather configuration
Create `.streamlit/secrets.toml`:
```toml
OPEN_METEO_API_KEY = "YOUR_KEY"
```

A demo fallback is used if live credentials are unavailable. The city-wide probability is a **prototype estimate** based on live forecast precipitation plus coarse city-level baseline assumptions; it is not an official emergency forecast.

## Tests
```bash
pytest -q
```
