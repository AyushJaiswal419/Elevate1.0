# SAARTHI — Streamlit Command Centre

Srinagar-focused disaster intelligence prototype.

## Run

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Open-Meteo

The app uses Open-Meteo for its live weather forecast.

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Put your customer API key in `OPEN_METEO_API_KEY` if you have one; the public endpoint works without a key.
3. Keep the key out of GitHub / source control.
4. Restart Streamlit.

Example:

```toml
OPEN_METEO_API_KEY = "YOUR_KEY"
```

If no key is configured, the dashboard uses its bundled demonstration data and clearly labels the weather feed as fallback.

## Map

The map is a Folium/Leaflet map centered on Srinagar. It has Street, Light and Dark base layers plus severity circles, zone markers, relief hubs, SOS markers, fullscreen and a minimap. If one tile provider is unavailable, switch layers using the map control.

## Theme

Use the **Dark / Light operational theme** button in the sidebar. The app also ships a Streamlit `config.toml` containing separate Dark / Light operational theme.

## Tests

```powershell
pytest -q
```

Expected bundled engine tests: `3 passed`.

## Important model note

The included Random Forest is a demonstration model trained on synthetic labels so the hackathon UI can run offline. It is not a validated operational flood-warning model. Replace the training dataset with historical Srinagar/Jammu & Kashmir flood/rainfall observations before making real-world claims.
