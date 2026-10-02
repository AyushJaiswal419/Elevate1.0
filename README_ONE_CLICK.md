# Saarthi — One-click Windows launcher

## First launch
Double-click **START_SAARTHI.bat**.

It will automatically:
1. Create the `.venv` Python environment if needed.
2. Install the required packages if needed.
3. Ask for an **Open-Meteo customer API key** (optional).
4. Save the keys locally in `.streamlit/secrets.toml`.
5. Start the Streamlit command centre and open it in your browser.

Open-Meteo's public forecast endpoint works without a key, so you can leave the Open-Meteo key blank for evaluation/prototyping. Open-Meteo customer keys use the dedicated `customer-api.open-meteo.com` endpoint. The app uses the public endpoint automatically when no customer key is configured. See the official Open-Meteo documentation for current API details.

## Weather source selector
In the Saarthi sidebar, the order is:
1. **Open-Meteo**

Open-Meteo is the sole live weather source used by Saarthi.

## Open-Meteo fields used by Saarthi
The integration requests a 24-hour hourly forecast for:
- precipitation
- precipitation probability
- temperature at 2 m
- relative humidity at 2 m
- wind speed at 10 m
- surface pressure

The resulting 24-hour precipitation and peak precipitation probability feed the existing Saarthi hazard/risk pipeline for the selected location.

## Day Mode tables
All Streamlit data tables now use explicit theme-aware styling. Day Mode forces light table backgrounds, readable dark text, visible borders, and light headers rather than inheriting the dark browser/Streamlit theme.

## Sidebar
Sidebar navigation and action controls are text-only; the decorative button icons/images were removed.

## Every later launch
Just double-click **START_SAARTHI.bat**. No Command Prompt commands are required.

## Update dependencies
Double-click **UPDATE_SAARTHI.bat** if you want to refresh installed Python packages.

## Security
Do not upload `.streamlit/secrets.toml` to GitHub or share the ZIP after entering real API keys. The provided `.gitignore` excludes it.
