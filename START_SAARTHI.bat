@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title SAARTHI - Disaster Intelligence Command Centre

echo.
echo ============================================================
echo   SAARTHI - INDIA DISASTER INTELLIGENCE COMMAND CENTRE
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] First-time setup: creating Python environment...
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
    if errorlevel 1 (
        echo.
        echo ERROR: Python was not found. Install Python 3.11+ and run this file again.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment could not be created.
    pause
    exit /b 1
)

if not exist ".venv\.saarthi_deps_ready" (
    echo [2/3] Installing Saarthi packages. This happens only once...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements-streamlit.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    type nul > ".venv\.saarthi_deps_ready"
) else (
    echo [2/3] Saarthi packages already installed.
)

if not exist ".streamlit\secrets.toml" (
    echo.
    echo [3/3] Weather API setup
    echo Keys are stored ONLY on this computer in:
    echo   .streamlit\secrets.toml
    echo.
    set /p "OPEN_METEO_KEY=Paste your Open-Meteo customer API key (leave blank to use the public endpoint): "
    > ".streamlit\secrets.toml" echo OPEN_METEO_API_KEY = "%OPEN_METEO_KEY%"
    echo Weather API configuration saved locally.
    echo Open-Meteo works without a key on its public endpoint.
) else (
    echo [3/3] Weather API credentials already configured.
)

echo.
echo Starting Saarthi...
echo The browser will open automatically.
echo Close this window to stop Saarthi.
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py

pause
endlocal
