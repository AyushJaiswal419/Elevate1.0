@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run START_SAARTHI.bat first.
  pause
  exit /b 1
)
echo Updating Saarthi packages...
".venv\Scripts\python.exe" -m pip install -r requirements-streamlit.txt
if errorlevel 1 pause
else echo Update complete. You can now double-click START_SAARTHI.bat
pause
endlocal
