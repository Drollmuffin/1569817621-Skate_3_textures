@echo off
rem Double-click: converts everything in the "input" folder into "output".
rem Drag and drop pictures (or .psg files) onto this file to convert just those.
cd /d "%~dp0"
python -c "import PIL, numpy" 2>nul || python -m pip install -r requirements.txt
python skate3tex.py %*
pause
