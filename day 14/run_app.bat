@echo off
cd /d "%~dp0"
set PORT=5000
echo Preview link: http://localhost:5000/
start "" "http://localhost:5000/"
python app.py
pause
