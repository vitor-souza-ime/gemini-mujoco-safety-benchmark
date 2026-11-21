@echo off
cd /d "%~dp0"
python run.py local --start-scene 21 --scenes 2 --output results\local_v13_smoke
pause
