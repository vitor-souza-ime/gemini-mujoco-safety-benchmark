@echo off
cd /d "%~dp0"
echo ATENCAO: este teste consome 1 chamada e NAO e necessario antes do lote cientifico.
set /p go=Executar 1 chamada de probe separada [S,N]?
if /I not "%go%"=="S" exit /b
python run.py gemini --start-scene 21 --scenes 1 --output results\probe_v13 --max-calls 1
pause
