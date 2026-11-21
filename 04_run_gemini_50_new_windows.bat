@echo off
cd /d "%~dp0"
echo ================================================================
echo GEMINI ROBOTICS ER 2 + MUJOCO - V13 - 50 NOVAS CENAS
echo IDs 21..70 = 50 respostas Gemini validas alvo
echo 4 metodos = 200 execucoes MuJoCo neste lote
echo Checkpoint por cena + resposta API salva ANTES da simulacao.
echo Se parar, use 05_resume_gemini_50_windows.bat.
echo ================================================================
set /p go=Iniciar/continuar novo lote de 50 cenas [S,N]?
if /I not "%go%"=="S" exit /b
python run.py gemini --start-scene 21 --scenes 50 --output results\gemini_50_v13 --resume --max-calls 50
pause
