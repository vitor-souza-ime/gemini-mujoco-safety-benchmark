@echo off
cd /d "%~dp0"
echo Retomando V13. Cenas completas serao puladas.
echo Se uma resposta Gemini ja foi recebida antes de uma queda, ela sera reutilizada sem nova chamada.
python run.py gemini --start-scene 21 --scenes 50 --output results\gemini_50_v13 --resume --max-calls 50
pause
