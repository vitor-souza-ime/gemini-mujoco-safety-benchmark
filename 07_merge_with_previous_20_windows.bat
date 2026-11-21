@echo off
cd /d "%~dp0"
echo Copie o resultado anterior para results\gemini_20_v11.
echo O lote novo deve estar em results\gemini_50_v13.
echo Sera criado results\gemini_70_combined com 70 cenas e novo relatorio.
python merge_20_plus_50.py
pause
