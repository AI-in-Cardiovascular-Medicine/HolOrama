@'
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python src\main.py
pause
'@ | Set-Content -Path run.bat -Encoding ascii