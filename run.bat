@echo off
rem Start HolOrama from a source install (run install.ps1 first).
cd /d "%~dp0"
".venv\Scripts\python.exe" src\main.py
if errorlevel 1 pause
