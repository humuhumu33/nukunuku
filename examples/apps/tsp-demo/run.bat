@echo off
REM Run script for TSP Demo on Windows

echo Starting TSP Demo...

REM Start backend in background
echo Starting Python backend...
start /B python backend\app.py

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting React frontend...
npm run dev


