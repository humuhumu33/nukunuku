@echo off
echo Starting Packing Demo...
echo.

echo Starting backend server...
start "Packing Demo Backend" cmd /k "cd backend && python app.py"

timeout /t 3 /nobreak >nul

echo Starting frontend server...
start "Packing Demo Frontend" cmd /k "cd examples\apps\packing-demo && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5001
echo Frontend: http://localhost:3001
echo.
pause

