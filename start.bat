@echo off
REM ---------------------------------------------------------------
REM  Start the CCTV Search app: backend (FastAPI :8000) + frontend
REM  (Vite :5173), each in its own window. Close a window to stop it.
REM  Uses start's /D flag to set the working directory (no "cd &&",
REM  which cmd's start parser rejects).
REM ---------------------------------------------------------------

echo Starting backend  -^>  http://127.0.0.1:8000
start "CCTV Backend" /D "%~dp0backend" cmd /k "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo Starting frontend -^>  http://localhost:5173
start "CCTV Frontend" /D "%~dp0frontend" cmd /k npm run dev

echo.
echo Both servers are launching in separate windows.
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:5173
echo Close a server's window to stop it.
