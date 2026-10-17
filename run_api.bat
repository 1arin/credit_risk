@echo off
echo =====================================================================
echo  Starting AI-Driven Credit Risk FastAPI Backend...
echo  Documentation: http://localhost:8000/docs
echo =====================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
