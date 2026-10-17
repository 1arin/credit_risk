@echo off
echo =====================================================================
echo  Starting AI-Driven Credit Risk Underwriting Portal...
echo =====================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

streamlit run app.py
pause
