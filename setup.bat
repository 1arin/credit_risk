@echo off
echo =====================================================================
echo  AI-Driven Credit Risk Prediction & Underwriting System
echo  Setup Script (Windows)
echo =====================================================================
echo.

if not exist ".venv" (
    echo [1/3] Creating Python virtual environment in .venv...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment .venv already exists.
)

echo [2/3] Installing dependencies from requirements.txt...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Verifying pre-trained models and dataset...
if exist "models\best_model.joblib" (
    echo Pre-trained XGBoost Champion Model found.
) else (
    echo Pre-trained model not found. Training model now...
    python src\train.py
)

echo.
echo =====================================================================
echo  Setup Completed Successfully!
echo  To run the dashboard: double click 'run_app.bat' or run:
echo    streamlit run app.py
echo =====================================================================
pause
