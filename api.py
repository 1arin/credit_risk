import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any

from src.utils import calculate_risk_tier
from src.database import save_assessment, get_recent_assessments, get_portfolio_kpis
from src.counterfactual import simulate_what_if
from src.explainability import generate_plain_text_explanation, get_thai_feature_name

app = FastAPI(
    title="AI Credit Risk Prediction API Service",
    description="Backend Scoring Engine & Explainable AI REST Service for Credit Risk Assessment",
    version="2.0.0"
)

# Enable CORS for React Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cached artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
champion_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.joblib"))

with open(os.path.join(MODELS_DIR, "evaluation_results.json"), "r", encoding="utf-8") as f:
    eval_data = json.load(f)

feature_names = eval_data["feature_names"]

def normalize_intent(intent: str) -> str:
    i = str(intent).upper()
    if 'DEBT' in i or 'รวมหนี้' in i:
        return 'DEBTCONSOLIDATION'
    elif 'EDU' in i or 'ศึกษา' in i:
        return 'EDUCATION'
    elif 'VENTURE' in i or 'ธุรกิจ' in i:
        return 'VENTURE'
    elif 'HOME' in i or 'บ้าน' in i or 'อยู่อาศัย' in i:
        return 'HOMEIMPROVEMENT'
    elif 'MED' in i or 'แพทย์' in i or 'รักษา' in i:
        return 'MEDICAL'
    return 'PERSONAL'

class ApplicantInput(BaseModel):
    applicant_name: Optional[str] = "Anonymous"
    officer_name: Optional[str] = "Credit Officer"
    person_age: int = Field(..., ge=18, le=100, example=28)
    person_income: float = Field(..., gt=0, example=55000.0)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: float = Field(..., ge=0, le=50, example=4.0)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    loan_amnt: float = Field(..., gt=0, example=12000.0)
    loan_int_rate: float = Field(..., ge=1.0, le=40.0, example=11.5)
    loan_percent_income: Optional[float] = None
    cb_person_default_on_file: str = Field(..., example="N")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=40, example=4)
    down_payment: Optional[float] = 0.0
    officer_notes: Optional[str] = ""

@app.get("/")
def root():
    return {
        "system": "AI-Driven Credit Risk Prediction Backend Service",
        "status": "online",
        "champion_model": "XGBoost",
        "champion_auc": eval_data["summary"]["champion_auc"],
        "docs_url": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    return {
        "metrics": eval_data["metrics"],
        "threshold_analysis": eval_data["threshold_analysis"]
    }

@app.post("/predict")
def predict_credit_risk(applicant: ApplicantInput):
    data_dict = applicant.dict()
    
    # Net loan after down payment if applicable
    down_pay = float(data_dict.get("down_payment") or 0.0)
    if down_pay > 0:
        data_dict["loan_amnt"] = max(100.0, data_dict["loan_amnt"] - down_pay)
        
    data_dict["loan_intent"] = normalize_intent(data_dict["loan_intent"])
    
    # Auto compute DTI if omitted
    if not data_dict.get("loan_percent_income"):
        data_dict["loan_percent_income"] = round(data_dict["loan_amnt"] / max(data_dict["person_income"], 1.0), 3)
        
    df_single = pd.DataFrame([data_dict])
    X_trans = preprocessor.transform(df_single)
    
    prob = float(champion_model.predict_proba(X_trans)[0, 1])
    risk_info = calculate_risk_tier(prob)
    
    # SHAP local explanation
    shap_vals = explainer(X_trans)
    shap_1d = shap_vals.values[0]
    plain_exp = generate_plain_text_explanation(shap_1d, feature_names, data_dict)
    
    # Structured SHAP list for React UI
    df_shap = pd.DataFrame({'feature': feature_names, 'value': shap_1d})
    df_shap['abs_val'] = df_shap['value'].abs()
    top_shap = df_shap.sort_values(by='abs_val', ascending=False).head(6)
    shap_items = [
        {"label": get_thai_feature_name(r['feature']), "value": round(float(r['value']), 2)}
        for _, r in top_shap.iterrows()
    ]
    
    # Save to SQLite audit log
    record_payload = data_dict.copy()
    record_payload["risk_score"] = risk_info["score"]
    record_payload["risk_tier"] = risk_info["tier"]
    record_payload["probability"] = prob
    record_payload["underwriting_decision"] = "Approved" if risk_info["score"] <= 30 else ("Under Review" if risk_info["score"] <= 70 else "Rejected")
    assessment_id = save_assessment(record_payload)
    
    return {
        "assessment_id": assessment_id,
        "risk_score": risk_info["score"],
        "risk_tier": risk_info["tier"],
        "probability": prob,
        "risk_evaluation": risk_info,
        "shap_summary": shap_items,
        "text_summary": {
            "top_risk_drivers": plain_exp["risk_drivers"],
            "top_protective_factors": plain_exp["protective_factors"]
        }
    }

@app.post("/what-if")
def what_if_analysis(applicant: ApplicantInput):
    data_dict = applicant.dict()
    down_pay = float(data_dict.get("down_payment") or 0.0)
    if down_pay > 0:
        data_dict["loan_amnt"] = max(100.0, data_dict["loan_amnt"] - down_pay)
    data_dict["loan_intent"] = normalize_intent(data_dict["loan_intent"])
    if not data_dict.get("loan_percent_income"):
        data_dict["loan_percent_income"] = round(data_dict["loan_amnt"] / max(data_dict["person_income"], 1.0), 3)
    return simulate_what_if(data_dict, preprocessor, champion_model)

@app.get("/assessments")
def list_assessments(limit: int = 50):
    df = get_recent_assessments(limit=limit)
    return df.to_dict(orient="records")

@app.get("/kpis")
def get_kpis():
    kpi_dict = get_portfolio_kpis()
    kpi_dict["total"] = kpi_dict.get("total_assessments", 1284)
    kpi_dict["average"] = kpi_dict.get("avg_risk_score", 28.5)
    kpi_dict["expected_loss_reduction"] = eval_data.get("threshold_analysis", {}).get("cost_savings_pct", -8.85)
    return kpi_dict
