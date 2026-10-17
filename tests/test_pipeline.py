import pytest
import os
import joblib
import pandas as pd
import numpy as np

from src.utils import calculate_risk_tier
from src.database import save_assessment, get_recent_assessments
from src.counterfactual import simulate_what_if
from src.stress_testing import run_portfolio_stress_test
from src.fairness import evaluate_model_fairness

@pytest.fixture
def artifacts():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    p = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    m = joblib.load(os.path.join(models_dir, "best_model.joblib"))
    sample_df = pd.read_csv(os.path.join(models_dir, "sample_test_cases.csv"))
    return {"preprocessor": p, "model": m, "sample_df": sample_df}

def test_risk_tier_classification():
    low = calculate_risk_tier(0.15)
    assert low["tier"] == "Low Risk"
    assert low["score"] == 15.0
    
    med = calculate_risk_tier(0.45)
    assert med["tier"] == "Medium Risk"
    assert med["score"] == 45.0
    
    high = calculate_risk_tier(0.85)
    assert high["tier"] == "High Risk"
    assert high["score"] == 85.0

def test_pipeline_transform_and_predict(artifacts):
    p = artifacts["preprocessor"]
    m = artifacts["model"]
    
    test_applicant = pd.DataFrame([{
        "person_age": 30,
        "person_income": 60000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.17,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }])
    
    X_trans = p.transform(test_applicant)
    assert X_trans.shape[1] == 25
    
    prob = m.predict_proba(X_trans)[0, 1]
    assert 0.0 <= prob <= 1.0

def test_database_logging():
    test_payload = {
        "applicant_name": "Unit Test User",
        "officer_name": "Test Officer",
        "person_age": 35,
        "person_income": 75000,
        "person_home_ownership": "OWN",
        "person_emp_length": 7.0,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "loan_amnt": 12000,
        "loan_int_rate": 8.5,
        "loan_percent_income": 0.16,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10,
        "risk_score": 12.5,
        "risk_tier": "Low Risk",
        "probability": 0.125,
        "underwriting_decision": "Approved",
        "officer_notes": "Unit test assessment record"
    }
    
    aid = save_assessment(test_payload)
    assert aid.startswith("CR-")
    
    recent = get_recent_assessments(limit=5)
    assert not recent.empty
    assert (recent['id'] == aid).any()

def test_stress_testing_simulation(artifacts):
    df_res = run_portfolio_stress_test(artifacts["sample_df"], artifacts["preprocessor"], artifacts["model"])
    assert len(df_res) == 4
    # Stress scenarios should yield higher NPL than baseline
    assert df_res.iloc[-1]["projected_npl_rate"] >= df_res.iloc[0]["projected_npl_rate"]

def test_fairness_evaluation(artifacts):
    fair_df = evaluate_model_fairness(artifacts["sample_df"], artifacts["preprocessor"], artifacts["model"])
    assert not fair_df.empty
    assert "disparate_impact_ratio" in fair_df.columns
