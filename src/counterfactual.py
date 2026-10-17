import pandas as pd
import numpy as np

def simulate_what_if(base_applicant: dict, preprocessor, model, target_max_score=30.0):
    """
    Simulates actionable recourse for applicants categorized as Medium or High Risk:
    Finds recommended adjustments (Loan Amount reduction, Down payment, Term adjustments)
    that bring the applicant's risk score into the Low Risk (< 30) tier.
    """
    original_loan_amnt = float(base_applicant.get("loan_amnt", 10000))
    current_df = pd.DataFrame([base_applicant])
    current_prob = float(model.predict_proba(preprocessor.transform(current_df))[0, 1])
    current_score = round(current_prob * 100, 1)
    
    if current_score <= target_max_score:
        return {
            "already_eligible": True,
            "current_score": current_score,
            "message": "ผู้สมัครรายนี้อยู่ในเกณฑ์ความเสี่ยงต่ำ (Low Risk) อยู่แล้ว ไม่จำเป็นต้องปรับโครงสร้างสินเชื่อ"
        }
        
    # Search for acceptable loan amount
    step_down = original_loan_amnt * 0.05
    test_amounts = np.arange(original_loan_amnt, 500, -step_down)
    
    qualifying_amount = None
    qualifying_score = None
    
    for amt in test_amounts:
        temp_dict = base_applicant.copy()
        temp_dict["loan_amnt"] = amt
        temp_dict["loan_percent_income"] = amt / max(float(temp_dict["person_income"]), 1.0)
        temp_df = pd.DataFrame([temp_dict])
        
        prob = float(model.predict_proba(preprocessor.transform(temp_df))[0, 1])
        score = round(prob * 100, 1)
        
        if score <= target_max_score:
            qualifying_amount = amt
            qualifying_score = score
            break
            
    # Also test for Medium Risk threshold (70.0) if High Risk
    qualifying_med_amount = None
    qualifying_med_score = None
    if current_score > 70.0:
        for amt in test_amounts:
            temp_dict = base_applicant.copy()
            temp_dict["loan_amnt"] = amt
            temp_dict["loan_percent_income"] = amt / max(float(temp_dict["person_income"]), 1.0)
            temp_df = pd.DataFrame([temp_dict])
            prob = float(model.predict_proba(preprocessor.transform(temp_df))[0, 1])
            score = round(prob * 100, 1)
            if score <= 70.0:
                qualifying_med_amount = amt
                qualifying_med_score = score
                break

    reduction_pct = 0
    if qualifying_amount:
        reduction_pct = round(((original_loan_amnt - qualifying_amount) / original_loan_amnt) * 100, 1)

    return {
        "already_eligible": False,
        "current_score": current_score,
        "original_loan_amnt": original_loan_amnt,
        "recommended_loan_amnt": qualifying_amount,
        "recommended_score": qualifying_score,
        "reduction_pct": reduction_pct,
        "medium_risk_threshold_loan_amnt": qualifying_med_amount,
        "medium_risk_score": qualifying_med_score
    }
