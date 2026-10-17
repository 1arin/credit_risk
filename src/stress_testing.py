import pandas as pd
import numpy as np

def run_portfolio_stress_test(sample_df: pd.DataFrame, preprocessor, model):
    """
    Simulates portfolio behavior under macroeconomic stress scenarios according to Basel III / IFRS 9:
    1. Baseline: Current economic conditions
    2. Mild Recession: +1.5% interest rate, -5% income
    3. Severe Recession: +3.5% interest rate, -15% income
    4. Stagflation Crisis: +5.0% interest rate, -25% income
    """
    scenarios = {
        "Baseline (สภาวะปกติ)": {
            "int_delta": 0.0,
            "inc_factor": 1.0,
            "desc": "สภาพเศรษฐกิจปกติ อัตราดอกเบี้ยและรายได้ตามสัญญาเดิม"
        },
        "Scenario 1: Mild Recession (เศรษฐกิจชะลอตัว)": {
            "int_delta": 1.5,
            "inc_factor": 0.95,
            "desc": "ธนาคารกลางปรับขึ้นดอกเบี้ยนโยบาย +1.5%, รายได้ลดลง 5%"
        },
        "Scenario 2: Severe Downturn (วิกฤตเศรษฐกิจรุนแรง)": {
            "int_delta": 3.5,
            "inc_factor": 0.85,
            "desc": "วิกฤตเศรษฐกิจ ดอกเบี้ยพุ่ง +3.5%, รายได้ผู้กู้หดตัว 15%"
        },
        "Scenario 3: Stagflation (เงินเฟ้อรุนแรง + หนี้ครัวเรือนสูง)": {
            "int_delta": 5.0,
            "inc_factor": 0.75,
            "desc": "ภาวะ Stagflation ดอกเบี้ยสูง +5.0%, รายได้ลดลง 25%"
        }
    }
    
    results = []
    
    for sc_name, sc_params in scenarios.items():
        sim_df = sample_df.copy()
        sim_df['loan_int_rate'] = sim_df['loan_int_rate'] + sc_params['int_delta']
        sim_df['person_income'] = np.maximum(sim_df['person_income'] * sc_params['inc_factor'], 1000.0)
        sim_df['loan_percent_income'] = sim_df['loan_amnt'] / sim_df['person_income']
        
        X_trans = preprocessor.transform(sim_df)
        probs = model.predict_proba(X_trans)[:, 1]
        
        scores = probs * 100.0
        npl_rate = (scores > 50.0).mean() * 100.0
        high_risk_share = (scores > 70.0).mean() * 100.0
        med_risk_share = ((scores > 30.0) & (scores <= 70.0)).mean() * 100.0
        low_risk_share = (scores <= 30.0).mean() * 100.0
        
        # Approximate expected loss = sum(loan_amnt * prob * LGD=0.45)
        expected_loss = np.sum(sim_df['loan_amnt'] * probs * 0.45)
        
        results.append({
            "scenario": sc_name,
            "description": sc_params['desc'],
            "avg_risk_score": round(float(np.mean(scores)), 2),
            "projected_npl_rate": round(float(npl_rate), 2),
            "low_risk_pct": round(float(low_risk_share), 1),
            "med_risk_pct": round(float(med_risk_share), 1),
            "high_risk_pct": round(float(high_risk_share), 1),
            "expected_credit_loss": round(float(expected_loss), 2)
        })
        
    return pd.DataFrame(results)
