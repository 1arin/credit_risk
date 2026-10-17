import numpy as np
import pandas as pd

def calculate_risk_tier(probability: float):
    """
    Categorizes credit risk according to Section 7.3 of the Project Proposal:
    - Low Risk (0 - 30): ความน่าจะเป็นผิดนัดต่ำ ควรอนุมัติสินเชื่อ
    - Medium Risk (31 - 70): ควรพิจารณาเพิ่มเติมหรือขอเอกสารเสริม
    - High Risk (71 - 100): ความเสี่ยงสูง แนะนำปฏิเสธสินเชื่อ
    """
    score = round(probability * 100, 1)
    
    if score <= 30.0:
        tier = "Low Risk"
        thai_label = "ความเสี่ยงต่ำ"
        action = "ควรอนุมัติสินเชื่อ (Approve)"
        color = "#10B981" # Green
        badge_class = "badge-low"
        icon = "✅"
    elif score <= 70.0:
        tier = "Medium Risk"
        thai_label = "ความเสี่ยงปานกลาง"
        action = "ควรพิจารณาเพิ่มเติมหรือขอเอกสารเสริม (Under Review / Request Documents)"
        color = "#F59E0B" # Amber
        badge_class = "badge-medium"
        icon = "⚠️"
    else:
        tier = "High Risk"
        thai_label = "ความเสี่ยงสูง"
        action = "ไม่ควรอนุมัติ / ปฏิเสธสินเชื่อ (Reject)"
        color = "#EF4444" # Red
        badge_class = "badge-high"
        icon = "🚫"
        
    return {
        "score": score,
        "probability": float(probability),
        "tier": tier,
        "thai_label": thai_label,
        "action": action,
        "color": color,
        "badge_class": badge_class,
        "icon": icon
    }

def get_sample_personas():
    """
    Provides pre-defined realistic test profiles for instant interactive evaluation in Streamlit.
    """
    return {
        "1. คุณธนกร (มนุษย์เงินเดือนประวัติดีเยี่ยม - Low Risk)": {
            "person_age": 34,
            "person_income": 95000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 8.0,
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "A",
            "loan_amnt": 15000,
            "loan_int_rate": 7.5,
            "loan_percent_income": 0.16,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 9
        },
        "2. คุณนภัสสร (จบใหม่เริ่มทำงาน กู้เพื่อการศึกษา - Low to Medium Risk)": {
            "person_age": 23,
            "person_income": 35000,
            "person_home_ownership": "RENT",
            "person_emp_length": 2.0,
            "loan_intent": "EDUCATION",
            "loan_grade": "B",
            "loan_amnt": 8000,
            "loan_int_rate": 10.5,
            "loan_percent_income": 0.23,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 3
        },
        "3. คุณวิทวัส (ภาระหนี้ค่อนข้างสูง ต้องการกู้ขยายธุรกิจ - Medium Risk)": {
            "person_age": 29,
            "person_income": 45000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "VENTURE",
            "loan_grade": "C",
            "loan_amnt": 18000,
            "loan_int_rate": 13.5,
            "loan_percent_income": 0.40,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 5
        },
        "4. คุณเกียรติศักดิ์ (เคยมีประวัติค้างชำระ ดอกเบี้ยสูง DTI สูง - High Risk)": {
            "person_age": 25,
            "person_income": 22000,
            "person_home_ownership": "RENT",
            "person_emp_length": 1.0,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "E",
            "loan_amnt": 14000,
            "loan_int_rate": 18.2,
            "loan_percent_income": 0.64,
            "cb_person_default_on_file": "Y",
            "cb_person_cred_hist_length": 2
        }
    }
