import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(
    page_title="REST API Documentation & Playground",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Sarabun:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Sarabun', sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div style="font-size: 13px; color: #a5b4fc; font-weight: 600; margin-bottom: 4px;">
        🔌 ระบบเชื่อมต่อภายนอก (Integration & API Gateway)
    </div>
    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 800;">
        FastAPI Credit Risk Scoring REST API Playground
    </h1>
    <p style="color: #94a3b8; margin-top: 6px; font-size: 15px;">
        ทดสอบยิง Request เข้าสู่โมเดล XGBoost Backend แบบ Real-Time ตามสถาปัตยกรรมรูปที่ 9.2 ใน Proposal
    </p>
</div>
""", unsafe_allow_html=True)

api_url = st.text_input("API Base URL:", value="http://localhost:8000")

tab_test1, tab_test2, tab_test3 = st.tabs([
    "⚡ ทดสอบคำนวณความเสี่ยง (POST /predict)",
    "🔄 ทดสอบ What-If Simulation (POST /what-if)",
    "📜 เอกสาร Swagger & cURL Examples"
])

with tab_test1:
    st.markdown("##### ยิง Payload ทดสอบการประเมินสินเชื่อ:")
    sample_json = {
        "applicant_name": "คุณนวพล ก้องกิดากร",
        "officer_name": "Senior Loan Officer",
        "person_age": 30,
        "person_income": 65000.0,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 5.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 12000.0,
        "loan_int_rate": 10.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6
    }
    
    req_body = st.text_area("JSON Request Body:", value=json.dumps(sample_json, indent=2, ensure_ascii=False), height=240)
    
    if st.button("🚀 ส่ง Request ไปยัง API", type="primary"):
        try:
            res = requests.post(f"{api_url}/predict", json=json.loads(req_body), timeout=5)
            st.success(f"Response Status: {res.status_code} OK")
            st.json(res.json())
        except Exception as err:
            st.warning(f"⚠️ ไม่สามารถเชื่อมต่อกับ API Server ที่ `{api_url}` ได้ (กรุณารัน `uvicorn api:app --port 8000` ในเทอร์มินัลก่อน)")
            st.info("คำสั่งรัน API Backend: `.\\.venv\\Scripts\\uvicorn api:app --reload --port 8000`")

with tab_test2:
    st.markdown("##### ทดสอบระบบแนะนำวงเงินกู้ (What-If Simulation):")
    sample_whatif = {
        "person_age": 27,
        "person_income": 30000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 2.0,
        "loan_intent": "VENTURE",
        "loan_grade": "D",
        "loan_amnt": 20000.0,
        "loan_int_rate": 16.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3
    }
    whatif_body = st.text_area("What-If JSON Request:", value=json.dumps(sample_whatif, indent=2, ensure_ascii=False), height=200)
    if st.button("⚡ ยิงทดสอบ What-If Analysis"):
        try:
            res = requests.post(f"{api_url}/what-if", json=json.loads(whatif_body), timeout=5)
            st.success(f"Status: {res.status_code}")
            st.json(res.json())
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถเชื่อมต่อได้: {e}")

with tab_test3:
    st.markdown("""
    #### 📖 ตัวอย่าง cURL สำหรับนักพัฒนาภายนอก:
    ```bash
    curl -X 'POST' \\
      'http://localhost:8000/predict' \\
      -H 'Content-Type: application/json' \\
      -d '{
        "applicant_name": "John Doe",
        "person_age": 28,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.0,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
      }'
    ```
    """)
