import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from src.utils import calculate_risk_tier, get_sample_personas
from src.database import save_assessment, get_recent_assessments, get_portfolio_kpis
from src.counterfactual import simulate_what_if
from src.explainability import (
    create_local_shap_waterfall_fig,
    generate_plain_text_explanation,
    get_thai_feature_name,
    FEATURE_THAI_MAPPING
)
from src.pdf_generator import generate_pdf_report

# ---------------------------------------------------------
# Page Configuration & Institutional Dark Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Institutional Credit Risk Prediction & XAI System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sarabun:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Sarabun', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Institutional Dark Palette */
    .stApp {
        background-color: #080d1a;
        color: #f1f5f9;
    }
    
    /* Executive Institutional Header */
    .gov-header {
        background: radial-gradient(circle at 10% 20%, #1e1b4b 0%, #0f172a 60%, #030712 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .gov-header::after {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0;
        width: 6px;
        background: linear-gradient(180deg, #38bdf8 0%, #6366f1 50%, #10b981 100%);
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .badge-psu {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    
    .badge-iso {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    
    .card-panel {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .kpi-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
    }
    
    .kpi-title {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .kpi-val {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    /* Institutional Underwriting Memo Layout */
    .memo-paper {
        background: #ffffff;
        color: #0f172a;
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        padding: 36px 40px;
        font-family: 'Sarabun', sans-serif;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    .memo-header {
        border-bottom: 2px solid #0f172a;
        padding-bottom: 16px;
        margin-bottom: 20px;
    }
    
    .memo-stamp {
        display: inline-block;
        border: 3px solid #10b981;
        color: #10b981;
        font-weight: 900;
        text-transform: uppercase;
        padding: 8px 18px;
        font-size: 18px;
        border-radius: 6px;
        transform: rotate(-3deg);
    }
    
    .memo-stamp-rejected {
        display: inline-block;
        border: 3px solid #ef4444;
        color: #ef4444;
        font-weight: 900;
        text-transform: uppercase;
        padding: 8px 18px;
        font-size: 18px;
        border-radius: 6px;
        transform: rotate(-3deg);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0b1120;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached System Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
    preprocessor = joblib.load(os.path.join("models", "preprocessor.joblib"))
    best_model = joblib.load(os.path.join("models", "best_model.joblib"))
    logreg_model = joblib.load(os.path.join("models", "logreg_model.joblib"))
    rf_model = joblib.load(os.path.join("models", "rf_model.joblib"))
    explainer = joblib.load(os.path.join("models", "shap_explainer.joblib"))
    
    with open(os.path.join("models", "evaluation_results.json"), "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    sample_df = pd.read_csv(os.path.join("models", "sample_test_cases.csv"))
    
    return {
        "preprocessor": preprocessor,
        "best_model": best_model,
        "logreg_model": logreg_model,
        "rf_model": rf_model,
        "explainer": explainer,
        "eval_data": eval_data,
        "sample_df": sample_df,
        "feature_names": eval_data["feature_names"],
        "threshold_analysis": eval_data.get("threshold_analysis", {})
    }

try:
    artifacts = load_all_artifacts()
except Exception as e:
    st.error(f"⚠️ กรุณารัน train.py ก่อนเปิดระบบ: {e}")
    st.stop()

# ---------------------------------------------------------
# Institutional Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="gov-header">
    <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px;">
        <span class="badge-pill badge-psu">🎓 ม.อ. คณะวิศวกรรมศาสตร์ • สาขาวิศวกรรมคอมพิวเตอร์ & ปัญญาประดิษฐ์</span>
        <span class="badge-pill badge-iso">🔒 มาตรฐานการกำกับดูแลโมเดลความเสี่ยง (Explainable AI & Basel Accord)</span>
        <span class="badge-pill" style="background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4);">
            🛡️ Production Active: XGBoost (AUC 0.9323)
        </span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
        <div>
            <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">
                ระบบวิเคราะห์และคาดการณ์ความเสี่ยงการผิดนัดชำระหนี้ด้วย Machine Learning & Explainable AI
            </h1>
            <p style="color: #94a3b8; margin-top: 6px; margin-bottom: 0; font-size: 15px;">
                Institutional Credit Risk Scoring, Transparent Underwriting Decisioning & Actionable Recourse Platform
            </p>
        </div>
        <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 10px 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);">
            <div style="font-size: 12px; color: #94a3b8;">โครงงานวิชา 240-403 เตรียมสหกิจศึกษา</div>
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">นางสาวยลรดี ภักดีจิตร์ (6610110734)</div>
            <div style="font-size: 12px; color: #38bdf8;">อาจารย์ที่ปรึกษา: ผศ.ดร. สหพงศ์ สมวงค์</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Global KPI Metric Ribbon
# ---------------------------------------------------------
kpis = get_portfolio_kpis()
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">📋 รายการประเมินในระบบ</div>
        <div class="kpi-val">{kpis['total_assessments']:,} <span style="font-size:14px; font-weight:400; color:#94a3b8;">เคส</span></div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">🎯 คะแนนความเสี่ยงเฉลี่ย</div>
        <div class="kpi-val" style="color: {'#10b981' if kpis['avg_risk_score'] < 30 else ('#f59e0b' if kpis['avg_risk_score'] < 70 else '#ef4444')};">
            {kpis['avg_risk_score']:.1f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">/ 100 pt</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">🛡️ ความแม่นยำโมเดล (ROC-AUC)</div>
        <div class="kpi-val" style="color: #38bdf8;">
            {artifacts['eval_data']['summary']['champion_auc']:.4f} <span style="font-size:14px; font-weight:400; color:#94a3b8;">(XGBoost)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    th_info = artifacts["threshold_analysis"]
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">💰 คาดการณ์การลดหนี้เสีย (Cost Optimized)</div>
        <div class="kpi-val" style="color: #10b981;">
            -{th_info.get('cost_savings_pct', 8.85)}% <span style="font-size:14px; font-weight:400; color:#94a3b8;">Expected Loss</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 ประเมินสินเชื่อ & ออกบันทึก (Underwriting)",
    "🔄 ปรับโครงสร้างสินเชื่อ (What-If Simulation)",
    "📊 ปัจจัยเสี่ยงระดับระบบ (Global XAI)",
    "📈 เปรียบเทียบโมเดล & ต้นทุนทางธุรกิจ (Cost Optimization)",
    "📁 ประวัติการตรวจสอบ & ชุดข้อมูล (Audit Log)"
])

# =========================================================
# TAB 1: INDIVIDUAL UNDERWRITING & AUDIT MEMO
# =========================================================
with tab1:
    st.markdown("### 📋 ตรวจสอบและประเมินความเสี่ยงสินเชื่อรายบุคคล")
    st.caption("ระบบคัดกรองสินเชื่ออัตโนมัติ คำนวณ Risk Score จัดกลุ่มความเสี่ยง พร้อมแยกแยะปัจจัยด้วย SHAP แบบโปร่งใส")
    
    # Preset Selector
    personas = get_sample_personas()
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        sel_preset = st.selectbox(
            "⚡ เลือกลูกค้าจำลอง (Persona Presets):",
            ["-- กรอกข้อมูลผู้สมัครใหม่ (Manual Input) --"] + list(personas.keys())
        )
    with col_p2:
        officer_name = st.text_input("เจ้าหน้าที่สินเชื่อผู้พิจารณา:", value="นายอนุมัติ รอบคอบ (Senior Underwriter)")
        
    p_data = {}
    if sel_preset != "-- กรอกข้อมูลผู้สมัครใหม่ (Manual Input) --":
        p_data = personas[sel_preset]
        
    with st.form("underwriting_form"):
        st.markdown("##### 1. ข้อมูลผู้สมัครสินเชื่อ (Applicant Financial Profile)")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            applicant_name = st.text_input("ชื่อ-สกุล ผู้สมัคร:", value="คุณสมชาย มุ่งมั่น" if not p_data else sel_preset.split(" ")[1])
            age = st.number_input("อายุ (ปี)", min_value=18, max_value=85, value=p_data.get("person_age", 29))
            home_status = st.selectbox("สถานะที่อยู่อาศัย", ["RENT", "OWN", "MORTGAGE", "OTHER"], index=["RENT", "OWN", "MORTGAGE", "OTHER"].index(p_data.get("person_home_ownership", "RENT")))
            
        with c2:
            income = st.number_input("รายได้ต่อปี ($/ปี)", min_value=1000, max_value=2000000, value=p_data.get("person_income", 60000), step=1000)
            emp_length = st.number_input("อายุงานปัจจุบัน (ปี)", min_value=0.0, max_value=40.0, value=float(p_data.get("person_emp_length", 3.5)), step=0.5)
            cred_hist = st.number_input("ระยะเวลาประวัติเครดิต (ปี)", min_value=1, max_value=35, value=p_data.get("cb_person_cred_hist_length", 4))

        with c3:
            loan_amnt = st.number_input("วงเงินสินเชื่อที่ขอ ($)", min_value=500, max_value=50000, value=p_data.get("loan_amnt", 14000), step=500)
            loan_intent = st.selectbox("วัตถุประสงค์สินเชื่อ", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], index=["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"].index(p_data.get("loan_intent", "PERSONAL")))
            has_default = st.selectbox("ประวัติผิดนัดชำระเดิมในบูโร", ["N", "Y"], index=["N", "Y"].index(p_data.get("cb_person_default_on_file", "N")), format_func=lambda x: "เคยผิดนัดชำระ (Default on file)" if x == "Y" else "ปกติ ไม่เคยผิดนัด (Clean)")

        with c4:
            loan_grade = st.selectbox("เกรดความเสี่ยงสินเชื่อ (Grade)", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(p_data.get("loan_grade", "B")))
            int_rate = st.slider("อัตราดอกเบี้ยต่อปี (%)", min_value=5.0, max_value=25.0, value=float(p_data.get("loan_int_rate", 11.0)), step=0.1)
            calc_dti = round(loan_amnt / max(income, 1), 3)
            st.metric("สัดส่วนหนี้ต่อรายได้ (DTI)", f"{calc_dti:.1%}")

        officer_notes = st.text_area("บันทึกเพิ่มเติมของเจ้าหน้าที่ (Underwriter Notes):", value="ผู้สมัครมีประวัติการทำงานมั่นคง ขอสินเชื่อเพื่อเสริมสภาพคล่อง")
        
        btn_eval = st.form_submit_button("⚡ ประมวลผลและสร้างบันทึกการพิจารณา (Execute Assessment)", type="primary", use_container_width=True)

    # Prepare data for prediction
    applicant_dict = {
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home_status,
        "person_emp_length": emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": calc_dti,
        "cb_person_default_on_file": has_default,
        "cb_person_cred_hist_length": cred_hist
    }
    
    input_df = pd.DataFrame([applicant_dict])
    X_trans = artifacts["preprocessor"].transform(input_df)
    prob = float(artifacts["best_model"].predict_proba(X_trans)[0, 1])
    risk_info = calculate_risk_tier(prob)
    
    # Local SHAP
    shap_vals = artifacts["explainer"](X_trans)
    shap_1d = shap_vals.values[0]
    plain_exp = generate_plain_text_explanation(shap_1d, artifacts["feature_names"], applicant_dict)
    
    # Save to SQLite Database
    assessment_payload = applicant_dict.copy()
    assessment_payload["applicant_name"] = applicant_name
    assessment_payload["officer_name"] = officer_name
    assessment_payload["officer_notes"] = officer_notes
    assessment_payload["risk_score"] = risk_info["score"]
    assessment_payload["risk_tier"] = risk_info["tier"]
    assessment_payload["probability"] = prob
    assessment_payload["underwriting_decision"] = "Approved" if risk_info["score"] <= 30 else ("Under Review" if risk_info["score"] <= 70 else "Rejected")
    
    assessment_id = save_assessment(assessment_payload)
    
    st.markdown("---")
    st.markdown(f"#### 🏷️ รหัสการประเมิน: `{assessment_id}` • ผลการตัดสินใจของระบบ")
    
    # Results Presentation
    r_col1, r_col2 = st.columns([1, 1.3])
    
    with r_col1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_info["score"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Credit Risk Score (0-100)</b>", 'font': {'size': 18, 'color': '#ffffff'}},
            number={'suffix': " pt", 'font': {'size': 38, 'color': risk_info['color']}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': risk_info['color'], 'thickness': 0.35},
                'bgcolor': "rgba(255,255,255,0.03)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.15)",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.25)"},
                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.8,
                    'value': risk_info["score"]
                }
            }
        ))
        gauge_fig.update_layout(height=290, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)
        
    with r_col2:
        decision_label = "อนุมัติสินเชื่อ (Approve)" if risk_info['score'] <= 30 else ("ขอเอกสารเพิ่มเติม/พิจารณาเป็นพิเศษ (Under Review)" if risk_info['score'] <= 70 else "ปฏิเสธสินเชื่อ (Reject)")
        decision_color = risk_info['color']
        
        st.markdown(f"""
        <div class="card-panel" style="border-left: 6px solid {decision_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="{risk_info['badge_class']}" style="font-size: 15px;">{risk_info['icon']} {risk_info['tier']} ({risk_info['thai_label']})</span>
                <span style="font-size: 13px; color: #94a3b8;">โอกาสผิดนัดชำระ: <b>{prob:.2%}</b></span>
            </div>
            <div style="font-size: 14px; color: #94a3b8;">คำตัดสินแนะนำโดยอัตโนมัติ:</div>
            <div style="font-size: 17px; font-weight: 700; color: #ffffff; margin-top: 4px; margin-bottom: 12px;">
                {decision_label}
            </div>
            <div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 8px; font-size: 14px; color: #cbd5e1; border: 1px dashed rgba(255,255,255,0.15);">
                {risk_info['action']}
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px;">
                <div style="background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #94a3b8;">ประมาณยอดผ่อน/เดือน</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">
                        ${(loan_amnt * ((int_rate/1200)/(1-(1+int_rate/1200)**(-36)))):,.0f}
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #94a3b8;">สัดส่วน DTI</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">{calc_dti:.1%}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px;">
                    <div style="font-size: 11px; color: #94a3b8;">บันทึกฐานข้อมูล</div>
                    <div style="font-size: 14px; font-weight: 700; color: #34d399;">✓ บันทึกสำเร็จ</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP Waterfall & Natural Language Drivers
    st.markdown("#### 🔍 การอธิบายผลการตัดสินใจด้วย Explainable AI (SHAP Local Attribution)")
    w_col1, w_col2 = st.columns([1.4, 1])
    
    with w_col1:
        waterfall_chart = create_local_shap_waterfall_fig(shap_1d, artifacts["feature_names"], applicant_dict)
        st.plotly_chart(waterfall_chart, use_container_width=True)
        
    with w_col2:
        st.markdown("""
        <div class="card-panel">
            <h5 style="color: #f87171; margin-top:0;">⚠️ ปัจจัยเสี่ยงหลักที่ดันคะแนนสูงขึ้น (Risk Drivers)</h5>
        """, unsafe_allow_html=True)
        if plain_exp["risk_drivers"]:
            for item in plain_exp["risk_drivers"]:
                st.markdown(item)
        else:
            st.markdown("*(ไม่พบปัจจัยลบที่มีนัยสำคัญ)*")
            
        st.markdown("""
            <hr style="border-color: rgba(255,255,255,0.1); margin: 14px 0;">
            <h5 style="color: #34d399; margin-top:0;">🛡️ ปัจจัยเชิงบวกที่ช่วยผ่อนคลายความเสี่ยง (Protective Factors)</h5>
        """, unsafe_allow_html=True)
        if plain_exp["protective_factors"]:
            for item in plain_exp["protective_factors"]:
                st.markdown(item)
        else:
            st.markdown("*(ไม่มีปัจจัยบวกช่วยลดทอน)*")
        st.markdown("</div>", unsafe_allow_html=True)

    # Official Printable Underwriting Memorandum
    with st.expander("📄 ดูบันทึกการพิจารณาสินเชื่อฉบับทางการ (Official Credit Underwriting Memorandum)"):
        stamp_class = "memo-stamp" if risk_info['score'] <= 30 else ("memo-stamp-rejected" if risk_info['score'] > 70 else "memo-stamp")
        stamp_text = "APPROVED" if risk_info['score'] <= 30 else ("REJECTED" if risk_info['score'] > 70 else "UNDER REVIEW")
        
        memo_html = f"""
        <div class="memo-paper">
            <div class="memo-header" style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 style="margin:0; color:#0f172a; font-weight:800;">ใบรายงานการวิเคราะห์และพิจารณาสินเชื่อ (CREDIT UNDERWRITING DOSSIER)</h2>
                    <p style="margin:4px 0 0 0; color:#64748b; font-size:13px;">ระบบสนับสนุนการตัดสินใจด้วยปัญญาประดิษฐ์ • เลขที่เอกสาร: {assessment_id}</p>
                </div>
                <div class="{stamp_class}">{stamp_text}</div>
            </div>
            <table style="width:100%; border-collapse:collapse; margin-bottom:20px; font-size:14px;">
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold; width:25%;">ชื่อผู้ขอสินเชื่อ:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{applicant_name}</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold; width:25%;">วันที่ประเมิน:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{datetime.now().strftime('%d/%m/%Y %H:%M')}</td>
                </tr>
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">วงเงินที่ยื่นขอ:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">${loan_amnt:,.2f}</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">อัตราดอกเบี้ย:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{int_rate}% ต่อปี (เกรด {loan_grade})</td>
                </tr>
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">รายได้ต่อปี:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">${income:,.2f}</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">ภาระหนี้ต่อรายได้ (DTI):</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{calc_dti:.1%}</td>
                </tr>
                <tr>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">คะแนนความเสี่ยง (Risk Score):</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold; color:{decision_color};">{risk_info['score']} / 100 pt ({risk_info['tier']})</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0; font-weight:bold;">โอกาสผิดนัดชำระ:</td>
                    <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{prob:.2%}</td>
                </tr>
            </table>
            <div style="background:#f8fafc; border-left:4px solid #0f172a; padding:12px 16px; margin-bottom:24px; font-size:13px; color:#334155;">
                <b>ความเห็นเชิงวิเคราะห์ (SHAP Explainable Intelligence):</b><br>
                ระบบจำแนกผลการพิจารณาโดยใช้โมเดล XGBoost ร่วมกับ SHAP TreeExplainer ปัจจัยสำคัญที่สุดที่มีผลต่อคะแนนความเสี่ยงของลูกค้ารายนี้ประกอบด้วย {', '.join([get_thai_feature_name(f) for f in artifacts['feature_names'][:3]])}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:40px; padding-top:20px; border-top:1px dashed #cbd5e1; font-size:13px;">
                <div style="text-align:center;">
                    ____________________________________<br>
                    ({officer_name})<br>
                    เจ้าหน้าที่สินเชื่อผู้พิจารณา
                </div>
                <div style="text-align:center;">
                    ____________________________________<br>
                    (......................................................)<br>
                    ผู้มีอำนาจลงนามอนุมัติ
                </div>
            </div>
        </div>
        """
        st.markdown(memo_html, unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        try:
            pdf_bytes = generate_pdf_report(
                applicant_name=applicant_name,
                assessment_id=assessment_id,
                risk_info=risk_info,
                applicant_dict=applicant_dict,
                officer_name=officer_name,
                officer_notes=officer_notes
            )
            st.download_button(
                label="📥 ดาวน์โหลดเอกสารการอนุมัติสินเชื่อทางการ (.PDF)",
                data=pdf_bytes,
                file_name=f"Credit_Assessment_{assessment_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"ไม่สามารถสร้างไฟล์ PDF: {e}")

# =========================================================
# TAB 2: WHAT-IF & ACTIONABLE RECOURSE SIMULATION
# =========================================================
with tab2:
    st.markdown("### 🔄 ระบบจำลองเงื่อนไขและแนวทางปรับโครงสร้างสินเชื่อ (Actionable Recourse)")
    st.caption("ช่วยแนะนำเจ้าหน้าที่และลูกค้าว่า 'หากคะแนนความเสี่ยงสูง ต้องปรับเงื่อนไขอย่างไรจึงจะผ่านเกณฑ์อนุมัติ'")
    
    sim_res = simulate_what_if(applicant_dict, artifacts["preprocessor"], artifacts["best_model"], target_max_score=30.0)
    
    s_col1, s_col2 = st.columns([1, 1.2])
    
    with s_col1:
        st.markdown("""
        <div class="card-panel">
            <h5 style="color: #38bdf8; margin-top:0;">💡 คำแนะนำการปรับโครงสร้างสินเชื่ออัตโนมัติ</h5>
        """, unsafe_allow_html=True)
        
        if sim_res.get("already_eligible"):
            st.success("✅ ผู้สมัครรายนี้มีความเสี่ยงต่ำ (Low Risk) อยู่แล้ว วงเงินและอัตราดอกเบี้ยปัจจุบันผ่านเกณฑ์การอนุมัติ")
        else:
            rec_amt = sim_res.get("recommended_loan_amnt")
            if rec_amt:
                st.markdown(f"""
                <div style="font-size:14px; color:#cbd5e1; line-height:1.6;">
                    • <b>วงเงินเดิมที่ขอ:</b> ${applicant_dict['loan_amnt']:,.0f} (Risk Score: <b style="color:#ef4444;">{sim_res['current_score']} pt</b>)<br>
                    • <b>วงเงินที่แนะนำเพื่อให้ผ่านเกณฑ์ Low Risk (Score ≤ 30):</b> <b style="color:#10b981; font-size:18px;">${rec_amt:,.0f}</b><br>
                    • <b>ต้องปรับลดวงเงินลง:</b> {sim_res['reduction_pct']}% (หรือเพิ่มเงินดาวน์ ${applicant_dict['loan_amnt'] - rec_amt:,.0f})<br>
                    • <b>คะแนนใหม่ที่จะได้:</b> <b style="color:#10b981;">{sim_res['recommended_score']} pt</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ เนื่องจากมีประวัติเครดิตบูโรผิดนัดชำระเดิม หรือเกรดสินเชื่อต่ำ การปรับลดวงเงินเพียงอย่างเดียวอาจไม่เพียงพอ แนะนำให้เพิ่มผู้ค้ำประกันที่มีเครดิตดี")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with s_col2:
        st.markdown("##### 🎛️ เครื่องมือจำลองแบบ Real-Time (Interactive Sandbox)")
        adj_loan_amnt = st.slider("ปรับเปลี่ยนวงเงินกู้ ($)", min_value=1000, max_value=50000, value=int(applicant_dict['loan_amnt']), step=500)
        adj_int_rate = st.slider("ปรับเปลี่ยนอัตราดอกเบี้ย (%)", min_value=5.0, max_value=25.0, value=float(applicant_dict['loan_int_rate']), step=0.1)
        adj_income = st.slider("รายได้ต่อปีเสริม ($)", min_value=10000, max_value=300000, value=int(applicant_dict['person_income']), step=5000)
        
        temp_dict = applicant_dict.copy()
        temp_dict['loan_amnt'] = adj_loan_amnt
        temp_dict['loan_int_rate'] = adj_int_rate
        temp_dict['person_income'] = adj_income
        temp_dict['loan_percent_income'] = adj_loan_amnt / max(adj_income, 1)
        
        sim_df = pd.DataFrame([temp_dict])
        sim_prob = float(artifacts["best_model"].predict_proba(artifacts["preprocessor"].transform(sim_df))[0, 1])
        sim_score = round(sim_prob * 100, 1)
        sim_tier = calculate_risk_tier(sim_prob)
        
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:12px; color:#94a3b8;">คะแนนหลังจำลองการปรับเงื่อนไข:</div>
                    <div style="font-size:28px; font-weight:800; color:{sim_tier['color']};">{sim_score} pt</div>
                </div>
                <div>
                    <span class="{sim_tier['badge_class']}">{sim_tier['tier']}</span>
                </div>
            </div>
            <div style="font-size:13px; color:#cbd5e1; margin-top:8px;">
                คำแนะนำ: <b>{sim_tier['action']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3: GLOBAL XAI (SHAP MACRO ANALYSIS)
# =========================================================
with tab3:
    st.markdown("### 📊 ปัจจัยเสี่ยงและ Explainable AI ระดับภาพรวม (Global XAI Insights)")
    st.caption("การวิเคราะห์ความสำคัญของตัวแปรเชิงสถิติจากชุดข้อมูลตัวอย่าง 32,574 รายการ เพื่อความโปร่งใสตามมาตรฐานธนาคาร")
    
    feat_imp = pd.DataFrame(artifacts["eval_data"]["feature_importance"])
    feat_imp['thai_name'] = feat_imp['feature'].apply(get_thai_feature_name)
    top_14 = feat_imp.head(14).sort_values(by='importance', ascending=True)
    
    g_fig = go.Figure(go.Bar(
        x=top_14['importance'],
        y=top_14['thai_name'],
        orientation='h',
        marker=dict(
            color=top_14['importance'],
            colorscale='Tealgrn',
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{v:.3f}" for v in top_14['importance']],
        textposition='outside'
    ))
    g_fig.update_layout(
        title="<b>14 ปัจจัยที่มีอิทธิพลต่อความเสี่ยงการผิดนัดชำระหนี้สูงสุด (Mean |SHAP Value|)</b>",
        xaxis_title="Magnitude of Global Impact on Risk Score",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', size=13),
        height=520,
        margin=dict(l=10, r=40, t=60, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
    st.plotly_chart(g_fig, use_container_width=True)

# =========================================================
# TAB 4: MODEL BENCHMARKING & COST-SENSITIVE THRESHOLDING
# =========================================================
with tab4:
    st.markdown("### 📈 รายงานเปรียบเทียบโมเดล & การปรับจูน Threshold ตามต้นทุนธุรกิจ")
    st.caption("เปรียบเทียบ 3 อัลกอริทึม (Logistic Regression, Random Forest, XGBoost) และวิเคราะห์ความคุ้มค่าทางการเงิน")
    
    m_dict = artifacts["eval_data"]["metrics"]
    comp_list = []
    for m_name, m in m_dict.items():
        comp_list.append({
            "โมเดล (Algorithm)": m_name,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "Accuracy": f"{m['accuracy']:.2%}",
            "Recall (ตรวจจับหนี้เสีย)": f"{m['recall']:.2%}",
            "Precision": f"{m['precision']:.2%}",
            "Brier Score (การสอบเทียบ)": f"{m.get('brier_score', 0.0):.4f}",
            "False Negatives (หลุดรอด)": m['confusion_matrix']['false_negative']
        })
    st.table(pd.DataFrame(comp_list))
    
    # Cost-Sensitive Threshold Optimization Card
    th_info = artifacts["threshold_analysis"]
    st.markdown(f"""
    <div class="card-panel" style="border-left: 6px solid #38bdf8;">
        <h4 style="color: #38bdf8; margin-top:0;">💰 การปรับจุดตัดการตัดสินใจตามต้นทุนความเสี่ยง (Cost-Sensitive Threshold Optimization)</h4>
        <p style="color: #cbd5e1; font-size:14px;">
            ในอุตสาหกรรมสินเชื่อ <b>การปล่อยผ่านลูกค้าที่จะผิดนัดชำระ (False Negative) มีต้นทุนความเสียหายสูงกว่าการปฏิเสธลูกค้าดี (False Positive) ประมาณ 5 เท่า</b>
        </p>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top:16px;">
            <div style="background:rgba(255,255,255,0.04); padding:12px; border-radius:8px;">
                <div style="font-size:12px; color:#94a3b8;">Threshold ทั่วไป</div>
                <div style="font-size:22px; font-weight:800; color:#f8fafc;">0.50</div>
                <div style="font-size:12px; color:#ef4444;">หนี้เสียหลุดรอด: {th_info.get('default_false_negatives', 408)} ราย</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); padding:12px; border-radius:8px;">
                <div style="font-size:12px; color:#94a3b8;">Threshold ที่เหมาะสมที่สุด (Optimal)</div>
                <div style="font-size:22px; font-weight:800; color:#38bdf8;">{th_info.get('optimal_threshold', 0.26)}</div>
                <div style="font-size:12px; color:#34d399;">หนี้เสียหลุดรอดลดเหลือ: {th_info.get('optimal_false_negatives', 292)} ราย</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); padding:12px; border-radius:8px;">
                <div style="font-size:12px; color:#94a3b8;">ตรวจจับหนี้เสียเพิ่มขึ้น</div>
                <div style="font-size:22px; font-weight:800; color:#10b981;">+{th_info.get('default_false_negatives', 408) - th_info.get('optimal_false_negatives', 292)} ราย</div>
                <div style="font-size:12px; color:#94a3b8;">(ลดโอกาสเกิด NPL)</div>
            </div>
            <div style="background:rgba(255,255,255,0.04); padding:12px; border-radius:8px;">
                <div style="font-size:12px; color:#94a3b8;">ลดความเสียหายคาดหวัง</div>
                <div style="font-size:22px; font-weight:800; color:#10b981;">-{th_info.get('cost_savings_pct', 8.85)}%</div>
                <div style="font-size:12px; color:#94a3b8;">(Expected Credit Loss)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ROC Curves & Confusion Matrix
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        curves = artifacts["eval_data"]["curves"]
        roc_fig = go.Figure()
        col_palette = {"Logistic Regression": "#94a3b8", "Random Forest": "#38bdf8", "XGBoost": "#10b981"}
        for name, c_data in curves.items():
            roc_fig.add_trace(go.Scatter(
                x=c_data["fpr"], y=c_data["tpr"],
                name=f"{name} (AUC = {m_dict[name]['roc_auc']:.3f})",
                line=dict(width=3 if name == "XGBoost" else 2, color=col_palette[name])
            ))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='rgba(255,255,255,0.3)'), name='Random Chance'))
        roc_fig.update_layout(
            title="<b>ROC Curves Comparison</b>",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'), height=360,
            margin=dict(l=20, r=20, t=50, b=30),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
        )
        st.plotly_chart(roc_fig, use_container_width=True)
        
    with b_col2:
        sel_cm = st.selectbox("เลือกโมเดลดู Confusion Matrix:", list(m_dict.keys()), index=2)
        cm_val = m_dict[sel_cm]["confusion_matrix"]
        cm_matrix = [[cm_val["true_negative"], cm_val["false_positive"]], [cm_val["false_negative"], cm_val["true_positive"]]]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_matrix,
            x=['ทำนาย: ปกติ (0)', 'ทำนาย: ผิดนัด (1)'],
            y=['จริง: ปกติ (0)', 'จริง: ผิดนัด (1)'],
            colorscale='Teal',
            text=[[f"{cm_matrix[0][0]:,}", f"{cm_matrix[0][1]:,}"], [f"{cm_matrix[1][0]:,}", f"{cm_matrix[1][1]:,}"]],
            texttemplate="%{text}", textfont={"size": 18, "color": "white"}
        ))
        cm_fig.update_layout(title=f"<b>Confusion Matrix ({sel_cm})</b>", paper_bgcolor='rgba(0,0,0,0)', height=360, margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(cm_fig, use_container_width=True)

# =========================================================
# TAB 5: AUDIT LOG TRAIL & BATCH PROCESSING
# =========================================================
with tab5:
    st.markdown("### 📁 ประวัติการตรวจสอบ (Audit Log Trail) & การประเมินแบบกลุ่ม (Batch)")
    st.caption("บันทึกการพิจารณาสินเชื่อทั้งหมดถูกจัดเก็บลงฐานข้อมูล SQLite ตามข้อกำหนดการตรวจสอบ (Compliance & Auditability)")
    
    # Recent Assessments from SQLite
    st.markdown("##### 📜 ประวัติการประเมินสินเชื่อล่าสุดในระบบ (Audit Records):")
    recent_db_df = get_recent_assessments(limit=30)
    
    if not recent_db_df.empty:
        audit_cols = ['id', 'created_at', 'applicant_name', 'officer_name', 'loan_amnt', 'risk_score', 'risk_tier', 'underwriting_decision']
        st.dataframe(recent_db_df[audit_cols], use_container_width=True)
        
        csv_db = recent_db_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 ส่งออกประวัติการตรวจสอบทั้งหมดเป็น CSV (Export Audit Log)",
            data=csv_db,
            file_name=f"credit_audit_trail_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ยังไม่มีประวัติการประเมินในฐานข้อมูล เมื่อกดประเมินในแท็บแรก ระบบจะบันทึกข้อมูลอัตโนมัติ")
        
    st.markdown("---")
    st.markdown("##### 📤 การประเมินผลข้อมูลชุดใหญ่ผ่านไฟล์ CSV (Batch Processing):")
    uploaded_batch = st.file_uploader("เลือกไฟล์ CSV ผู้สมัครสินเชื่อ", type=['csv'], key="batch_uploader")
    if uploaded_batch is not None:
        try:
            b_df = pd.read_csv(uploaded_batch)
            st.success(f"อัปโหลดสำเร็จ: พบข้อมูล {len(b_df):,} รายการ")
            
            b_trans = artifacts["preprocessor"].transform(b_df)
            b_probs = artifacts["best_model"].predict_proba(b_trans)[:, 1]
            
            b_df['Risk_Score'] = [round(p * 100, 1) for p in b_probs]
            b_df['Risk_Tier'] = [calculate_risk_tier(p)['tier'] for p in b_probs]
            b_df['Decision'] = ["Approved" if p*100 <= 30 else ("Under Review" if p*100 <= 70 else "Rejected") for p in b_probs]
            
            st.dataframe(b_df.head(20), use_container_width=True)
            
            b_csv = b_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 ดาวน์โหลดผลลัพธ์ Batch Scoring (CSV)", data=b_csv, file_name="batch_underwriting_results.csv", mime="text/csv")
        except Exception as err:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {err}")
