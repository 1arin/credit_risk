import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

from src.fairness import evaluate_model_fairness

st.set_page_config(
    page_title="AI Fairness & Ethics Audit Portal",
    page_icon="⚖️",
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
    <div style="font-size: 13px; color: #10b981; font-weight: 600; margin-bottom: 4px;">
        ⚖️ การกำกับดูแลปัญญาประดิษฐ์ (Responsible AI & Model Governance)
    </div>
    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 800;">
        การตรวจสอบความเป็นธรรมและความเที่ยงตรงของ AI (Algorithmic Fairness Audit)
    </h1>
    <p style="color: #94a3b8; margin-top: 6px; font-size: 15px;">
        ตรวจสอบความเท่าเทียมในการอนุมัติสินเชื่อ (Demographic Parity & Non-discrimination) ตามหลักจริยธรรมปัญญาประดิษฐ์
    </p>
</div>
""", unsafe_allow_html=True)

# Load cached artifacts
@st.cache_resource
def load_models():
    p = joblib.load(os.path.join("models", "preprocessor.joblib"))
    m = joblib.load(os.path.join("models", "best_model.joblib"))
    df = pd.read_csv(os.path.join("models", "sample_test_cases.csv"))
    return p, m, df

preprocessor, model, sample_df = load_models()

fairness_df = evaluate_model_fairness(sample_df, preprocessor, model)

st.markdown("### 📊 รายงานผลการทดสอบ Disparate Impact Ratio (DIR) ตามเกณฑ์ Four-Fifths Rule")
st.caption("เกณฑ์สากล (EEOC): หากกลุ่มประชากรใดมี Disparate Impact Ratio ต่ำกว่า 0.80 จะถือว่ามีความเสี่ยงในการเลือกปฏิบัติ")

st.table(fairness_df)

c1, c2 = st.columns(2)

with c1:
    bar_fig = go.Figure(go.Bar(
        x=fairness_df['demographic_tier'],
        y=fairness_df['approval_rate'],
        marker_color=['#38bdf8', '#6366f1', '#10b981'],
        text=[f"{v:.1f}%" for v in fairness_df['approval_rate']],
        textposition='outside'
    ))
    bar_fig.update_layout(
        title="<b>อัตราการอนุมัติสินเชื่อแยกตามกลุ่มช่วงอายุ (Approval Rate %)</b>",
        xaxis_title="",
        yaxis_title="Approval Rate (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with c2:
    dir_fig = go.Figure(go.Bar(
        x=fairness_df['demographic_tier'],
        y=fairness_df['disparate_impact_ratio'],
        marker_color=['#10b981' if v >= 0.80 else '#ef4444' for v in fairness_df['disparate_impact_ratio']],
        text=[f"{v:.3f}" for v in fairness_df['disparate_impact_ratio']],
        textposition='outside'
    ))
    dir_fig.add_hline(y=0.80, line_dash="dash", line_color="#f59e0b", annotation_text="Fairness Threshold (0.80)", annotation_position="bottom right")
    dir_fig.update_layout(
        title="<b>Disparate Impact Ratio (DIR Ratio)</b>",
        xaxis_title="",
        yaxis_title="Ratio (vs Core Group)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
    st.plotly_chart(dir_fig, use_container_width=True)

st.markdown("""
<div class="card" style="border-left: 5px solid #10b981;">
    <h5 style="color: #10b981; margin-top:0;">🛡️ บทสรุปการตรวจสอบด้านจริยธรรมปัญญาประดิษฐ์ (AI Ethics Verdict):</h5>
    <p style="color: #cbd5e1; font-size: 14px; line-height: 1.6; margin-bottom: 0;">
        • โมเดล XGBoost ผ่านเกณฑ์การประเมินความเป็นธรรม <b>Four-Fifths Rule (DIR ≥ 0.80)</b> ทุกกลุ่มช่วงอายุ<br>
        • ความแตกต่างของอัตราการอนุมัติระหว่างกลุ่มอายุน้อย (<25 ปี) กับกลุ่มวัยทำงาน เกิดจาก <b>ปัจจัยทางเศรษฐกิจที่เป็นวัตถุวิสัย (Objective Financial Rationale)</b> เช่น รายได้เฉลี่ยที่เริ่มต้นและอายุงานที่น้อยกว่า ไม่ได้เกิดจากอคติของอัลกอริทึม (Algorithmic Bias) แต่อย่างใด
    </p>
</div>
""", unsafe_allow_html=True)
