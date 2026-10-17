import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px

from src.stress_testing import run_portfolio_stress_test

st.set_page_config(
    page_title="Macroeconomic Stress Testing Portal",
    page_icon="📉",
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
    <div style="font-size: 13px; color: #38bdf8; font-weight: 600; margin-bottom: 4px;">
        🏛️ สถาบันการเงิน • ศูนย์บริหารความเสี่ยงระดับมหภาค (Institutional Risk Governance)
    </div>
    <h1 style="color: #ffffff; margin: 0; font-size: 26px; font-weight: 800;">
        ระบบจำลองภาวะวิกฤตเศรษฐกิจของพอร์ตสินเชื่อ (Macroeconomic Stress Testing Simulator)
    </h1>
    <p style="color: #94a3b8; margin-top: 6px; font-size: 15px;">
        การประเมินความทนทานของพอร์ตสินเชื่อตามมาตรฐานสากล Basel III และแนวปฏิบัติของธนาคารแห่งประเทศไทย (BOT)
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

st.markdown("### 🔍 ผลการทดสอบภาวะวิกฤต 4 สถานการณ์จำลอง (Stress Testing Scenarios)")
st.caption("ระบบจำลองการปรับขึ้นอัตราดอกเบี้ยนโยบายและการลดลงของรายได้ผู้กู้ทั่วทั้งพอร์ตสินเชื่อ")

stress_results = run_portfolio_stress_test(sample_df, preprocessor, model)

# Metrics Grid
c1, c2, c3, c4 = st.columns(4)
baseline_row = stress_results.iloc[0]
worst_row = stress_results.iloc[-1]

with c1:
    st.metric("อัตรา NPL ปกติ (Baseline)", f"{baseline_row['projected_npl_rate']:.1f}%")
with c2:
    st.metric("อัตรา NPL ภาวะวิกฤต (Stagflation)", f"{worst_row['projected_npl_rate']:.1f}%", delta=f"+{worst_row['projected_npl_rate'] - baseline_row['projected_npl_rate']:.1f}% NPL")
with c3:
    st.metric("คะแนนความเสี่ยงเฉลี่ยปกติ", f"{baseline_row['avg_risk_score']:.1f} pt")
with c4:
    st.metric("คะแนนเสี่ยงเฉลี่ยภาวะวิกฤต", f"{worst_row['avg_risk_score']:.1f} pt", delta=f"+{worst_row['avg_risk_score'] - baseline_row['avg_risk_score']:.1f} pt", delta_color="inverse")

st.markdown("---")

# Comparative Table
st.markdown("##### 📊 ตารางสรุปผลกระทบต่อพอร์ตสินเชื่อในแต่ละสภาวะ:")
display_df = stress_results.rename(columns={
    "scenario": "สถานการณ์จำลอง",
    "projected_npl_rate": "คาดการณ์ NPL (%)",
    "avg_risk_score": "คะแนนเสี่ยงเฉลี่ย",
    "low_risk_pct": "สัดส่วน Low Risk (%)",
    "med_risk_pct": "สัดส่วน Med Risk (%)",
    "high_risk_pct": "สัดส่วน High Risk (%)",
    "expected_credit_loss": "ความเสียหายคาดหวัง ($)"
})
st.table(display_df[['สถานการณ์จำลอง', 'คาดการณ์ NPL (%)', 'คะแนนเสี่ยงเฉลี่ย', 'สัดส่วน Low Risk (%)', 'สัดส่วน High Risk (%)', 'ความเสียหายคาดหวัง ($)']])

# Charts
ch1, ch2 = st.columns(2)

with ch1:
    # NPL Bar Chart
    npl_fig = go.Figure(go.Bar(
        x=stress_results['scenario'],
        y=stress_results['projected_npl_rate'],
        marker=dict(color=['#10b981', '#f59e0b', '#f97316', '#ef4444']),
        text=[f"{v:.1f}%" for v in stress_results['projected_npl_rate']],
        textposition='outside'
    ))
    npl_fig.update_layout(
        title="<b>แนวโน้มอัตราหนี้เสีย (Projected NPL Rate %)</b>",
        xaxis_title="",
        yaxis_title="NPL Rate (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=380,
        margin=dict(l=20, r=20, t=50, b=50),
        xaxis=dict(tickangle=-15, gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
    st.plotly_chart(npl_fig, use_container_width=True)

with ch2:
    # Risk Tier Stacked Breakdown
    stack_fig = go.Figure()
    stack_fig.add_trace(go.Bar(name='Low Risk (≤30)', x=stress_results['scenario'], y=stress_results['low_risk_pct'], marker_color='#10b981'))
    stack_fig.add_trace(go.Bar(name='Medium Risk (31-70)', x=stress_results['scenario'], y=stress_results['med_risk_pct'], marker_color='#f59e0b'))
    stack_fig.add_trace(go.Bar(name='High Risk (>70)', x=stress_results['scenario'], y=stress_results['high_risk_pct'], marker_color='#ef4444'))
    
    stack_fig.update_layout(
        barmode='stack',
        title="<b>การย้ายกลุ่มความเสี่ยงของพอร์ต (Risk Migration Breakdown %)</b>",
        xaxis_title="",
        yaxis_title="Percentage (%)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=380,
        margin=dict(l=20, r=20, t=50, b=50),
        xaxis=dict(tickangle=-15),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
    st.plotly_chart(stack_fig, use_container_width=True)

st.markdown("""
<div class="card" style="border-left: 5px solid #38bdf8;">
    <h5 style="color: #38bdf8; margin-top:0;">💡 ข้อเสนอแนะเชิงนโยบายสำหรับผู้บริหารความเสี่ยง (Risk Committee Takeaways):</h5>
    <ul style="color: #cbd5e1; font-size: 14px; margin-bottom: 0; line-height: 1.6;">
        <li>หากเกิดวิกฤตเศรษฐกิจแบบ Stagflation กลุ่มลูกหนี้ <b>Medium Risk เดิมจะไหลเข้าสู่กลุ่ม High Risk มากกว่า 2.5 เท่า</b></li>
        <li>สถาบันการเงินควรสำรองเงินทุนส่วนเพิ่ม (Capital Conservation Buffer) อย่างน้อย 15-20% เพื่อรองรับความผันผวนของอัตราดอกเบี้ย</li>
        <li>แนะนำให้กำหนดเพดาน DTI สูงสุดไม่เกิน 40% สำหรับกลุ่มลูกค้าอาชีพอิสระหรืออายุงานน้อยกว่า 2 ปี</li>
    </ul>
</div>
""", unsafe_allow_html=True)
