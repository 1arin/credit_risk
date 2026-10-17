import numpy as np
import pandas as pd
import plotly.graph_objects as go

FEATURE_THAI_MAPPING = {
    "loan_grade": "เกรดสินเชื่อ (Loan Grade)",
    "person_income": "รายได้ผู้กู้ (Income)",
    "loan_percent_income": "สัดส่วนหนี้ต่อรายได้ (DTI)",
    "loan_intent_VENTURE": "วัตถุประสงค์เพื่อการลงทุน (Venture)",
    "loan_intent_DEBTCONSOLIDATION": "วัตถุประสงค์รวมหนี้ (Debt Consolidation)",
    "loan_intent_EDUCATION": "วัตถุประสงค์เพื่อการศึกษา (Education)",
    "loan_intent_HOMEIMPROVEMENT": "วัตถุประสงค์ปรับปรุงบ้าน (Home Improvement)",
    "loan_intent_MEDICAL": "วัตถุประสงค์ค่ารักษาพยาบาล (Medical)",
    "loan_intent_PERSONAL": "วัตถุประสงค์สินเชื่อส่วนบุคคล (Personal)",
    "installment_to_monthly_income": "ภาระค่างวดต่อรายได้ต่อเดือน",
    "disposable_income_est": "รายได้สุทธิคงเหลือหลังผ่อนชำระต่อเดือน",
    "disposable_income_ratio": "สัดส่วนรายได้สุทธิคงเหลือ (Disposable Ratio)",
    "interest_spread_over_grade": "ส่วนต่างดอกเบี้ยเทียบเกณฑ์เกรดสินเชื่อ",
    "person_home_ownership_RENT": "สถานะที่อยู่อาศัย: เช่า (Rent)",
    "person_home_ownership_OWN": "สถานะที่อยู่อาศัย: เจ้าของ (Own)",
    "person_home_ownership_OTHER": "สถานะที่อยู่อาศัย: อื่นๆ (Other)",
    "loan_int_rate": "อัตราดอกเบี้ยเงินกู้ (Interest Rate)",
    "cb_person_default_on_file": "เคยมีประวัติผิดนัดชำระ (Default on File)",
    "loan_amnt": "วงเงินสินเชื่อที่ขอ (Loan Amount)",
    "person_emp_length": "อายุงาน (Employment Length)",
    "compound_risk_flag": "ความเสี่ยงซ้ำ (เคยผิดนัด + เกรดต่ำ)",
    "cred_hist_to_age_ratio": "สัดส่วนประวัติเครดิตต่ออายุ",
    "loan_to_emp_ratio": "วงเงินต่ออายุงาน",
    "cb_person_cred_hist_length": "ระยะเวลาประวัติเครดิต (ปี)",
    "installment_monthly_est": "ประมาณการยอดผ่อนต่อเดือน",
    "person_age": "อายุผู้กู้ (Age)"
}

def get_thai_feature_name(col: str) -> str:
    return FEATURE_THAI_MAPPING.get(col, col)

def create_local_shap_waterfall_fig(shap_values_1d, feature_names, raw_input_dict, max_display=10):
    df_contrib = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_1d
    })
    
    df_contrib['abs_val'] = df_contrib['shap_value'].abs()
    df_top = df_contrib.sort_values(by='abs_val', ascending=False).head(max_display).copy()
    df_top = df_top.sort_values(by='shap_value', ascending=True)
    
    labels = [get_thai_feature_name(f) for f in df_top['feature']]
    df_top['label'] = labels
    
    colors = ['#EF4444' if v > 0 else '#10B981' for v in df_top['shap_value']]
    
    fig = go.Figure(go.Bar(
        x=df_top['shap_value'],
        y=df_top['label'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=1, color='rgba(255,255,255,0.2)')
        ),
        text=[f"{v:+.3f}" for v in df_top['shap_value']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title={
            'text': "<b>ปัจจัยสำคัญที่มีผลต่อคะแนนความเสี่ยง (SHAP Local Explanation)</b><br><span style='font-size:12px;color:#94a3b8;'>แท่งสีแดง: ดันความเสี่ยงให้สูงขึ้น | แท่งสีเขียว: ช่วยลดความเสี่ยงลง</span>",
            'x': 0.05,
            'xanchor': 'left'
        },
        xaxis_title="SHAP Value (อิทธิพลต่อผลการตัดสินใจ)",
        yaxis_title="",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans-serif", size=13, color="#f8fafc"),
        margin=dict(l=10, r=40, t=70, b=40),
        height=420,
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.08)',
            zerolinecolor='rgba(255,255,255,0.3)',
            zerolinewidth=2
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0)'
        )
    )
    return fig

def generate_plain_text_explanation(shap_values_1d, feature_names, raw_input_dict):
    df_contrib = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_1d
    })
    
    top_risk_increasers = df_contrib[df_contrib['shap_value'] > 0].sort_values(by='shap_value', ascending=False).head(3)
    top_risk_reducers = df_contrib[df_contrib['shap_value'] < 0].sort_values(by='shap_value', ascending=True).head(3)
    
    increaser_bullets = []
    for _, r in top_risk_increasers.iterrows():
        name = get_thai_feature_name(r['feature'])
        increaser_bullets.append(f"• **{name}** (เพิ่มความเสี่ยง +{r['shap_value']:.2f})")
        
    reducer_bullets = []
    for _, r in top_risk_reducers.iterrows():
        name = get_thai_feature_name(r['feature'])
        reducer_bullets.append(f"• **{name}** (ช่วยลดความเสี่ยง {r['shap_value']:.2f})")
        
    return {
        "risk_drivers": increaser_bullets,
        "protective_factors": reducer_bullets
    }
