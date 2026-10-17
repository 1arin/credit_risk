import pandas as pd
import numpy as np

def evaluate_model_fairness(sample_df: pd.DataFrame, preprocessor, model):
    """
    Evaluates algorithmic bias, demographic parity, and ethical fairness:
    Analyzes whether the model treats different demographic age tiers equitably.
    """
    df = sample_df.copy()
    
    # Define demographic groups
    conditions = [
        df['person_age'] < 25,
        (df['person_age'] >= 25) & (df['person_age'] <= 40),
        df['person_age'] > 40
    ]
    choices = ['Young (<25 yrs)', 'Core Career (25-40 yrs)', 'Mature (>40 yrs)']
    df['age_group'] = np.select(conditions, choices, default='Core Career (25-40 yrs)')
    
    X_trans = preprocessor.transform(df)
    probs = model.predict_proba(X_trans)[:, 1]
    df['risk_score'] = probs * 100.0
    df['approved'] = (df['risk_score'] <= 50.0).astype(int)
    
    groups = df['age_group'].unique()
    group_stats = []
    
    # Baseline group: Core Career
    core_df = df[df['age_group'] == 'Core Career (25-40 yrs)']
    core_approval_rate = core_df['approved'].mean() if not core_df.empty else 0.80
    
    for g in choices:
        sub = df[df['age_group'] == g]
        if sub.empty:
            continue
            
        n = len(sub)
        app_rate = sub['approved'].mean()
        avg_score = sub['risk_score'].mean()
        avg_income = sub['person_income'].mean()
        avg_dti = sub['loan_percent_income'].mean()
        
        # Disparate Impact Ratio (DIR) relative to core group
        dir_ratio = app_rate / core_approval_rate if core_approval_rate > 0 else 1.0
        
        # Four-fifths rule check (EEOC guidelines: DIR should be >= 0.80)
        is_compliant = dir_ratio >= 0.80
        
        group_stats.append({
            "demographic_tier": g,
            "sample_size": n,
            "approval_rate": round(float(app_rate) * 100, 2),
            "avg_risk_score": round(float(avg_score), 2),
            "avg_income": round(float(avg_income), 2),
            "avg_dti": round(float(avg_dti) * 100, 2),
            "disparate_impact_ratio": round(float(dir_ratio), 3),
            "fairness_compliant": "✓ Pass (≥ 0.80)" if is_compliant else "⚠️ Flag (< 0.80)"
        })
        
    return pd.DataFrame(group_stats)
