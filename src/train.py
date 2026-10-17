import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss
)
import shap

from src.preprocessing import clean_raw_data, build_full_pipeline, get_feature_names

def find_cost_optimal_threshold(y_true, y_prob, cost_fn=5.0, cost_fp=1.0):
    """
    Computes business-optimal classification threshold.
    In Banking: False Negative (default uncaught) is 5x to 8x more costly
    than False Positive (rejecting a creditworthy customer).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        costs.append((th, total_cost, fn, fp, tp, tn))
        
    best_th, min_cost, opt_fn, opt_fp, opt_tp, opt_tn = min(costs, key=lambda x: x[1])
    
    # Compare with default threshold 0.50
    default_pred = (y_prob >= 0.50).astype(int)
    d_tn, d_fp, d_fn, d_tp = confusion_matrix(y_true, default_pred).ravel()
    default_cost = (d_fn * cost_fn) + (d_fp * cost_fp)
    
    return {
        "optimal_threshold": round(float(best_th), 2),
        "optimal_cost": float(min_cost),
        "default_threshold": 0.50,
        "default_cost": float(default_cost),
        "cost_savings_pct": round(float((default_cost - min_cost) / default_cost) * 100, 2),
        "optimal_false_negatives": int(opt_fn),
        "default_false_negatives": int(d_fn)
    }

def train_and_evaluate():
    print("=" * 65)
    print("Institutional AI-Driven Credit Risk Underwriting Pipeline")
    print("=" * 65)
    
    # 1. Load Data
    data_path = os.path.join("data", "credit_risk_dataset.csv")
    if not os.path.exists(data_path):
        fallback = os.path.join("credit_risk_dataset.csv", "credit_risk_dataset.csv")
        data_path = fallback
            
    print(f"[1/7] Ingesting dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    cleaned_df = clean_raw_data(raw_df)
    print(f"      Initial: {raw_df.shape[0]:,} records | After Audit & Outlier Removal: {cleaned_df.shape[0]:,} records")
    
    # 2. Stratified Train-Test Split
    X = cleaned_df.drop(columns=['loan_status'])
    y = cleaned_df['loan_status']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[2/7] Stratified Partitioning: {X_train_raw.shape[0]:,} Train | {X_test_raw.shape[0]:,} Test")
    
    # 3. Fit End-to-End Pipeline
    print("[3/7] Building Advanced Feature Engineering & Scaling Pipeline...")
    pipeline, num_cols, nom_cols, bin_cols, ord_cols = build_full_pipeline()
    
    X_train_trans = pipeline.fit_transform(X_train_raw)
    X_test_trans = pipeline.transform(X_test_raw)
    feature_names = get_feature_names(pipeline, num_cols, nom_cols, bin_cols, ord_cols)
    print(f"      Total Engineered Feature Dimensionality: {len(feature_names)}")
    
    # 4. SMOTE Balancing (Section 9.5 Proposal)
    print("[4/7] Applying SMOTE Resampling for Class Imbalance Mitigation...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_trans, y_train)
    print(f"      Balanced Training Volume: {X_train_smote.shape[0]:,} observations")
    
    # 5. Train Candidate Algorithms
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    curves_data = {}
    trained_models = {}
    
    print("\n[5/7] Executing Multi-Model Benchmark:")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train_smote, y_train_smote)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(auc), 4),
            "f1_score": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": fpr[::20].tolist(),
            "tpr": tpr[::20].tolist(),
            "precision_curve": p_curve[::20].tolist(),
            "recall_curve": r_curve[::20].tolist()
        }
        print(f"      ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | False Negatives: {fn}")
        
    champion_model = trained_models["XGBoost"]
    
    # 6. Cost-Sensitive Threshold Analysis
    print("\n[6/7] Computing Bank Cost-Sensitive Optimal Underwriting Threshold...")
    xgb_probs = champion_model.predict_proba(X_test_trans)[:, 1]
    threshold_analysis = find_cost_optimal_threshold(y_test, xgb_probs, cost_fn=5.0, cost_fp=1.0)
    print(f"      Default Threshold: 0.50 -> Optimal Threshold: {threshold_analysis['optimal_threshold']}")
    print(f"      False Negatives reduced from {threshold_analysis['default_false_negatives']} down to {threshold_analysis['optimal_false_negatives']}")
    print(f"      Projected Expected Loss Reduction: {threshold_analysis['cost_savings_pct']}%")
    
    # 7. Explainable AI with SHAP
    print("\n[7/7] Generating SHAP TreeExplainer & Feature Attributions...")
    sample_indices = np.random.RandomState(42).choice(X_test_trans.shape[0], size=500, replace=False)
    background_data = X_test_trans[sample_indices]
    
    explainer = shap.TreeExplainer(champion_model)
    shap_values = explainer(background_data)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)
    top_feature_dict = top_feature_importance.to_dict(orient='records')
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, os.path.join("models", "preprocessor.joblib"))
    joblib.dump(trained_models["Logistic Regression"], os.path.join("models", "logreg_model.joblib"))
    joblib.dump(trained_models["Random Forest"], os.path.join("models", "rf_model.joblib"))
    joblib.dump(champion_model, os.path.join("models", "best_model.joblib"))
    joblib.dump(explainer, os.path.join("models", "shap_explainer.joblib"))
    
    sample_test_df = X_test_raw.iloc[sample_indices].copy()
    sample_test_df['actual_default'] = y_test.iloc[sample_indices].values
    sample_test_df['predicted_prob'] = champion_model.predict_proba(background_data)[:, 1]
    sample_test_df.to_csv(os.path.join("models", "sample_test_cases.csv"), index=False)
    
    eval_payload = {
        "metrics": results,
        "curves": curves_data,
        "feature_names": feature_names,
        "feature_importance": top_feature_dict,
        "threshold_analysis": threshold_analysis,
        "summary": {
            "total_samples": len(raw_df),
            "cleaned_samples": len(cleaned_df),
            "train_size": len(X_train_raw),
            "test_size": len(X_test_raw),
            "champion_model": "XGBoost",
            "champion_auc": results["XGBoost"]["roc_auc"],
            "champion_recall": results["XGBoost"]["recall"]
        }
    }
    
    with open(os.path.join("models", "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2, ensure_ascii=False)
        
    print("\nSUCCESS: All models and banking underwriting benchmarks successfully generated!")

if __name__ == "__main__":
    train_and_evaluate()
