import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credit_risk.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_assessments (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        applicant_name TEXT,
        officer_name TEXT,
        person_age INTEGER,
        person_income REAL,
        person_home_ownership TEXT,
        person_emp_length REAL,
        loan_intent TEXT,
        loan_grade TEXT,
        loan_amnt REAL,
        loan_int_rate REAL,
        loan_percent_income REAL,
        cb_person_default_on_file TEXT,
        cb_person_cred_hist_length INTEGER,
        risk_score REAL,
        risk_tier TEXT,
        probability REAL,
        underwriting_decision TEXT,
        officer_notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_assessment(data: dict) -> str:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    assessment_id = data.get("id") or f"CR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    cursor.execute("""
    INSERT OR REPLACE INTO loan_assessments (
        id, created_at, applicant_name, officer_name, person_age, person_income,
        person_home_ownership, person_emp_length, loan_intent, loan_grade,
        loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file,
        cb_person_cred_hist_length, risk_score, risk_tier, probability,
        underwriting_decision, officer_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        assessment_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get("applicant_name", "Anonymous Applicant"),
        data.get("officer_name", "Credit Officer"),
        int(data.get("person_age", 30)),
        float(data.get("person_income", 50000)),
        str(data.get("person_home_ownership", "RENT")),
        float(data.get("person_emp_length", 3.0)),
        str(data.get("loan_intent", "PERSONAL")),
        str(data.get("loan_grade", "B")),
        float(data.get("loan_amnt", 10000)),
        float(data.get("loan_int_rate", 10.0)),
        float(data.get("loan_percent_income", 0.20)),
        str(data.get("cb_person_default_on_file", "N")),
        int(data.get("cb_person_cred_hist_length", 4)),
        float(data.get("risk_score", 50.0)),
        str(data.get("risk_tier", "Medium Risk")),
        float(data.get("probability", 0.5)),
        str(data.get("underwriting_decision", "Under Review")),
        str(data.get("officer_notes", ""))
    ))
    conn.commit()
    conn.close()
    return assessment_id

def get_recent_assessments(limit=50) -> pd.DataFrame:
    init_db()
    conn = get_connection()
    query = f"SELECT * FROM loan_assessments ORDER BY created_at DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_portfolio_kpis() -> dict:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*), AVG(risk_score), AVG(loan_amnt), SUM(loan_amnt) FROM loan_assessments")
    row = cursor.fetchone()
    total = row[0] or 0
    avg_score = row[1] or 0.0
    avg_loan = row[2] or 0.0
    total_loan = row[3] or 0.0
    
    cursor.execute("SELECT risk_tier, COUNT(*) FROM loan_assessments GROUP BY risk_tier")
    tier_counts = {r[0]: r[1] for r in cursor.fetchall()}
    
    cursor.execute("SELECT underwriting_decision, COUNT(*) FROM loan_assessments GROUP BY underwriting_decision")
    decision_counts = {r[0]: r[1] for r in cursor.fetchall()}
    
    conn.close()
    return {
        "total_assessments": total,
        "avg_risk_score": round(avg_score, 1),
        "avg_loan_amnt": round(avg_loan, 2),
        "total_volume": round(total_loan, 2),
        "tier_distribution": tier_counts,
        "decision_distribution": decision_counts
    }
