import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced Feature Engineering Transformer for Institutional Credit Underwriting:
    - Interest rate imputation by loan grade
    - Employment length imputation
    - Installment payment approximation (36-month standard amortization)
    - Disposable income estimation and monthly burden ratio
    - Interest spread over loan grade benchmark
    - Credit history length to age ratio
    - Loan amount to employment tenure ratio
    - Compound risk flag (Prior Default + Low Loan Grade)
    """
    def __init__(self):
        self.grade_int_rates_ = {}
        self.overall_int_rate_median_ = 11.0
        self.emp_length_median_ = 4.0

    def fit(self, X, y=None):
        df = X.copy()
        if 'loan_grade' in df.columns and 'loan_int_rate' in df.columns:
            self.grade_int_rates_ = df.groupby('loan_grade')['loan_int_rate'].median().to_dict()
            self.overall_int_rate_median_ = float(df['loan_int_rate'].median())
        if 'person_emp_length' in df.columns:
            self.emp_length_median_ = float(df['person_emp_length'].median())
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Impute missing values with learned grade-specific medians
        if 'loan_int_rate' in df.columns:
            def impute_rate(row):
                if pd.isna(row['loan_int_rate']):
                    return self.grade_int_rates_.get(row.get('loan_grade'), self.overall_int_rate_median_)
                return row['loan_int_rate']
            df['loan_int_rate'] = df.apply(impute_rate, axis=1)
            
        if 'person_emp_length' in df.columns:
            df['person_emp_length'] = df['person_emp_length'].fillna(self.emp_length_median_)
            
        # 2. Recompute or verify loan_percent_income (DTI)
        if 'loan_amnt' in df.columns and 'person_income' in df.columns:
            safe_income = np.where(df['person_income'] <= 0, 1.0, df['person_income'])
            calculated_dti = df['loan_amnt'] / safe_income
            if 'loan_percent_income' not in df.columns or df['loan_percent_income'].isnull().any():
                df['loan_percent_income'] = calculated_dti
            else:
                df['loan_percent_income'] = df['loan_percent_income'].fillna(calculated_dti)
                
        # 3. Monthly Installment Approximation (36-month loan term amortization)
        n_months = 36
        monthly_rate = (df['loan_int_rate'] / 100.0) / 12.0
        numerator = monthly_rate * ((1 + monthly_rate) ** n_months)
        denominator = ((1 + monthly_rate) ** n_months) - 1
        denominator = np.where(denominator == 0, 1e-6, denominator)
        df['installment_monthly_est'] = df['loan_amnt'] * (numerator / denominator)
        
        monthly_income = df['person_income'] / 12.0
        safe_monthly_income = np.where(monthly_income <= 0, 1.0, monthly_income)
        df['installment_to_monthly_income'] = df['installment_monthly_est'] / safe_monthly_income
        
        # 4. Disposable Income & Disposable Ratio
        df['disposable_income_est'] = monthly_income - df['installment_monthly_est']
        df['disposable_income_ratio'] = df['disposable_income_est'] / safe_monthly_income
        
        # 5. Interest spread over grade benchmark
        def calc_spread(row):
            grade_med = self.grade_int_rates_.get(row.get('loan_grade'), self.overall_int_rate_median_)
            return row['loan_int_rate'] - grade_med
        df['interest_spread_over_grade'] = df.apply(calc_spread, axis=1)
        
        # 6. Credit history length to age ratio
        safe_age = np.where(df['person_age'] <= 0, 20.0, df['person_age'])
        df['cred_hist_to_age_ratio'] = df['cb_person_cred_hist_length'] / safe_age
        
        # 7. Loan amount per year of employment tenure
        df['loan_to_emp_ratio'] = df['loan_amnt'] / (df['person_emp_length'] + 1.0)
        
        # 8. Compound risk flag: Prior default AND lower loan grade (D, E, F, G)
        is_prior_default = df['cb_person_default_on_file'].astype(str).str.upper() == 'Y'
        is_lower_grade = df['loan_grade'].isin(['D', 'E', 'F', 'G'])
        df['compound_risk_flag'] = (is_prior_default & is_lower_grade).astype(int)
        
        return df

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unrealistic outliers found during EDA:
    - person_age > 100
    - person_emp_length > 60
    """
    cleaned = df.copy()
    cleaned = cleaned[cleaned['person_age'] <= 100]
    cleaned = cleaned[(cleaned['person_emp_length'].isnull()) | (cleaned['person_emp_length'] <= 60)]
    return cleaned

def build_full_pipeline():
    numeric_features = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'installment_monthly_est',
        'installment_to_monthly_income',
        'disposable_income_est',
        'disposable_income_ratio',
        'interest_spread_over_grade',
        'cred_hist_to_age_ratio',
        'loan_to_emp_ratio',
        'compound_risk_flag'
    ]
    
    nominal_features = ['person_home_ownership', 'loan_intent']
    binary_features = ['cb_person_default_on_file']
    ordinal_features = ['loan_grade']
    
    grade_categories = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]
    binary_categories = [['N', 'Y']]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features),
            ('bin', OrdinalEncoder(categories=binary_categories, handle_unknown='use_encoded_value', unknown_value=-1), binary_features),
            ('ord', OrdinalEncoder(categories=grade_categories, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('engineer', CreditFeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline, numeric_features, nominal_features, binary_features, ordinal_features

def get_feature_names(pipeline, numeric_features, nominal_features, binary_features, ordinal_features):
    ct = pipeline.named_steps['preprocessor']
    nom_encoder = ct.named_transformers_['nom']
    nom_cols = list(nom_encoder.get_feature_names_out(nominal_features))
    return numeric_features + nom_cols + binary_features + ordinal_features
