import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lifelines import CoxPHFitter

METRICS_SUPPORTED = 'rfm_score'

# Distribution-based Threshold Calculation
def calculate_all_distribution_thresholds(customer_df: pd.DataFrame, metrics: str = None) -> float:
    
    if metrics is None:
        metrics = 'rfm_score'
    thresholds = {}
    for metric in metrics:
        if customer_df.empty or metric not in customer_df.columns:
            print(f"[WARN] No data or '{metric}' missing. Using threshold=0")
            thresholds[f"{metric}_threshold"] = 0
        else:
            values = customer_df[metric]
            lower = values.quantile(0.10)
            upper = values.quantile(0.90)
            filtered = values[(values >= lower) & (values <= upper)]
            threshold_value = float(filtered.quantile(0.25))
            print(f"[INFO] {metric} threshold calculated: {threshold_value:.2f}")
            thresholds = threshold_value
    return thresholds

# User-Defined Value Threshold
def calculate_user_value_threshold(metric: str, user_value: float) -> dict:
    
    print(f"[INFO] User set {metric} threshold: {user_value:.2f}")
    return {f"{metric}_threshold": user_value}

# ML-Based Threshold Calculation (Stub for future)
def calculate_ml_based_threshold(customer_df: pd.DataFrame, metric: str) -> dict:
    
    print(f"[WARN] ML threshold for '{metric}' not implemented. Using default value = 0.")
    return {f"{metric}_threshold": 0}

# Get Customers At Risk 
def get_customers_at_risk(customer_df: pd.DataFrame) -> pd.DataFrame:
    
    df = customer_df[customer_df['is_churned'] == 1]
    return df

# Label Churned Customers (for any metric)

def label_churned_customers(customer_df: pd.DataFrame, metric: str, inactive_days_threshold: float) -> pd.DataFrame:
    
    df = customer_df.copy()
    metric_name = metric
    threshold_value = inactive_days_threshold

    df['is_churned'] = (df[metric_name] < threshold_value).astype(int)
        
    churned_count = df['is_churned'].sum()
    print(f"[INFO] Labeled {churned_count} customers as churned by '{metric_name}' with the threshold {threshold_value}.")
    print("columns name", df.columns)
    return df

# Feature & Label Extraction, Model Training, Prediction — (unchanged from your code)

def get_churn_features_labels(customer_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'User ID' not in customer_df.columns:
        raise ValueError("'User ID' column is required.")
    df = customer_df.set_index('User ID', drop=False)
    feature_cols = [
        'frequency', 'monetary', 'aov',
        'avg_days_between_orders', 'CLTV_30d', 'CLTV_60d', 'CLTV_90d', 'recency', 'is_churned'
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        print(f"[WARN] Missing features: {set(feature_cols) - set(available_cols)}")
    X = df[available_cols]
    y = df[['is_churned']]
    print(X.columns, y.columns)
    return X, y

def train_churn_prediction_model(
    X: pd.DataFrame, y: pd.DataFrame, n_estimators: int, random_state: int
) -> Tuple[RandomForestClassifier, Dict, List, pd.DataFrame, pd.DataFrame]:
    
    if X.empty or y.empty:
        print("[WARN] Empty features or labels. Returning dummy model.")
        dummy_model = RandomForestClassifier(n_estimators=1, random_state=random_state)
        return dummy_model, {}, [], pd.DataFrame(), pd.DataFrame()
    
    # Check if the target variable has more than one unique class.
    # If not, the model cannot be trained for binary classification.
    if y.nunique().max() < 2:
        print("[WARN] The target variable 'y' contains only a single class. The model will not be able to predict churn.")
        dummy_model = RandomForestClassifier(n_estimators=1, random_state=random_state)
        return dummy_model, {}, [], X, y
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train.values.ravel())
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    importances = model.feature_importances_.tolist()
    return model, report, importances, X_test, y_test

def predict_churn_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn_prob'])

    try:
        probs = model.predict_proba(X)[:, 1]

    except IndexError:
        print("Warning: Model was trained on a single class. Predicting probabilities accordingly.")
        
        if model.classes_[0] == 1:
            probs = model.predict_proba(X)[:, 0]
        else:
            probs = np.zeros(len(X))

    predictions = pd.DataFrame(data={
        'User ID': X.index,  
        'predicted_churn_prob': probs
    })

    return predictions

def assign_predicted_churn_labels(predicted_churn_prob: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if predicted_churn_prob.empty:
        return pd.DataFrame(columns=['User ID', 'predicted_churn'])
    predicted_churn_prob = predicted_churn_prob.copy()
    predicted_churn_prob['predicted_churn'] = (
        predicted_churn_prob['predicted_churn_prob'] >= threshold
    ).astype(int)
    return predicted_churn_prob[['User ID', 'predicted_churn']]

def train_cox_survival_model(customer_df: pd.DataFrame, feature_col: List[str]) -> Tuple[CoxPHFitter, pd.DataFrame]:
    df = customer_df.copy()
    
    required_cols = feature_col + ['lifespan_1d', 'is_churned']
    if df.empty or not all(col in df.columns for col in required_cols):
        print("[WARN] Missing survival model columns.")
        return CoxPHFitter(), df
    cph = CoxPHFitter()
    cph.fit(df[required_cols], duration_col='lifespan_1d', event_col='is_churned')
    df['expected_active_days'] = cph.predict_expectation(df[feature_col]).round(0).astype(int)
    #df.to_csv("expected_active_days.csv", index=False)
    print(df.columns)
    return cph, df
