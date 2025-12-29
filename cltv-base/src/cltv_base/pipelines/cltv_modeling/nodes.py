# Solutions_CLTV_A2A\cltv-base\src\cltv_base\pipelines\cltv_modeling\nodes.py

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor
import numpy as np
from typing import Dict, Tuple # Added Dict import

def predict_cltv_bgf_ggf(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits BG/NBD and Gamma-Gamma models to predict 3-month CLTV (fixed horizon).
    This function acts as a Kedro node.
    """
    print(f"Predicting CLTV using BG/NBD + Gamma-Gamma models for 3 months (fixed horizon)...")
    df = transactions_df.copy()
    
    if not all(col in df.columns for col in ['Purchase Date', 'User ID', 'Total Amount']):
        raise KeyError("Missing required columns ('Purchase Date', 'User ID', 'Total Amount') in the transaction dataset for CLTV prediction.")

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    observation_period_end = df['Purchase Date'].max()
    summary_df = summary_data_from_transaction_data(
        df, 
        customer_id_col='User ID',
        datetime_col='Purchase Date',
        monetary_value_col='Total Amount',
        observation_period_end=observation_period_end
    )

    summary_df = summary_df[(summary_df['frequency'] > 0) & (summary_df['monetary_value'] > 0)]
    if summary_df.empty:
        print("Warning: No valid data for CLTV prediction after filtering. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'predicted_cltv_3m'])

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(summary_df['frequency'], summary_df['monetary_value'])

    summary_df['predicted_cltv_3m'] = ggf.customer_lifetime_value(
        bgf,
        summary_df['frequency'],
        summary_df['recency'],
        summary_df['T'],
        summary_df['monetary_value'],
        time=3,
        freq='D', 
        discount_rate=0.01
    )

    summary_df = summary_df.reset_index()
    summary_df['User ID'] = summary_df['User ID'].astype(str)
    
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df head:\n{summary_df.head()}")
    print(f"Diagnostic: predict_cltv_bgf_ggf - predicted_cltv_df null counts:\n{summary_df.isnull().sum()}")
    print(summary_df)
    return summary_df[['User ID', 'predicted_cltv_3m']]

def predict_cltv_xgboost(customers_df: pd.DataFrame, predicted_churn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Predict CLTV using XGBoost regression.
    Automatically adapts to user-uploaded data by selecting only available necessary features.
    Acts as a Kedro node.
    """
    print("Predicting CLTV using XGBoost...")

    df = customers_df.copy()

    df = df.merge(
        predicted_churn_probabilities[['User ID', 'predicted_churn_prob']],
        on='User ID',
        how='left'
    )

    df['predicted_churn_prob'] = df['predicted_churn_prob'].fillna(0)
    target_col = "Total Order Value"
    candidate_features = [
        'Total Unique Visits','Total Unique Sessions','Total Unique Devices',
        'Device Type','Channel','Geo Location',
        'Total Session Duration','Avg Session Duration','Total Page Views',
        'Avg Page Views','Total Bounces','Bounce Rate',
        'sessions_score','visits_score','pageviews_score','bounce_score','Engagement_Score', 'predicted_churn_prob'
    ]

    if target_col not in df.columns:
        raise KeyError(f"Missing required target column '{target_col}' in dataset.")

    available_features = [col for col in candidate_features if col in df.columns]
    print(f"Using features: {available_features}")

    X = df[available_features].copy()
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"XGBoost RMSE: {rmse}")

    df['cltv_xgboost'] = model.predict(X)
    return df[['User ID','cltv_xgboost']] if 'User ID' in df.columns else df[['cltv_xgboost']]

def predict_cltv_lightgbm(customers_df: pd.DataFrame, predicted_churn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Predict CLTV using LightGBM regression.
    Automatically adapts to user-uploaded data by selecting only available necessary features.
    Acts as a Kedro node.
    """
    print("Predicting CLTV using LightGBM...")

    df = customers_df.copy()
      # Assuming predicted_churn_probabilities has 'User ID' and 'predicted_churn_prob'
    df = df.merge(
        predicted_churn_probabilities[['User ID', 'predicted_churn_prob']],
        on='User ID',
        how='left'
    )
    df['predicted_churn_prob'] = df['predicted_churn_prob'].fillna(0)

    target_col = "Total Order Value"
    candidate_features = [
        'Total Unique Visits','Total Unique Sessions','Total Unique Devices',
        'Device Type','OS <lambda>','Channel','Geo Location',
        'Total Session Duration','Avg Session Duration','Total Page Views',
        'Avg Page Views','Total Bounces','Bounce Rate',
        'sessions_score','visits_score','pageviews_score','bounce_score','Engagement_Score', 'predicted_churn_prob'
    ]

    if target_col not in df.columns:
        raise KeyError(f"Missing required target column '{target_col}' in dataset.")

    available_features = [col for col in candidate_features if col in df.columns]
    print(f"Using features: {available_features}")

    X = df[available_features].copy()
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(objective='regression', n_estimators=300, learning_rate=0.05, max_depth=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"LightGBM RMSE: {rmse}")

    df['cltv_lightgbm'] = model.predict(X) 
    return df[['User ID','cltv_lightgbm']] if 'User ID' in df.columns else df[['cltv_lightgbm']]

def predict_cltv_catboost(customers_df: pd.DataFrame, predicted_churn_probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Predict CLTV using CatBoost regression.
    Handles categorical features automatically and cleans None/NaN values.
    """
    print("Predicting CLTV using CatBoost...")

    df = customers_df.copy()
    df = df.merge(predicted_churn_probabilities[['User ID', 'predicted_churn_prob']], on='User ID', how='left')
    df['predicted_churn_prob'] = df['predicted_churn_prob'].fillna(0)

    target_col = "Total Order Value"
    candidate_features = [
        'Total Unique Visits','Total Unique Sessions','Total Unique Devices',
        'Device Type','Channel','Geo Location',
        'Total Page Views','Avg Page Views','Total Bounces','Bounce Rate',
        'sessions_score','visits_score','pageviews_score','bounce_score','Engagement_Score','predicted_churn_prob'
    ]

    if target_col not in df.columns:
        raise KeyError(f"Missing required target column '{target_col}' in dataset.")

    available_features = [col for col in candidate_features if col in df.columns]
    print(f"Using features: {available_features}")

    # --- Clean data ---
    X = df[available_features].copy()
    y = df[target_col].copy()

    # Fill numeric nulls
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(0)

    # Fill object columns with 'Unknown'
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str).fillna("Unknown")

    # Drop rows where target is NaN
    valid_rows = y.notna()
    X, y = X[valid_rows], y[valid_rows]

    # Identify categorical features
    categorical_features = [col for col in X.select_dtypes(include=['object']).columns]
    print(f"Categorical features detected: {categorical_features}")

    # Split train-test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize CatBoost
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=categorical_features,
        verbose=False
    )

    # Fit model safely
    model.fit(X_train, y_train, cat_features=categorical_features)

    preds = model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"CatBoost RMSE: {rmse:.4f}")

    df['cltv_catboost'] = model.predict(X)
    df['cltv_catboost'] = df['cltv_catboost'].fillna(0)

    return df[['User ID','cltv_catboost']] if 'User ID' in df.columns else df[['cltv_catboost']]

def evaluate_cltv_models(
    true_cltv_df: pd.DataFrame,
    cltv_prediction_xgboost: pd.DataFrame,
    cltv_prediction_lightgbm: pd.DataFrame,
    cltv_prediction_catboost: pd.DataFrame,
) -> dict:
    """
    Evaluates and compares XGBoost, LightGBM, and CatBoost models based on their predictions.
    Returns RMSE, MAE, R² for each and identifies the best model.
    """
    print("Evaluating CLTV model performances...")

    metrics = {}
    models = {
        "xgboost": cltv_prediction_xgboost,
        "lightgbm": cltv_prediction_lightgbm,
        "catboost": cltv_prediction_catboost,
    }

    # Ensure common merge key
    true_df = true_cltv_df[["User ID", "CLTV"]]

    for model_name, pred_df in models.items():
        df = true_df.merge(pred_df, on="User ID", how="inner")

        # Pick predicted column name dynamically
        pred_col = [c for c in df.columns if c.startswith("cltv_") or "pred" in c.lower()][-1]
        
        y_true = df["CLTV"].values
        y_pred = df[pred_col].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[model_name] = {"rmse": rmse, "mae": mae, "r2": r2}

    # Pick best based on RMSE (lowest)
    best_model = min(metrics, key=lambda k: metrics[k]["rmse"])
    metrics["best_model"] = best_model

    print(f"Best CLTV model based on RMSE: {best_model.upper()}")
    return metrics

# ⭐ CRITICAL FIX: Add the definition for select_best_ml_cltv
def select_best_ml_cltv(
    cltv_prediction_xgboost: pd.DataFrame,
    cltv_prediction_lightgbm: pd.DataFrame,
    cltv_prediction_catboost: pd.DataFrame,
    cltv_model_comparison_metrics: dict
) -> pd.DataFrame:
    """
    Selects the best ML model's prediction and returns it using the standard
    output column name ('predicted_cltv_3m') so it can overwrite the BG/NBD result.
    """
    best_model_key = cltv_model_comparison_metrics.get("best_model", "xgboost")
    
    if best_model_key == 'xgboost':
        best_df = cltv_prediction_xgboost
        pred_col = 'cltv_xgboost'
    elif best_model_key == 'lightgbm':
        best_df = cltv_prediction_lightgbm
        pred_col = 'cltv_lightgbm'
    elif best_model_key == 'catboost':
        best_df = cltv_prediction_catboost
        pred_col = 'cltv_catboost'
    else:
        # Fallback to XGBoost if comparison failed
        best_df = cltv_prediction_xgboost
        pred_col = 'cltv_xgboost'
        
    if best_df.empty or pred_col not in best_df.columns or 'User ID' not in best_df.columns:
        print(f"[WARN] Failed to retrieve predictions from best model ({best_model_key}). Returning empty dataframe.")
        return pd.DataFrame(columns=['User ID', 'predicted_cltv_3m'])

    print(f"[INFO] Overwriting final predicted_cltv_df output with best ML model: {best_model_key.upper()}")
    
    # Rename the selected prediction column to the BG/NBD standard name
    final_cltv = best_df[['User ID', pred_col]].rename(
        columns={pred_col: 'predicted_cltv_3m'}
    )
    return final_cltv