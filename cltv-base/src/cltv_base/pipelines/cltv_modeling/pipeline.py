# Solutions_CLTV_A2A\cltv-base\src\cltv_base\pipelines\cltv_modeling\pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    predict_cltv_bgf_ggf,
    predict_cltv_xgboost,
    predict_cltv_lightgbm,
    predict_cltv_catboost,
    evaluate_cltv_models,
    select_best_ml_cltv  # <-- ASSUMING YOU ADDED THIS NODE FUNCTION
)

# --- 1. Mandatory CLTV (BG/NBD + GG - required for churn features) ---
def create_bgf_ggf_pipeline() -> Pipeline:
    """CLTV modeling using BG/NBD and Gamma-Gamma (mandatory baseline)."""
    return Pipeline(
        [
            node(
                func=predict_cltv_bgf_ggf,
                inputs="transactions_typed",
                outputs="predicted_cltv_df",
                name="predict_cltv_bgf_ggf",
            ),
        ],
        tags="bgf_ggf_cltv"
    )

# --- 2. Optional CLTV (ML models - runs only if behavioral data is present) ---
def create_ml_cltv_pipeline() -> Pipeline:
    """CLTV modeling using ML regressors (optional, requires behavioral data)."""
    return Pipeline(
        [
            node(
                func=predict_cltv_xgboost,
                inputs=["customer_level_merged_data_engagement_score", "predicted_churn_probabilities"],
                outputs="cltv_prediction_xgboost",
                name="cltv_prediction_xgboost",
            ),
            node(
                func=predict_cltv_lightgbm,
                inputs=["customer_level_merged_data_engagement_score", "predicted_churn_probabilities"],
                outputs="cltv_prediction_lightgbm",
                name="cltv_prediction_lightgbm",
            ),
            node(
                func=predict_cltv_catboost,
                inputs=["customer_level_merged_data_engagement_score", "predicted_churn_probabilities"],
                outputs="cltv_prediction_catboost",
                name="cltv_prediction_catboost",
            ),
            node(
                func=evaluate_cltv_models,
                inputs=[
                    "historical_cltv_customers",
                    "cltv_prediction_xgboost",
                    "cltv_prediction_lightgbm",
                    "cltv_prediction_catboost",
                ],
                outputs="cltv_model_comparison_metrics",
                name="evaluate_cltv_models_node",
            ),
            # ⭐ CRITICAL OVERWRITE NODE: This node overwrites the BG/NBD output
            # with the best ML prediction, ensuring ML is the final CLTV result 
            # in the OTB path, and fixing the loading crash.
            node(
                func=select_best_ml_cltv,
                inputs=[
                    "cltv_prediction_xgboost",
                    "cltv_prediction_lightgbm",
                    "cltv_prediction_catboost",
                    "cltv_model_comparison_metrics"
                ],
                outputs="predicted_cltv_df", 
                name="select_best_ml_cltv_and_overwrite",
            )
        ],
        tags="ml_cltv_modeling"
    )

def create_pipeline(**kwargs) -> Pipeline:
    return create_bgf_ggf_pipeline() + create_ml_cltv_pipeline()