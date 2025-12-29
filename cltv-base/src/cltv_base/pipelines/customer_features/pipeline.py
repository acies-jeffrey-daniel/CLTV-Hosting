from kedro.pipeline import Pipeline, node
from .nodes import (
    calculate_customer_level_features,
    perform_rfm_segmentation,
    calculate_historical_cltv,
    calculate_engagement_score
)

# --- 1. Mandatory RFM/Historical CLTV Features ---
def create_mandatory_features_pipeline() -> Pipeline:
    """Creates pipeline for calculating core RFM features and historical CLTV."""
    return Pipeline(
        [
            node(
                func=calculate_customer_level_features,
                inputs="transactions_typed",
                outputs="customer_level_features",
                name="calculate_customer_level_features",
            ),
            node(
                func=perform_rfm_segmentation,
                inputs="customer_level_features",
                outputs="rfm_segmented_df",
                name="perform_rfm_segmentation",
            ),
            node(
                func=calculate_historical_cltv,
                inputs="rfm_segmented_df",
                outputs="historical_cltv_customers",
                name="calculate_historical_cltv",
            ),
            # ⭐ ADDED: DUMMY NODE TO ENSURE ENGAGEMENT FEATURE OUTPUT EXISTS
            node(
                func=lambda df: df,
                inputs="customer_level_merged_data", # Input is the combined (O/T only) data
                outputs="customer_level_merged_data_engagement_score_DUMMY", # New output
                name="create_dummy_engagement_score_feature"
            ),
        ],
        tags="mandatory_features"
    )

# --- 2. Optional Behavioral Features (Engagement Score) ---
def create_optional_behavioral_features_pipeline() -> Pipeline:
    """Creates pipeline for calculating the Engagement Score (optional)."""
    return Pipeline(
        [
            node(
                func=calculate_engagement_score,
                inputs="customer_level_merged_data",
                outputs="customer_level_merged_data_engagement_score", 
                name="customer_level_data_with_engagement_score"
            )
        ],
        tags="optional_features"
    )

def create_pipeline(**kwargs) -> Pipeline:
    return create_mandatory_features_pipeline() + create_optional_behavioral_features_pipeline()