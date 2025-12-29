# Solutions_CLTV_A2A\cltv-base\src\cltv_base\pipeline_registry.py

"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, node

# --- Import sub-pipelines ---
from cltv_base.pipelines.data_processing.pipeline import (
    create_mandatory_data_pipeline,
    create_optional_behavioral_pipeline,
    create_conversion_and_txn_aggregation_pipeline,
    create_final_merge_normal_pipeline,
    create_final_merge_capped_pipeline,
)
from cltv_base.pipelines.customer_features.pipeline import (
    create_mandatory_features_pipeline,
    create_optional_behavioral_features_pipeline
)
from cltv_base.pipelines.cltv_modeling.pipeline import (
    create_bgf_ggf_pipeline,
    create_ml_cltv_pipeline
)
from cltv_base.pipelines.churn_modeling import pipeline as churn_modeling_module
from cltv_base.pipelines.customer_migration import pipeline as customer_migration_module
from cltv_base.pipelines.ui_data_preparation.pipeline import (
    create_mandatory_ui_pipeline,
    create_optional_ml_ui_pipeline
)
from cltv_base.pipelines.eda_process.pipeline import (
    create_mandatory_eda_pipeline,
    create_optional_behavioral_eda_pipeline
)

# --- Import shared nodes ---
from cltv_base.nodes import combine_final_customer_data


# ---------------------------------------------------------------------
# Helper: combine all model outputs for UI / final dataset
# ---------------------------------------------------------------------
def _create_combine_final_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=combine_final_customer_data,
                inputs=[
                    "historical_cltv_customers",
                    "predicted_churn_probabilities",
                    "predicted_churn_labels",
                    "cox_predicted_active_days",
                    "predicted_cltv_df",
                ],
                outputs="final_rfm_cltv_churn_data",
                name="combine_final_customer_data_for_ui",
            )
        ]
    )


# ---------------------------------------------------------------------
# Core Pipeline Components
# ---------------------------------------------------------------------

# Mandatory core analysis components (RFM, Churn, Migration, UI Prep, O/T Data Proc, Final Merge)
CORE_MANDATORY_PIPELINE = (
    create_mandatory_data_pipeline()
    + create_conversion_and_txn_aggregation_pipeline()
    + create_mandatory_features_pipeline()
    + churn_modeling_module.create_pipeline() 
    + _create_combine_final_pipeline()
    + create_mandatory_ui_pipeline()
    + customer_migration_module.create_pipeline()
)

# Optional components (Behavioral data aggregation, Engagement score, ML CLTV models, ML UI prep)
OPTIONAL_BEHAVIORAL_PIPELINE = (
    create_optional_behavioral_pipeline()
    + create_optional_behavioral_features_pipeline()
    + create_ml_cltv_pipeline()
    + create_optional_ml_ui_pipeline()
)

# ---------------------------------------------------------------------
# User Journey Pipelines (CLTV selection is now conditional)
# ---------------------------------------------------------------------

def register_pipelines() -> Dict[str, Pipeline]:
    """Register all project pipelines."""
    
    # 1. Orders/Transactions Only (OT-ONLY)
    # The MemoryDataSet configuration in catalog.yml handles the missing files.
    
    # OT-ONLY: Normal Run (No Capping)
    pipeline_ot_normal = (
        CORE_MANDATORY_PIPELINE
        + create_bgf_ggf_pipeline()          
        + create_final_merge_normal_pipeline()
        + create_mandatory_eda_pipeline()
    ).tag("ot_normal")

    # OT-ONLY: Capped Run (With Capping)
    pipeline_ot_capped = (
        CORE_MANDATORY_PIPELINE
        + create_bgf_ggf_pipeline()          
        + create_final_merge_capped_pipeline()
        + create_mandatory_eda_pipeline()
    ).tag("ot_capped")

    # 2. Orders/Transactions/Behavioral (OTB-FULL)
    
    # OTB-FULL: Normal Run (No Capping)
    pipeline_otb_normal = (
        CORE_MANDATORY_PIPELINE
        + OPTIONAL_BEHAVIORAL_PIPELINE       
        + create_final_merge_normal_pipeline()
        + create_mandatory_eda_pipeline()
        + create_optional_behavioral_eda_pipeline()
    ).tag("otb_normal")

    # OTB-FULL: Capped Run (With Capping)
    pipeline_otb_capped = (
        CORE_MANDATORY_PIPELINE
        + OPTIONAL_BEHAVIORAL_PIPELINE       
        + create_final_merge_capped_pipeline()
        + create_mandatory_eda_pipeline()
        + create_optional_behavioral_eda_pipeline()
    ).tag("otb_capped")

    return {
        "eda_only": create_mandatory_eda_pipeline() + create_optional_behavioral_eda_pipeline(),
        "pipeline_ot_normal": pipeline_ot_normal,
        "pipeline_ot_capped": pipeline_ot_capped,
        "pipeline_otb_normal": pipeline_otb_normal,
        "pipeline_otb_capped": pipeline_otb_capped,
        "__default__": pipeline_ot_normal,
    }