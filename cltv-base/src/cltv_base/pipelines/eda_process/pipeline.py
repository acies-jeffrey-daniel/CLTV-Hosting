from kedro.pipeline import Pipeline, node
from .nodes import (
    generate_orders_eda_report,
    generate_transactions_eda_report,
    generate_behavioral_eda_report,
)

# --- Mandatory EDA (Orders & Transactions) ---
def create_mandatory_eda_pipeline(**kwargs) -> Pipeline:
    """Creates EDA pipeline for Orders and Transactions."""
    return Pipeline(
        [
            node(
                func=generate_orders_eda_report,
                inputs=["current_orders_data", "params:expected_orders_cols"],
                outputs="orders_eda_report",
                name="generate_orders_eda_report_node",
            ),
            node(
                func=generate_transactions_eda_report,
                inputs=["current_transactions_data", "params:expected_transaction_cols"],
                outputs="transactions_eda_report",
                name="generate_transactions_eda_report_node",
            ),
        ],
        tags="mandatory_eda"
    )

# --- Optional EDA (Behavioral) ---
def create_optional_behavioral_eda_pipeline(**kwargs) -> Pipeline:
    """Creates EDA pipeline for Behavioral data (runs only if data is present)."""
    return Pipeline(
        [
            node(
                func=generate_behavioral_eda_report,
                inputs=["current_behavioral_data", "params:expected_behavioral_cols"],
                outputs="behavioral_eda_report",
                name="generate_behavioral_eda_report_node",
            ),
        ],
        tags="behavioral_eda"
    )

# The original create_pipeline will now be a combination used in the registry
def create_pipeline(**kwargs) -> Pipeline:
    return create_mandatory_eda_pipeline() + create_optional_behavioral_eda_pipeline()