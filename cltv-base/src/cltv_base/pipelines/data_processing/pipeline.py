from kedro.pipeline import Pipeline, node
from .nodes import (
    standardize_columns_advanced, 
    convert_data_types, 
    merge_orders_transactions, 
    aggregate_behavioral_customer_level, 
    aggregate_orders_transactions_customer_level, 
    merge_customer_ord_txn_behavioral_data,
    cap_outliers_by_quantile 
)

# --- 1. Mandatory Data Processing (Orders/Transactions) ---
def create_mandatory_data_pipeline() -> Pipeline:
    """
    Creates the base data processing pipeline for Orders and Transactions,
    PLUS the Behavioral Standardization and Aggregation steps to ensure the required 
    intermediate datasets exist (even if empty/MemoryDataSet).
    """
    return Pipeline(
        [
            node(
                func=standardize_columns_advanced, 
                inputs=["current_orders_data", "ssot_mapping_file", "params:orders_df_name", "params:data_standardisation_params"],
                outputs=["orders_standardized", "orders_mapping_report"], 
                name="standardize_orders_columns_advanced",
            ),
            node(
                func=standardize_columns_advanced, 
                inputs=["current_transactions_data", "ssot_mapping_file", "params:transactions_df_name", "params:data_standardisation_params"],
                outputs=["transactions_standardized", "transactions_mapping_report"], 
                name="standardize_transactions_columns_advanced",
            ),
            # ⭐ MANDATORY STEP 3: Standardization of Behavioral Data
            node(
                func=standardize_columns_advanced, 
                inputs=["current_behavioral_data", "ssot_mapping_file", "params:behavioral_df_name", "params:data_standardisation_params"],
                outputs=["behavioral_standardized", "behavioral_mapping_report"], 
                name="standardize_behavioral_columns_advanced",
            ),
            # ⭐ MANDATORY STEP 4: Aggregation of Behavioral Data (Crucial to produce MemoryDataSet output)
            node(
                func=aggregate_behavioral_customer_level,
                inputs=["behavioral_typed"], # behavioral_typed is output by create_conversion_and_txn_aggregation_pipeline
                outputs="customer_aggregated_behavioral_data",
                name="customer_aggregated_behavioral_data_node"
            ),
        ]
    )

# --- 2. Optional Behavioral Data Processing ---
def create_optional_behavioral_pipeline() -> Pipeline:
    """Creates the data processing pipeline for Behavioral data (runs only if data is present)."""
    return Pipeline(
        [
            node(
                func=standardize_columns_advanced, 
                inputs=["current_behavioral_data", "ssot_mapping_file", "params:behavioral_df_name", "params:data_standardisation_params"],
                outputs=["behavioral_standardized", "behavioral_mapping_report"], 
                name="standardize_behavioral_columns_advanced",
            ),
            node(
                func=aggregate_behavioral_customer_level,
                inputs=["behavioral_typed"], # behavioral_typed will be the output of create_conversion_pipeline
                outputs="customer_aggregated_behavioral_data",
                name="customer_aggregated_behavioral_data_node"
            ),
        ],
        tags="behavioral_processing" # Tag for easy identification
    )

# --- 3. Shared Conversion/Aggregation (Orders/Transactions only) ---
def create_conversion_and_txn_aggregation_pipeline() -> Pipeline:
    """Converts types and aggregates Orders/Transactions data."""
    return Pipeline([
        node(
            func=convert_data_types,
            # Inputs: orders/transactions from mandatory, behavioral is passed from the environment (loaded as empty df if missing)
            inputs=["orders_standardized", "transactions_standardized","behavioral_standardized"],
            outputs=["orders_typed", "transactions_typed","behavioral_typed"],
            name="convert_raw_data_types",
        ),
        node(
            func=merge_orders_transactions,
            inputs=["orders_typed", "transactions_typed"],
            outputs="orders_merged_with_user_id",
            name="merge_orders_and_transactions",
        ),
        node(
            func=aggregate_orders_transactions_customer_level,
            inputs=["orders_merged_with_user_id"],
            outputs="customer_aggregated_orders_transaction_data_pre_cap", 
            name="customer_aggregated_orders_transaction_node"
        ),
    ])

# --- 4. Final Merge (Conditional on Capping) ---

def create_final_merge_normal_pipeline() -> Pipeline:
    """Final merge path for Orders/Transactions data (UNCAPPED)."""
    return Pipeline([
        node(
            func=lambda x: x,
            inputs="customer_aggregated_orders_transaction_data_pre_cap",
            outputs="customer_aggregated_orders_transaction_data", 
            name="bypass_capping_identity_node"
        ),
        node(
            func=merge_customer_ord_txn_behavioral_data,
            inputs=[
                "customer_aggregated_orders_transaction_data",
                "customer_aggregated_behavioral_data" # Loaded as empty if behavioral data is missing
            ],
            outputs="customer_level_merged_data",
            name="customer_level_merged_data_node"
        ),
    ], tags="normal_merge")

def create_final_merge_capped_pipeline() -> Pipeline:
    """Final merge path for Orders/Transactions data (CAPPED)."""
    return Pipeline([
        node(
            func=cap_outliers_by_quantile,
            inputs=[
                "customer_aggregated_orders_transaction_data_pre_cap",
                "params:capping_columns",
                "params:capping_multiplier",
                "params:capping_quantile"
            ],
            outputs=[
                "customer_aggregated_orders_transaction_data",
                "orders_txn_capping_values"
            ],
            name="cap_orders_transactions_outliers_node"
        ),
        node(
            func=merge_customer_ord_txn_behavioral_data,
            inputs=[
                "customer_aggregated_orders_transaction_data",
                "customer_aggregated_behavioral_data"
            ],
            outputs="customer_level_merged_data",
            name="customer_level_merged_data_node"
        ),
    ], tags="capped_merge")
    
# Export the main creation functions for use in pipeline_registry
def create_pipeline(**kwargs) -> Pipeline:
    # Default pipeline creation must be explicit for Kedro CLI. 
    # We will use the specific variants in the registry.
    raise NotImplementedError("Use create_pipeline_ot_only, create_pipeline_ot_normal, or create_pipeline_otb_normal/capped variants.")

# Placeholder for the final pipelines (used by the registry)
def create_pipeline_ot_only() -> Pipeline:
    return create_mandatory_data_pipeline() + create_conversion_and_txn_aggregation_pipeline()

def create_pipeline_ot_normal() -> Pipeline:
    return create_pipeline_ot_only() + create_final_merge_normal_pipeline()

def create_pipeline_ot_capped() -> Pipeline:
    return create_pipeline_ot_only() + create_final_merge_capped_pipeline()

def create_pipeline_otb_normal() -> Pipeline:
    return create_pipeline_ot_only() + create_optional_behavioral_pipeline() + create_final_merge_normal_pipeline()

def create_pipeline_otb_capped() -> Pipeline:
    return create_pipeline_ot_only() + create_optional_behavioral_pipeline() + create_final_merge_capped_pipeline()