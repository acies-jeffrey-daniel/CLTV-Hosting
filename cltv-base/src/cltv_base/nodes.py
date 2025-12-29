import pandas as pd
from typing import Dict

def combine_final_customer_data(
    historical_cltv_customers: pd.DataFrame,
    predicted_churn_probabilities: pd.DataFrame,
    predicted_churn_labels: pd.DataFrame,
    cox_predicted_active_days: pd.DataFrame,
    predicted_cltv_df: pd.DataFrame 
) -> pd.DataFrame:
    """
    Combines various customer-related dataframes into a single final dataframe
    for UI consumption.
    """
    print("Combining final customer data for UI...")
    
    # Start with historical_cltv_customers as the base
    final_df = historical_cltv_customers.copy()

    # Merge predicted churn probabilities
    if not predicted_churn_probabilities.empty and 'User ID' in predicted_churn_probabilities.columns:
        final_df = final_df.merge(
            predicted_churn_probabilities[['User ID', 'predicted_churn_prob']],
            on='User ID',
            how='left'
        )
        print(final_df.columns)
    else:
        print("Warning: predicted_churn_probabilities is empty or missing 'User ID'. Skipping merge.")
        final_df['predicted_churn_prob'] = None

    # Merge predicted churn labels
    if not predicted_churn_labels.empty and 'User ID' in predicted_churn_labels.columns:
        final_df = final_df.merge(
            predicted_churn_labels[['User ID', 'predicted_churn']],
            on='User ID',
            how='left'
        )
    else:
        print("Warning: predicted_churn_labels is empty or missing 'User ID'. Skipping merge.")
        final_df['predicted_churn'] = None

    # Merge Cox predicted active days
    if not cox_predicted_active_days.empty and 'User ID' in cox_predicted_active_days.columns:
        final_df = final_df.merge(
            cox_predicted_active_days[['User ID', 'expected_active_days']],
            on='User ID',
            how='left'
        )
    else:
        print("Warning: cox_predicted_active_days is empty or missing 'User ID'. Skipping merge.")
        final_df['expected_active_days'] = None 

    # Merge predicted CLTV (newly added)
    if not predicted_cltv_df.empty and 'User ID' in predicted_cltv_df.columns:
        final_df = final_df.merge(
            predicted_cltv_df[['User ID', 'predicted_cltv_3m']],
            on='User ID',
            how='left'
        )
    else:
        print("Warning: predicted_cltv_df is empty or missing 'User ID'. Skipping merge.")
        final_df['predicted_cltv_3m'] = None 

    # Fill any remaining NaNs for numerical columns that were merged
    for col in ['predicted_churn_prob', 'predicted_churn', 'expected_active_days', 'predicted_cltv_3m']:
        if col in final_df.columns:
            if final_df[col].dtype == 'object': # Handle cases where None might make it object
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)
            else:
                final_df[col] = final_df[col].fillna(0)
    
    print(f"Final combined customer data shape: {final_df.shape}")
    print(final_df['predicted_churn'].value_counts())
    return final_df
