import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime
from datetime import timedelta


def format_date_with_ordinal(date):
    if pd.isna(date):
        return "N/A"
    day = int(date.strftime('%d'))
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date.strftime('%B %Y')}"


def prepare_kpi_data(orders_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    
    print("Preparing KPI data...")
    kpis = {}

    # Total Revenue 
    if 'Total Amount' in transactions_df.columns:
        kpis['total_revenue'] = float(transactions_df['Total Amount'].sum())
    else:
        kpis['total_revenue'] = 0.0

    # CLTV, AOV, Avg Txns/User 
    if not rfm_segmented_df.empty and 'aov' in rfm_segmented_df.columns and 'CLTV' in rfm_segmented_df.columns and 'frequency' in rfm_segmented_df.columns:
        kpis['avg_cltv'] = float(rfm_segmented_df['CLTV'].mean())
        kpis['avg_aov'] = float(rfm_segmented_df['aov'].mean())
        kpis['avg_txns_per_user'] = float(rfm_segmented_df['frequency'].mean())
    else:
        kpis['avg_cltv'] = 0.0
        kpis['avg_aov'] = 0.0
        kpis['avg_txns_per_user'] = 0.0
        print("Warning: Missing RFM columns in segmented data or empty DataFrame for CLTV, AOV, Avg Txns/User KPIs.")

    # Data Timeframe 
    if 'Purchase Date' in transactions_df.columns:
        start_dt = pd.to_datetime(transactions_df['Purchase Date'], dayfirst=True).min()
        end_dt = pd.to_datetime(transactions_df['Purchase Date'], dayfirst=True).max()
        kpis['start_date'] = format_date_with_ordinal(start_dt)
        kpis['end_date'] = format_date_with_ordinal(end_dt)
    else:
        kpis['start_date'] = "N/A"
        kpis['end_date'] = "N/A"
        print("Warning: Missing 'Purchase Date' in transactions data for Data Timeframe KPI.")

    kpis['total_customers'] = int(len(rfm_segmented_df)) 
    if 'Transaction ID' in transactions_df.columns:
        kpis['total_orders'] = transactions_df['Transaction ID'].count()
    kpi_df = pd.DataFrame([kpis])
    return kpi_df

def prepare_segment_summary_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    print("Preparing segment summary data...")
    if rfm_segmented_df.empty or 'segment' not in rfm_segmented_df.columns:
        print("Warning: 'segment' column not found or empty rfm_segmented_df. Cannot prepare segment summary.")
        return pd.DataFrame()

    segment_summary = rfm_segmented_df.groupby("segment").agg({
        "aov": "mean",
        "CLTV": "mean",
        "frequency": "mean",
        "avg_days_between_orders": "mean",
        "recency": "mean",
        "monetary": "mean"
    }).round(2)
    print(segment_summary)
    return segment_summary

def prepare_segment_counts_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the distribution of customers across RFM segments for the pie chart.
    """
    print("Preparing segment counts data...")
    if rfm_segmented_df.empty or 'segment' not in rfm_segmented_df.columns:
        print("Warning: 'segment' column not found or empty rfm_segmented_df. Cannot prepare segment counts.")
        return pd.DataFrame(columns=['Segment', 'Count'])
    
    segment_counts = rfm_segmented_df['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    return segment_counts


def prepare_top_products_by_segment_data(orders_df: pd.DataFrame, transactions_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    
    print("Preparing top products by segment data...")
    all_possible_segments = ['Champions', 'Potential Champions', 'Recent Customers', 
                'Customers Needing Attention', 'At Risk', 'About to Sleep', 'Lost']
    
    # Initialize with both columns
    top_products_by_segment = {segment: pd.DataFrame(columns=['product_id', 'Total_Quantity', 'Total_Revenue']) for segment in all_possible_segments}

    orders_for_products = orders_df.copy()
    orders_for_products.columns = [c.strip().replace(" ", "_").lower() for c in orders_for_products.columns]
    if 'unit_price' in orders_for_products.columns:
        orders_for_products.rename(columns={'unit_price': 'unitprice'}, inplace=True)
    
    # Ensure both quantity and unitprice are available for revenue calculation
    required_product_cols = {'transaction_id', 'product_id', 'quantity', 'unitprice'} 
    if orders_for_products.empty or not required_product_cols.issubset(set(orders_for_products.columns)):
        print(f"Warning: Missing required columns for top products: {required_product_cols - set(orders_for_products.columns)} or empty orders data. Returning empty dict for all segments.")
        return top_products_by_segment 

    if 'segment' not in rfm_segmented_df.columns or 'User ID' not in rfm_segmented_df.columns:
        print("Warning: RFM segmented data or User ID not available for top product calculation. Returning empty dict for all segments.")
        return top_products_by_segment

    for segment in all_possible_segments:
        segment_users = rfm_segmented_df[rfm_segmented_df['segment'] == segment]['User ID']
        
        if segment_users.empty:
            print(f"Info: No users found for segment '{segment}'. Skipping top product calculation for this segment.")
            continue 

        # Ensure 'User ID' in transactions_df is string for consistent merging
        transactions_df['User ID'] = transactions_df['User ID'].astype(str)
        segment_transaction_ids = transactions_df[transactions_df['User ID'].isin(segment_users)]['Transaction ID']

        filtered_orders = orders_for_products[orders_for_products['transaction_id'].isin(segment_transaction_ids)].copy()
        if not filtered_orders.empty:
            filtered_orders['revenue'] = filtered_orders['quantity'] * filtered_orders['unitprice']
            top_products = (
                filtered_orders.groupby('product_id')
                .agg(Total_Quantity=('quantity', 'sum'), Total_Revenue=('revenue', 'sum')) 
                .sort_values(by='Total_Quantity', ascending=False)
                .head(5)
                .reset_index()
            )
            top_products_by_segment[segment] = top_products
        else:
            print(f"Info: No orders found for segment '{segment}'.")


    return top_products_by_segment

def prepare_predicted_cltv_display_data(rfm_segmented_df: pd.DataFrame, predicted_cltv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for the predicted CLTV display table.
    """
    print("Preparing predicted CLTV display data...")
    if rfm_segmented_df.empty or predicted_cltv_df.empty or 'User ID' not in rfm_segmented_df.columns or 'User ID' not in predicted_cltv_df.columns:
        print("Warning: Missing 'User ID' or empty DataFrames for merging CLTV prediction data. Returning empty DataFrame.")
        return pd.DataFrame(columns=['User ID', 'segment', 'CLTV', 'predicted_cltv_3m'])

    # Merge predicted CLTV into the RFM segmented data
    rfm_segmented_df = rfm_segmented_df.merge(predicted_cltv_df, on='User ID', how='left')
    rfm_segmented_df['predicted_cltv_3m'] = rfm_segmented_df['predicted_cltv_3m'].fillna(0)
    
    return rfm_segmented_df[['User ID', 'segment', 'CLTV', 'predicted_cltv_3m']].sort_values(by='predicted_cltv_3m', ascending=False).reset_index(drop=True)

def prepare_cltv_comparison_data(predicted_cltv_display_data: pd.DataFrame) -> pd.DataFrame:
    
    print("Preparing CLTV comparison data...")
    if predicted_cltv_display_data.empty or not all(col in predicted_cltv_display_data.columns for col in ['segment', 'CLTV', 'predicted_cltv_3m']):
        print("Warning: Missing required CLTV comparison columns or empty DataFrame. Cannot prepare comparison data.")
        return pd.DataFrame()

    segment_comparison = predicted_cltv_display_data.groupby('segment')[['CLTV', 'predicted_cltv_3m']].mean().reset_index()
    segment_melted = segment_comparison.melt(
        id_vars='segment',
        value_vars=['CLTV', 'predicted_cltv_3m'],
        var_name='CLTV Type',
        value_name='Average CLTV')

    segment_order = ['Champions', 'Potential Champions', 'Recent Customers', 
                'Customers Needing Attention', 'At Risk', 'About to Sleep', 'Lost', 'Unclassified']
    segment_melted['segment'] = pd.Categorical(segment_melted['segment'], categories=segment_order, ordered=True)
    return segment_melted.sort_values(by='segment')


def calculate_realization_curve_data(transactions_df: pd.DataFrame, rfm_segmented_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    print("Calculating realization curve data...")
    
    all_possible_segments = ['Champions', 'Potential Champions', 'Recent Customers', 
                'Customers Needing Attention', 'At Risk', 'About to Sleep', 'Lost', 'Unclassified']
    realization_data = {
        "Overall Average": pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User"]),
        "All Segments": pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User", "Segment"])
    }
    for segment in all_possible_segments:
        realization_data[segment] = pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User"])
    
    df = transactions_df.copy()

    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], dayfirst=True)
    df['revenue'] = df['Total Amount']

    

    segment_options = {
        "Overall Average": rfm_segmented_df['User ID'].unique(),
    }
    
    for seg_name in all_possible_segments:
        segment_options[seg_name] = rfm_segmented_df[rfm_segmented_df['segment'] == seg_name]['User ID'].unique()


    intervals = [15, 30, 45, 60, 90] 
    
    all_segments_data_list = [] 

    for option_name, selected_users in segment_options.items():
        if len(selected_users) == 0:
            
            continue 

        # Filter the full orders data by segment users
        filtered_df_by_user = df[df['User ID'].isin(selected_users)]
        user_count = filtered_df_by_user['User ID'].nunique()

        if user_count == 0:
            continue

        # The start date for the curve calculation should be the earliest purchase date within the filtered data for this segment
        start_date = filtered_df_by_user['Purchase Date'].min()
        if pd.isna(start_date):
            continue

        cltv_values = []

        for days in intervals:
            cutoff = start_date + pd.Timedelta(days=days)
            
            revenue = filtered_df_by_user[
                (filtered_df_by_user['Purchase Date'] <= cutoff)
            ]['revenue'].sum()
            avg_cltv = revenue / user_count
            cltv_values.append(round(avg_cltv, 2))

        segment_curve_df = pd.DataFrame({
            "Period (Days)": intervals,
            "Avg CLTV per User": cltv_values
        })
        realization_data[option_name] = segment_curve_df


        if option_name in all_possible_segments: 
            segment_curve_df['Segment'] = option_name 
            all_segments_data_list.append(segment_curve_df)

    
    if all_segments_data_list:
        realization_data["All Segments"] = pd.concat(all_segments_data_list, ignore_index=True)
    else:
        realization_data["All Segments"] = pd.DataFrame(columns=["Period (Days)", "Avg CLTV per User", "Segment"])

    return realization_data

def prepare_churn_summary_data(rfm_segmented_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for churn summary metrics and expected active days by segment.
    """
    print("Preparing churn summary data...")
    if rfm_segmented_df.empty or not all(col in rfm_segmented_df.columns for col in ['predicted_churn', 'predicted_churn_prob', 'expected_active_days', 'segment']):
        print("Warning: Missing churn prediction or segment columns or empty DataFrame for churn summary. Returning empty DataFrames.")
        missing_cols = [col for col in ['predicted_churn', 'predicted_churn_prob', 'expected_active_days', 'segment'] if col not in rfm_segmented_df.columns]
        print(f"Diagnostic: prepare_churn_summary_data - Missing columns: {missing_cols}")
        print(f"Diagnostic: prepare_churn_summary_data - Columns available: {rfm_segmented_df.columns.tolist()}")
        return pd.DataFrame(), pd.DataFrame()

    churn_by_segment = (
        rfm_segmented_df
        .groupby("segment")['predicted_churn_prob']
        .mean()
        .reset_index()
        .rename(columns={'predicted_churn_prob': 'Avg Churn Probability'})
    )

    active_days = (
        rfm_segmented_df
        .groupby("segment")['expected_active_days']
        .mean()
        .reset_index()
        .rename(columns={'expected_active_days': 'Avg Expected Active Days'})
    )
    return churn_by_segment, active_days

def prepare_churn_detailed_view_data(rfm_segmented_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares detailed churn analysis data for display.
    """
    print("Preparing detailed churn view data...")
    print(rfm_segmented_df.info())
    required_cols = ['User ID', 'segment', 'predicted_cltv_3m', 'predicted_churn_prob', 'predicted_churn', 'expected_active_days']
    if rfm_segmented_df.empty or not all(col in rfm_segmented_df.columns for col in required_cols):
        print("Warning: Missing required columns or empty DataFrame for detailed churn view. Returning empty DataFrame.")
        missing_cols = [col for col in required_cols if col not in rfm_segmented_df.columns]
        print(f"Diagnostic: prepare_churn_detailed_view_data - Missing columns: {missing_cols}")
        print(f"Diagnostic: prepare_churn_detailed_view_data - Columns available: {rfm_segmented_df.columns.tolist()}")
        return pd.DataFrame(columns=required_cols)
        print(rfm_segmented_df.columns)
    
    else:
        not_churn_mask = rfm_segmented_df['predicted_churn'] == 0
        rfm_segmented_df.loc[not_churn_mask, 'expected_next_purchase'] = (rfm_segmented_df.loc[not_churn_mask].apply(
    lambda row: row['last_purchase'] + timedelta(days=row['avg_days_between_orders']),
    axis=1
)).dt.date
        required_cols = ['User ID', 'segment', 'predicted_cltv_3m', 'predicted_churn_prob', 'predicted_churn', 'expected_active_days', 'expected_next_purchase']
    return rfm_segmented_df[required_cols].sort_values(by='predicted_churn_prob', ascending=False).copy()

def prepare_ml_cltv_display_data(
    rfm_segmented_df: pd.DataFrame,
    cltv_prediction_xgboost: pd.DataFrame,
    cltv_prediction_lightgbm: pd.DataFrame,
    cltv_prediction_catboost: pd.DataFrame   # <-- Added CatBoost input
) -> dict:
    """
    Prepares predicted CLTV data from ML models for display in the UI.

    Args:
        rfm_segmented_df: DataFrame containing 'User ID' and 'segment'.
        cltv_prediction_xgboost: DataFrame containing 'User ID' and 'cltv_xgboost'.
        cltv_prediction_lightgbm: DataFrame containing 'User ID' and 'cltv_lightgbm'.
        cltv_prediction_catboost: DataFrame containing 'User ID' and 'cltv_catboost'.

    Returns:
        A dictionary containing DataFrames for the three prediction models.
    """
    print("Preparing ML CLTV display data...")

    # 1. Merge XGBoost predictions
    xgb_display = rfm_segmented_df[['User ID', 'segment']].merge(
        cltv_prediction_xgboost, 
        on='User ID', 
        how='inner'
    )

    # 2. Merge LightGBM predictions
    lgbm_display = rfm_segmented_df[['User ID', 'segment']].merge(
        cltv_prediction_lightgbm, 
        on='User ID', 
        how='inner'
    )

    # 3. Merge CatBoost predictions
    cat_display = rfm_segmented_df[['User ID', 'segment']].merge(
        cltv_prediction_catboost, 
        on='User ID', 
        how='inner'
    )

    return {
        "xgboost": xgb_display,
        "lightgbm": lgbm_display,
        "catboost": cat_display
    }
