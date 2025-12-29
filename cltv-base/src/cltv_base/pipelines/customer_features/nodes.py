import pandas as pd
from typing import Tuple, List
import numpy as np
from typing import Optional, List

def safe_qcut(s: pd.Series, q: int = 5, labels: Optional[List[int]] = None, ascending: bool = True) -> pd.Series:
    """
    Safely assigns quantile scores, handling cases with insufficient unique values.
    'ascending=False' is used for Recency.
    """
    if labels is None:
        labels = list(range(1, q + 1))
    
    unique_values = s.dropna().nunique()
    
    
    if unique_values < q:
        bins = min(unique_values, q)
        if bins < 2:
            return pd.Series([int(np.ceil(q / 2))] * len(s), index=s.index, dtype=object)
        
        # Use pd.cut as a fallback, which is more tolerant of duplicate values
        scores = pd.cut(s, bins=bins, labels=labels[:bins], include_lowest=True)
        
        # Recency (needs special handling)
        if not ascending:
            scores = scores.map({label: labels[q-1-i] for i, label in enumerate(scores.cat.categories)})
            return scores.astype(object)
            
        return scores.astype(object)
    
    # Normal quantile assignment using pd.qcut
    if ascending:
        ranks = s.rank(method='first', ascending=True)
        scores = pd.qcut(ranks, q, labels=labels).astype(object)
    else:
        # For recency, we need to rank in descending order and then apply qcut
        ranks = s.rank(method='first', ascending=False)
        scores = pd.qcut(ranks, q, labels=labels).astype(object)
    
    return scores

#7 Bucket Segments
def assign_segment(row):
    """
    Assigns a customer segment based on R and FM scores.
    """
    r = row['r_score']
    fm = row['fm_score']

    if r >= 4 and fm == 5:
        return 'Champions'
    elif r >= 4 and fm >= 4:
        return 'Potential Champions'
    elif (r >= 4 and fm == 3) or (r == 3 and fm == 4):
        return 'Customers Needing Attention'
    elif r >= 4 and fm <= 2:
        return 'Activated/Reactived'
    elif r <= 3 and fm >= 3:
        return 'Loyal Lapsers'
    elif (r <= 3 and fm <= 2) or (r <= 2 and fm <= 2):
        return 'About to Sleep'
    elif r <= 2 and fm <= 2:
        return 'Lost'
    else:
        return 'Unclassified'

def calculate_customer_level_features(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes customer-level RFM and other derived features.
    This function acts as a Kedro node.
    (This part remains unchanged from your original code)
    """
    print("Calculating customer level features...")
    if not all(col in transactions_df.columns for col in ['Purchase Date', 'Total Amount', 'User ID']):
        raise ValueError("transactions_df must contain 'Purchase Date', 'Total Amount', and 'User ID' columns.")
    
    transactions_df['Purchase Date'] = pd.to_datetime(transactions_df['Purchase Date'])

    today = transactions_df['Purchase Date'].max() + pd.Timedelta(days=1)
    print(today)
    customer_level = transactions_df.groupby('User ID').agg(
        recency=('Purchase Date', lambda x: (today - x.max()).days),
        frequency=('Purchase Date', 'count'),
        monetary=('Total Amount', 'sum'),
        last_purchase=('Purchase Date', 'max'),
        first_purchase=('Purchase Date', 'min')
    ).reset_index()
    
    
    customer_level['User ID'] = customer_level['User ID'].astype(str)

    customer_level['aov'] = round(customer_level['monetary'] / customer_level['frequency'], 2)
    
    customer_level['avg_days_between_orders'] = (
        (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days / 
        (customer_level['frequency'] - 1).replace(0, pd.NA)
    )
    valid_avg = customer_level['avg_days_between_orders'][customer_level['avg_days_between_orders'].notna() & (customer_level['avg_days_between_orders'] != float('inf'))]
    median_gap = valid_avg.median() if not valid_avg.empty else 0
    customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].replace([float('inf'), -float('inf')], None)
    customer_level['avg_days_between_orders'] = customer_level['avg_days_between_orders'].fillna(median_gap).round(0).astype(int)
    #customer_level['avg_days_between_orders'].to_csv("days")
    customer_level['lifespan_1d'] = (customer_level['last_purchase'] - customer_level['first_purchase']).dt.days + 1
    customer_level['lifespan_7d'] = round(customer_level['lifespan_1d'] / 7, 2)
    customer_level['lifespan_15d'] = round(customer_level['lifespan_1d'] / 15, 2)
    customer_level['lifespan_30d'] = round(customer_level['lifespan_1d'] / 30, 2)
    customer_level['lifespan_60d'] = round(customer_level['lifespan_1d'] / 60, 2)
    customer_level['lifespan_90d'] = round(customer_level['lifespan_1d'] / 90, 2)
    
    customer_level['CLTV_1d'] = round(customer_level['monetary'] / customer_level['lifespan_1d'].replace(0, 1), 2)
    customer_level['CLTV_7d'] = round(customer_level['monetary'] / customer_level['lifespan_7d'].replace(0, 0.1), 2)
    customer_level['CLTV_15d'] = round(customer_level['monetary'] / customer_level['lifespan_15d'].replace(0, 0.1), 2)
    customer_level['CLTV_30d'] = round(customer_level['monetary'] / customer_level['lifespan_30d'].replace(0, 0.1), 2)
    customer_level['CLTV_60d'] = round(customer_level['monetary'] / customer_level['lifespan_60d'].replace(0, 0.1), 2)
    customer_level['CLTV_90d'] = round(customer_level['monetary'] / customer_level['lifespan_90d'].replace(0, 0.1), 2)
    customer_level['CLTV_total'] = customer_level['monetary']
    
    return customer_level


def perform_rfm_segmentation(customer_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs RFM segmentation on customer-level data using quantiles for scoring.
    This function acts as a Kedro node.
    """
    print("Performing RFM segmentation using quantiles...")
    df = customer_level_df.copy()
    
    if df.empty:
        print("Warning: customer_level_df is empty. Cannot perform RFM segmentation. Returning empty DataFrame.")
        return pd.DataFrame(columns=[
            'User ID', 'recency', 'frequency', 'monetary', 'last_purchase', 
            'first_purchase', 'aov', 'avg_days_between_orders', 'lifespan_1d',
            'lifespan_7d', 'lifespan_15d', 'lifespan_30d', 'lifespan_60d',
            'lifespan_90d', 'CLTV_1d', 'CLTV_7d', 'CLTV_15d', 'CLTV_30d',
            'CLTV_60d', 'CLTV_90d', 'CLTV_total', 'r_score', 'f_score',
            'm_score', 'fm_score', 'rfm_segment', 'CLTV'
        ])

    required_rfm_cols = ['recency', 'frequency', 'monetary']
    if not all(col in df.columns for col in required_rfm_cols):
        print(f"Warning: Missing RFM columns {set(required_rfm_cols) - set(df.columns)}. Cannot perform RFM segmentation. Returning original DataFrame.")
        return df
    
    # ⭐ CRITICAL FIX: DIRECT MANUAL RANKING FOR RECENCY (Lowest days -> Highest score)
    
    # 1. Rank the 'recency' column ascendingly (lowest days = rank 1)
    #    We use np.float64 to ensure compatibility with pd.qcut ranking
    df['r_rank'] = df['recency'].rank(ascending=True, method='first').astype(np.float64) 
    
    # 2. Use qcut on the Rank, with reversed labels [5, 4, 3, 2, 1]
    #    The lowest ranks (best recency) get the highest score (5).
    df['r_score'] = pd.qcut(
        df['r_rank'],
        q=5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int) 
    
    # Frequency and Monetary scores remain ascending (1->5) using the helper
    df['f_score'] = safe_qcut(df['frequency'], q=5, labels=list(range(1, 6)), ascending=True).astype(int)
    df['m_score'] = safe_qcut(df['monetary'], q=5, labels=list(range(1, 6)), ascending=True).astype(int)
    
    
    # --- Ensure all scores are numeric for FM calculation ---
    # Convert back to numeric, using 'ignore' as they should already be int/float from the qcut calls above
    df['r_score'] = pd.to_numeric(df['r_score'], errors='coerce')
    df['f_score'] = pd.to_numeric(df['f_score'], errors='coerce')
    df['m_score'] = pd.to_numeric(df['m_score'], errors='coerce')

    # Calculate FM Score (average of F and M scores, rounded to nearest integer)
    df['fm_score'] = ((df['f_score'] + df['m_score']) / 2).round().astype(int)
    df['rfm_score'] = ((df['r_score'] + df['f_score'] + df['m_score'])).round().astype(int)
    
    # Apply custom segmentation
    df['segment'] = df.apply(assign_segment, axis=1)

    return df

def calculate_historical_cltv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates historical CLTV based on AOV and frequency.
    This function acts as a Kedro node.
    (This part remains unchanged from your original code)
    """
    print("Calculating historical CLTV...")
    if df.empty or 'aov' not in df.columns or 'frequency' not in df.columns:
        print("Warning: Missing 'aov' or 'frequency' or empty DataFrame for historical CLTV. Returning original DataFrame.")
        if 'CLTV' not in df.columns:
            df['CLTV'] = 0.0
        return df
    df['CLTV'] = df['aov'] * df['frequency']
    return df

def calculate_engagement_score(ads_df: pd.DataFrame, weights: Optional[dict] = None, q: int = 5) -> pd.DataFrame:
    """
    Calculates engagement score for customers based on visits, sessions,
    page views, and bounce rate.
    
    Parameters
    ----------
    ads_df : pd.DataFrame
        Input DataFrame containing at least the columns:
        ['Total Unique Sessions', 'Total Unique Visits', 'Total Page Views', 'Bounce Rate'].
    weights : dict, optional
        Weights for each metric. Default: equal weights.
        Example: {'sessions': 0.25, 'visits': 0.25, 'page_views': 0.3, 'bounce_rate': 0.2}
    q : int, default=5
        Number of quantile buckets for scoring.
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added score columns and Engagement Score.
    """
    
    required_cols = ['Total Unique Sessions', 'Total Unique Visits', 'Total Page Views', 'Bounce Rate']
    missing = [c for c in required_cols if c not in ads_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for engagement scoring: {missing}")
    
    df = ads_df.copy()
    
    # Default equal weights if not provided
    if weights is None:
        weights = {
            'sessions': 0.25,
            'visits': 0.25,
            'page_views': 0.25,
            'bounce_rate': 0.25
        }
    
    # Quantile-based scoring (1–5)
    df['sessions_score'] = safe_qcut(df['Total Unique Sessions'], q=q, labels=list(range(1, q+1)), ascending=True)
    df['visits_score'] = safe_qcut(df['Total Unique Visits'], q=q, labels=list(range(1, q+1)), ascending=True)
    df['pageviews_score'] = safe_qcut(df['Total Page Views'], q=q, labels=list(range(1, q+1)), ascending=True)
    df['bounce_score'] = safe_qcut(df['Bounce Rate'], q=q, labels=list(range(1, q+1)), ascending=False)  # lower is better
    
    # Convert scores to numeric
    for col in ['sessions_score', 'visits_score', 'pageviews_score', 'bounce_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3)  # fallback to neutral if NA
    
    # Weighted engagement score scaled to 0–100
    df['Engagement_Score'] = (
        df['sessions_score'] * weights['sessions'] +
        df['visits_score'] * weights['visits'] +
        df['pageviews_score'] * weights['page_views'] +
        df['bounce_score'] * weights['bounce_rate']
    )
    
    # Normalize to 0–100
    df['Engagement_Score'] = (df['Engagement_Score'] / q) * 100
    return df
