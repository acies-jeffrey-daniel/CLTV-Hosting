import pandas as pd
from typing import Dict, Any, List, Tuple
import numpy as np

ID_KEYWORDS = [
    '_id', 'id', 'user_id', 'customer_id', 'transaction_id', 'order_item_id',
    'product_id', 'coupon_id', 'session_id', 'action_id', 
    'revenue_listing_id', 'cost_id', 'campaign_id', 'visit_id', 'device_id', 'cookie_id' # Expanded from your lists
]

EXCLUDED_COLUMNS = [
    'customer_name', 'email', 'product_name', 'title', 'description', 
    'page_url', 'entry_page', 'exit_page', 'page_title', 'geo_location', 'campaign_name'
]

# Helper functions (Copied from data_processing to ensure self-contained EDA checks)
def _auto_map_column(column_list, candidate_names):
    import difflib
    for name in candidate_names:
        match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.75)
        if match:
            return match[0]
    return None

def _get_standardized_column_name(col, expected_mapping):
    """Tries to find the standard name for a given column based on mapping."""
    for standard_name, candidates in expected_mapping.items():
        if col in candidates:
            return standard_name
    return col

def _generate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates basic descriptive statistics for numeric columns."""
    stats = {}
    for col in df.select_dtypes(include=np.number).columns:
        stats[col] = {
            'count': int(df[col].count()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            '25%': float(df[col].quantile(0.25)),
            '50%': float(df[col].quantile(0.5)),
            '75%': float(df[col].quantile(0.75)),
            'max': float(df[col].max()),
        }
    return stats

def _generate_null_report(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates a report on missing values for all columns."""
    report = []
    total_rows = len(df)
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            report.append({
                'column': col,
                'type': str(df[col].dtype),
                'null_count': int(null_count),
                'null_percent': round((null_count / total_rows) * 100, 2)
            })
    return report

def _generate_cardinality_report(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates a report on high-cardinality categorical columns."""
    report = []
    cardinality_threshold = 50 # Flag columns with more than 50 unique non-numeric values
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_count = df[col].nunique(dropna=True)
        if unique_count > cardinality_threshold:
            report.append({
                'column': col,
                'unique_count': int(unique_count),
                'top_5_values': df[col].value_counts(dropna=True).head(5).index.tolist()
            })
    return report

# --- Main Kedro Nodes for EDA ---

def generate_orders_eda_report(orders_df: pd.DataFrame, expected_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generates EDA report for the orders dataset."""
    df = orders_df.copy()
    print("Generating EDA report for Orders data...")
    
    # Standardize column names for better reporting clarity
    column_map = {}
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
    df = df.rename(columns=column_map)

    report = {
        'total_rows': len(df),
        'columns': df.columns.tolist(),
        'null_report': _generate_null_report(df),
        'descriptive_stats': _generate_descriptive_stats(df),
        'cardinality_report': _generate_cardinality_report(df)
    }

    # 2. Add plotting data (NEW)
    report['plot_data'] = _generate_plot_data(df, ID_KEYWORDS, EXCLUDED_COLUMNS)
    
    return report

def generate_transactions_eda_report(transactions_df: pd.DataFrame, expected_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generates EDA report for the transactions dataset."""
    df = transactions_df.copy()
    print("Generating EDA report for Transactions data...")
    
    # Standardize column names for better reporting clarity
    column_map = {}
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
    df = df.rename(columns=column_map)

    report = {
        'total_rows': len(df),
        'columns': df.columns.tolist(),
        'null_report': _generate_null_report(df),
        'descriptive_stats': _generate_descriptive_stats(df),
        'cardinality_report': _generate_cardinality_report(df)
    }

    report['plot_data'] = _generate_plot_data(df, ID_KEYWORDS, EXCLUDED_COLUMNS)
    
    return report

def generate_behavioral_eda_report(behavioral_df: pd.DataFrame, expected_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
    """Generates EDA report for the behavioral dataset."""
    df = behavioral_df.copy()
    # ⭐ ADDED: Check for empty DataFrame (meaning file was missing)
    if df.empty:
        print("[INFO] Behavioral data is empty (file likely missing). Skipping EDA report generation.")
        return {
            'total_rows': 0,
            'columns': [],
            'null_report': [],
            'descriptive_stats': {},
            'cardinality_report': [],
            'plot_data': {'numerical_histograms': {}, 'time_series_daily_counts': {}}
        }
    
    print("Generating EDA report for Behavioral data...")
    
    # Standardize column names for better reporting clarity
    column_map = {}
    for standard_name, candidates in expected_mapping.items():
        mapped_col = _auto_map_column(df.columns.tolist(), candidates)
        if mapped_col:
            column_map[mapped_col] = standard_name
    df = df.rename(columns=column_map)

    report = {
        'total_rows': len(df),
        'columns': df.columns.tolist(),
        'null_report': _generate_null_report(df),
        'descriptive_stats': _generate_descriptive_stats(df),
        'cardinality_report': _generate_cardinality_report(df)
    }

    report['plot_data'] = _generate_plot_data(df, ID_KEYWORDS, EXCLUDED_COLUMNS)
    return report

# --- ADD new helper functions in nodes.py ---
def _generate_plot_data(df: pd.DataFrame, id_keywords: List[str], excluded_columns: List[str]) -> Dict[str, Any]:
    """Generates data required for plotting distributions and time series."""
    plot_data = {
        'numerical_distributions': {}, # Renamed from numerical_histograms
        'time_series_daily_counts': {}
    }
    
    # 1. Numerical Columns (for Histogram/Box Plot)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    filtered_numerical_cols = [
        col for col in numerical_cols
        if not any(keyword in col.lower() for keyword in id_keywords) and col.lower() not in [e.lower() for e in excluded_columns]
    ]

    for col in filtered_numerical_cols:
        valid_data = df[col].dropna()
        if not valid_data.empty:
            
            # Use NumPy to calculate histogram bins and counts explicitly
            # This provides consistent, serializable data for both the histogram and box plot
            if valid_data.nunique() > 1:
                hist_counts, hist_bins = np.histogram(valid_data, bins='auto')
                
                plot_data['numerical_distributions'][col] = {
                    'raw_values': valid_data.tolist(),  # Used for box plot
                    'hist_counts': hist_counts.tolist(),
                    'hist_bins': hist_bins.tolist(),
                    'count': int(valid_data.count()),
                }
            else:
                # Handle edge case where column has only one unique value (prevents NumPy crash)
                plot_data['numerical_distributions'][col] = {
                    'raw_values': valid_data.tolist(),
                    'hist_counts': [len(valid_data)],
                    'hist_bins': [valid_data.min() - 0.5, valid_data.max() + 0.5],
                    'count': int(valid_data.count()),
                }

    # 2. Datetime Columns (for Time Series)
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()
    
    for col in datetime_cols:
        if not df[col].empty and df[col].count() > 0:
            # Resample to daily count, convert index/values to lists for JSON serialization
            df_time_series = df.set_index(col).resample('D').size().reset_index(name='Count')
            
            # Ensure the datetime index is converted to strings
            df_time_series[col] = df_time_series[col].dt.strftime('%Y-%m-%d')
            
            plot_data['time_series_daily_counts'][col] = {
                'dates': df_time_series[col].tolist(),
                'counts': df_time_series['Count'].tolist()
            }
            
    return plot_data