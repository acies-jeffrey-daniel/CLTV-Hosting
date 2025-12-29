import pandas as pd
import pandas.api.types as pdt
import difflib
from typing import Tuple, List, Dict
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 
import numpy as np
import os

#Helper function for column mapping
# def _auto_map_column(column_list, candidate_names):
    
#     for name in candidate_names:
#         match = difflib.get_close_matches(name, column_list, n=1, cutoff=0.75)
#         if match:
#             return match[0]
#     return None

# def standardize_columns(df: pd.DataFrame, expected_mapping: dict, df_name: str) -> pd.DataFrame:
    
#     print(f"Standardizing columns for {df_name} DataFrame...")
#     column_map = {}
#     for standard_name, candidates in expected_mapping.items():
#         mapped_col = _auto_map_column(df.columns.tolist(), candidates)
#         if mapped_col:
#             column_map[mapped_col] = standard_name
#         else:
#             print(f"Warning: Could not find a suitable column for '{standard_name}' in {df_name} DataFrame.")
    
#     df_standardized = df.rename(columns=column_map)
#     print(df_standardized.info())
#     return df_standardized

# ==============================
# SSOT Mapping Loader
# ==============================
def load_ssot_mapping(ssot_df: pd.DataFrame, sheet_name: str = "Sheet1") -> dict:
    """
    Loads the mapping dictionary from the SSOT DataFrame, preserving the SSOT
    column case for the final rename key, but using lowercase for aliases lookup.
    """
    df = ssot_df
    mapping_dict = {}
    
    for _, row in df.iterrows():
        # 1. Normalize file_name for robust dictionary key lookup (e.g., 'orders_file')
        # We perform heavy normalization to match the file_name parameter ('orders', 'transactions')
        file_name_key = str(row["file_name"]).strip().lower().replace('_file', '').replace('_', '').replace(' ', '')
        
        # 2. CRITICAL FIX: Preserve the original case of the SSOT column (e.g., 'Purchase Date')
        ssot_col_title_case = str(row["ssot_column"]).strip() 
        ssot_col_lower_case = ssot_col_title_case.lower()
        
        # 3. Process aliases and normalize to lowercase
        aliases = str(row["aliases"]).split(",")
        # Ensure all explicit aliases are lowercase
        aliases = [a.strip().lower() for a in aliases if a]
        
        # The value stored is the list of aliases + the SSOT column's lowercased name (for lookups)
        alias_list = aliases + [ssot_col_lower_case]
        
        # Use the file_name_key (e.g., 'orders') and the Title Case SSOT name (e.g., 'Purchase Date')
        mapping_dict.setdefault(file_name_key, {})[ssot_col_title_case] = alias_list
        
    return mapping_dict
# ==============================
# MAPPING METHODS (Adapted for Kedro Node)
# ==============================

def dict_lookup(col: str, alias_dict: Dict[str, List[str]]):
    for ssot, aliases in alias_dict.items():
        if col.lower() in aliases:
            # Returns: (SSOT Name, Alias Used, Method, Score)
            return ssot, col.lower(), "dictionary", 1.0 
    return None, None, None, 0.0

def fuzzy_match(col: str, alias_dict: Dict[str, List[str]], threshold: float):
    best_match, best_score, best_ssot = None, 0, None
    for ssot, aliases in alias_dict.items():
        for alias in aliases:
            score = fuzz.token_sort_ratio(col.lower(), alias.lower())
            if score > best_score:
                best_score, best_match, best_ssot = score, alias, ssot
    if best_score >= threshold:
        return best_ssot, best_match, "fuzzy", best_score / 100
    return None, None, None, 0.0

def tfidf_match(col: str, alias_dict: Dict[str, List[str]], threshold: float):
    corpus, labels = [], []
    for ssot, aliases in alias_dict.items():
        for alias in aliases:
            corpus.append(alias.lower())
            labels.append((ssot, alias))
            
    # If corpus is empty, skip to prevent TFIDF error
    if not corpus:
        return None, None, None, 0.0
        
    vectorizer = TfidfVectorizer().fit(corpus + [col.lower()])
    vectors = vectorizer.transform(corpus + [col.lower()])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        ssot, alias = labels[idx]
        return ssot, alias, "tfidf", float(sims[idx])
    return None, None, None, 0.0

# NOTE: The full embedding match function must be run carefully due to model loading.
def embedding_match(col: str, ssot_name: str, df_col_embs: np.ndarray, df_cols: List[str], model: SentenceTransformer, threshold: float):
    
    # Encode the SSOT column name
    ssot_emb = model.encode([ssot_name])
    
    # Calculate similarity against all uploaded DataFrame columns
    sims = cosine_similarity(ssot_emb, df_col_embs)[0]
    idx = np.argmax(sims)
    
    if sims[idx] >= threshold:
        best_match_col = df_cols[idx]
        return ssot_name, best_match_col, "embedding", float(sims[idx])
    return None, None, None, 0.0

# ==============================
# Map Columns (The Core Logic)
# ==============================
def map_file_columns(df: pd.DataFrame, file_name: str, ssot_mapping: Dict[str, Dict[str, List[str]]], params: Dict) -> Tuple[Dict[str, str], pd.DataFrame]:
    
    # --- Setup ---
    # Load Model (must be imported globally: from sentence_transformers import SentenceTransformer)
    MODEL = SentenceTransformer(params['ssot_mapping_model'])
    
    # Lowercase the file name for dictionary lookup
    file_name = file_name.lower()
    # Normalize the incoming file_name for robust lookup against SSOT keys
    file_name_key = file_name.replace('_file', '').replace('_', '').replace(' ', '')
    alias_dict = ssot_mapping.get(file_name_key, {})
    results = [] # Stores mapping report (for final report DF)
    
    original_cols = df.columns.tolist()
    # Strip whitespace from columns but preserve original case for mapping back
    df.columns = [col.strip() for col in original_cols] 
    lower_cols = [col.lower() for col in df.columns]
    
    # Precompute embeddings for df columns
    df_col_embs = MODEL.encode(lower_cols)
    
    # Dictionary to store final column remapping: {Original Name: SSOT Name}
    column_rename_map = {}
    
    print(f"Starting advanced mapping for file: {file_name}")

    # --- Core Mapping Logic ---
    # Iterate over every EXPECTED SSOT column for this file type
    for ssot_col, aliases in alias_dict.items():
        
        best_match_found = False
        original_col_name = None
        mapped_col_name = None
        method = None
        score = 0.0

        # 1. DICTIONARY LOOKUP (Highest Priority)
        for i, col in enumerate(df.columns):
            # Check if this column is already mapped, or if its lowercased name is in the SSOT aliases
            if original_cols[i] not in column_rename_map and col.lower() in aliases:
                
                mapped_col_name = col
                original_col_name = original_cols[i]
                method = "dictionary"
                score = 1.0
                best_match_found = True
                break
        
        # If an exact dictionary match was found, record it and move to the next SSOT column
        if best_match_found:
            results.append([file_name, ssot_col, mapped_col_name, method, score, original_col_name])
            column_rename_map[original_col_name] = ssot_col # Map Original name to SSOT name
            continue


        # --- Fallback to Semantic/Fuzzy Methods (Only runs if Dict Lookup failed) ---
        
        # 2. EMBEDDING MATCH (Semantic Similarity - SSOT vs. Uploaded Columns)
        ssot_, mapped_col_emb, method_emb, score_emb = embedding_match(
            col=ssot_col, 
            ssot_name=ssot_col,
            df_col_embs=df_col_embs,
            df_cols=df.columns.tolist(),
            model=MODEL,
            threshold=params['ssot_threshold_emb']
        )
        
        # Check if the semantic match is better than current best (which is 0.0 here)
        if mapped_col_emb is not None and score_emb > score:
             mapped_col_name = mapped_col_emb
             method = method_emb
             score = score_emb
             # Get the original column name for the mapped column
             original_col_name = original_cols[df.columns.tolist().index(mapped_col_name)]
        
        
        # 3. FALLBACK FUZZY (Last-chance fuzzy against UNMAPPED columns)
        best_match_fuzzy, best_score_fuzzy, best_match_original_fuzzy = None, 0.0, None
        
        for i, col in enumerate(df.columns):
             # Only check UNMAPPED columns
            if original_cols[i] not in column_rename_map: 
                score_fuzz = fuzz.token_sort_ratio(ssot_col.lower(), col.lower()) / 100
                if score_fuzz > best_score_fuzzy:
                    best_match_fuzzy, best_score_fuzzy, best_match_original_fuzzy = col, score_fuzz, original_cols[i]

        
        # FINAL DECISION: Prioritize Embedding if score is high, otherwise use Fallback Fuzzy if above threshold
        if score > 0.0 and original_col_name not in column_rename_map: # Embedding was the best semantic match
            pass # Use Embedding results
        elif best_match_fuzzy and best_score_fuzzy >= params['ssot_threshold_fallback'] and best_match_original_fuzzy not in column_rename_map: 
             mapped_col_name = best_match_fuzzy
             method = "fallback_fuzzy"
             score = best_score_fuzzy
             original_col_name = best_match_original_fuzzy
        else: # No acceptable match found
            mapped_col_name = None
            method = None
            score = 0.0
            original_col_name = None

        # Record the final best match (or lack thereof)
        if mapped_col_name is not None:
             results.append([file_name, ssot_col, mapped_col_name, method, score, original_col_name])
             column_rename_map[original_col_name] = ssot_col
        else:
             results.append([file_name, ssot_col, None, None, 0.0, None]) # Unmapped entry

    
    # ----------------------------------------------------
    # Apply Renaming & Create Report
    # ----------------------------------------------------
    
    # Create the report from results list (optional output)
    mapping_report_df = pd.DataFrame(results, columns=[
        "file_name", "expected_column", "mapped_column", "method_used", "confidence_score", "original_column"
    ])
    
    # Apply the final renaming to the DataFrame
    final_rename_map = {original: ssot for original, ssot in column_rename_map.items()}
    df_standardized = df.rename(columns=final_rename_map)
    
    return final_rename_map, df_standardized
# ==============================
# THE NEW KEDRO NODE (Replaces standardize_columns)
# ==============================
def standardize_columns_advanced(
    df: pd.DataFrame, 
    ssot_mapping_file: pd.DataFrame, # SSOT Excel file loaded as DataFrame
    df_name: str, # e.g., "orders" or "transactions" (used for file_name lookup)
    params: Dict # All threshold and model parameters
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    

    if df.empty:
        if df_name.lower() == 'behavioral':
            print(f"[INFO] {df_name} DataFrame is empty. Skipping standardization and returning empty outputs.")
            # Returns an empty DataFrame with the expected report format
            return pd.DataFrame(), {}
    
    # Note: load_ssot_mapping and map_file_columns functions are assumed to be defined/imported correctly
    # and contain the SSOT mapping logic discussed previously.
    ssot_mapping = load_ssot_mapping(ssot_mapping_file)
    
    # The new mapping logic returns the rename map and the transformed df
    # The original_cols parameter from map_file_columns is crucial for linking back.
    rename_map, df_standardized = map_file_columns(df.copy(), df_name.lower(), ssot_mapping, params)
    
    print(f"Standardizing columns for {df_name} DataFrame...")

    # --- DIAGNOSTIC ADDITION ---
    print(f"\n[DIAGNOSTIC] --- MAPPING RESULTS FOR {df_name.upper()} ---")
    print(f"[DIAGNOSTIC] Input Columns: {df.columns.tolist()}")
    print(f"[DIAGNOSTIC] Final Rename Map (Original -> SSOT): {rename_map}")
    
    # Assuming SSOT mapping structure is Dict[file_name: Dict[SSOT_Col: List[Aliases]]]
    try:
        ssot_target_cols = ssot_mapping.get(df_name.lower(), {}).keys()
        print(f"[DIAGNOSTIC] SSOT Target Columns: {list(ssot_target_cols)}")
    except:
        print("[DIAGNOSTIC] Could not list SSOT Target Columns.")

    print(f"[DIAGNOSTIC] Resulting Columns: {df_standardized.columns.tolist()}")
    print(f"[DIAGNOSTIC] --------------------------------------------\n")
    # -----------------------------
    
    # We return the standardized DataFrame and the map for reference/reporting (optional output)
    return df_standardized, rename_map


def convert_data_types(
    orders_df: pd.DataFrame, 
    transactions_df: pd.DataFrame, 
    behavioral_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Converting data types...")

    # --- DIAGNOSTIC ADDITIONS ---
    txn_required = ['Purchase Date', 'User ID', 'Total Amount']
    txn_missing = [c for c in txn_required if c not in transactions_df.columns]
    
    if txn_missing:
        print(f"\n[DIAGNOSTIC] TXN FAIL: Missing critical columns for features/typing: {txn_missing}")
        print(f"[DIAGNOSTIC] TXN FAIL: Available columns: {transactions_df.columns.tolist()}")
    else:
        print(f"\n[DIAGNOSTIC] TXN SUCCESS: All critical columns ({txn_required}) are present.")
    # -----------------------------

    # Transactions
    if 'Purchase Date' in transactions_df.columns:
        transactions_df['Purchase Date'] = pd.to_datetime(
            transactions_df['Purchase Date'], dayfirst=False, errors='coerce'
        )
    else:
        print("Warning: 'Purchase Date' not found in transactions_df.")

    if 'User ID' in transactions_df.columns:
        transactions_df['User ID'] = transactions_df['User ID'].astype(str)
    else:
        print("Warning: 'User ID' not found in transactions_df.")

    numeric_cols_txn = ['Total Amount', 'Total Payable', 'Discount Value', 'Shipping Cost']
    for col in numeric_cols_txn:
        if col in transactions_df.columns:
            transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')

    # Orders
    if 'Return Date' in orders_df.columns:
        orders_df['Return Date'] = pd.to_datetime(
            orders_df['Return Date'], dayfirst=True, errors='coerce'
        )
    else:
        print("Warning: 'Return Date' not found in orders_df.")

    numeric_cols_orders = ['Unit Price', 'Quantity']
    for col in numeric_cols_orders:
        if col in orders_df.columns:
            orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')

    # Behavioral
    if 'Visit Timestamp' in behavioral_df.columns:
        behavioral_df['Visit Timestamp'] = pd.to_datetime(
            behavioral_df['Visit Timestamp'], errors='coerce'
        )
    else:
        print("Warning: 'Visit Timestamp' not found in behavioral_df.")

    numeric_cols_behavioral = [
        # 'Session Total Cost',
        'Session Duration',
        'Page Views'
    ]
    for col in numeric_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = pd.to_numeric(behavioral_df[col], errors='coerce')

    bool_like_cols = [
        'Sponsored Listing Viewed',
        'Banner Viewed',
        'Homepage Promo Seen',
        'Product Search View',
        'Bounce Flag'
    ]
    for col in bool_like_cols:
        if col in behavioral_df.columns:
            behavioral_df[col] = behavioral_df[col].astype(bool)

    id_cols_behavioral = [
        'Visit ID', 'Customer ID', 'Session ID', 'Device ID', 'Cookie ID', 'Ad Campaign ID'
    ]
    for col in id_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = behavioral_df[col].astype(str)

    str_cols_behavioral = [
        'Channel', 'Geo Location', 'Device Type', 'OS', 'Entry Page', 'Exit Page'
    ]
    for col in str_cols_behavioral:
        if col in behavioral_df.columns:
            behavioral_df[col] = behavioral_df[col].astype(str)


    return orders_df, transactions_df, behavioral_df


def merge_orders_transactions(orders_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:

    print("Merging orders and transactions...")
    if 'Transaction ID' in transactions_df.columns and \
       'Transaction ID' in orders_df.columns and \
       'User ID' in transactions_df.columns:
        
        if 'User ID' in transactions_df.columns:
            df_orders_merged = orders_df.merge(
                transactions_df[['Transaction ID', 'User ID']],
                on='Transaction ID',
                how='left'
            )
        else:
            print("Warning: 'User ID' not found in transactions_df for merge. Skipping User ID merge.")
            df_orders_merged = orders_df.copy()
    else:
        print("Warning: 'Transaction ID' or 'User ID' not found in both orders and transactions for merging. Skipping merge.")
        df_orders_merged = orders_df.copy()
        print(df_orders_merged.info())

    return df_orders_merged

def aggregate_behavioral_customer_level(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the behavioral dataset at the Customer ID level.
    Handles missing columns gracefully (skips them if not present).
    Produces cleaner, business-friendly column names.
    """

    print("Aggregating behavioral data at customer level...")
    if behavioral_df.empty:
        print("[INFO] Input behavioral data is empty. Skipping aggregation and returning empty DataFrame.")
        # Return an empty DataFrame with the expected key column
        return pd.DataFrame(columns=["Customer ID"])

    # Define aggregation logic
    agg_map = {
        "Visit ID": "nunique",
        "Session ID": "nunique",
        "Visit Timestamp": ["min", "max"],
        "Channel": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Geo Location": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Device ID": "nunique",
        "Device Type": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Cookie ID": "nunique",
        "OS": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Entry Page": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Exit Page": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        "Sponsored Listing Viewed": "sum",
        "Banner Viewed": "sum",
        "Homepage Promo Seen": "sum",
        "Product Search View": "sum",
        # "Session Total Cost": ["sum", "mean"],
        "Session Duration": ["sum", "mean"],
        "Page Views": ["sum", "mean"],
        "Bounce Flag": ["sum", "mean"],
        "Ad Campaign ID": lambda x: list(set(x.dropna())),
    }

    available_agg_map = {
        col: agg for col, agg in agg_map.items() if col in behavioral_df.columns
    }

    if "Customer ID" not in behavioral_df.columns:
        raise ValueError("Customer ID column is required for aggregation.")

    agg_df = behavioral_df.groupby("Customer ID").agg(available_agg_map).reset_index()

    agg_df.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in agg_df.columns.values
    ]
    rename_map = {
        "Visit ID nunique": "Total Unique Visits",
        "Session ID nunique": "Total Unique Sessions",
        "Visit Timestamp min": "First Visit Timestamp",
        "Visit Timestamp max": "Last Visit Timestamp",
        "Visit Timestamp count": "Total Visits",
        "Device ID nunique": "Total Unique Devices",
        "Cookie ID nunique": "Total Unique Cookies",
        "Sponsored Listing Viewed sum": "Total Sponsored Listings Viewed",
        "Banner Viewed sum": "Total Banners Viewed",
        "Homepage Promo Seen sum": "Total Homepage Promos Seen",
        "Product Search View sum": "Total Product Searches Viewed",
        "Session Total Cost sum": "Total Session Cost",
        "Session Total Cost mean": "Avg Session Cost",
        "Session Duration sum": "Total Session Duration",
        "Session Duration mean": "Avg Session Duration",
        "Page Views sum": "Total Page Views",
        "Page Views mean": "Avg Page Views",
        "Bounce Flag sum": "Total Bounces",
        "Bounce Flag mean": "Bounce Rate",
        "Channel <lambda>": "Channel",
        "Geo Location <lambda>": "Geo Location",
        "Device Type <lambda>":"Device Type",
        "Exit Page <lambda>": "Exit Page"


    }

    agg_df = agg_df.rename(columns={k: v for k, v in rename_map.items() if k in agg_df.columns})
    print(agg_df.info())
    return agg_df

def aggregate_orders_transactions_customer_level(
    orders_txn_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates merged orders+transactions dataset at the User ID (customer) level.
    Focuses on spend and counts (transactions, orders, unique products).
    """

    print("Aggregating orders+transactions at customer level...")

    if "User ID" not in orders_txn_df.columns:
        raise ValueError("'User ID' column is required for customer-level aggregation.")

    agg_map = {
        "Transaction ID": "nunique",
        "Order ID": "nunique",
        "Product ID": "nunique" if "Product ID" in orders_txn_df.columns else None,
        "Total Amount": "sum" if "Total Amount" in orders_txn_df.columns else None,
        "Total Product Price": "sum" if "Total Product Price" in orders_txn_df.columns else None,
        "Total Payable": "sum" if "Total Payable" in orders_txn_df.columns else None,
        # "Discount Value": "sum" if "Discount Value" in orders_txn_df.columns else None,
        # "Shipping Cost": "sum" if "Shipping Cost" in orders_txn_df.columns else None,
        # "Purchase Date": ["min", "max"] if "Purchase Date" in orders_txn_df.columns else None,
    }

    agg_map = {col: agg for col, agg in agg_map.items() if agg is not None and col in orders_txn_df.columns}
    agg_df = orders_txn_df.groupby("User ID").agg(agg_map).reset_index()
    agg_df.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in agg_df.columns.values
    ]
    rename_map = {
        "Transaction ID nunique": "Total Transactions",
        "Order ID nunique": "Total Orders",
        "Product ID nunique": "Total Unique Products",
        "Total Product Price": "Total Order Value",
        "Total Amount sum": "Total Product Sum",
        "Total Payable sum": "Total Payable Value",
        "Discount Value sum": "Total Discounts Availed",
        "Shipping Cost sum": "Total Shipping Paid",
        "Purchase Date min": "First Purchase Date",
        "Purchase Date max": "Last Purchase Date",
    }
    agg_df = agg_df.rename(columns={k: v for k, v in rename_map.items() if k in agg_df.columns})
    print(agg_df.info())
    return agg_df


def merge_customer_ord_txn_behavioral_data(
    orders_txn_customer_df: pd.DataFrame,
    behavioral_agg_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges aggregated customer-level orders+transactions data 
    with aggregated behavioral data.
    Handles missing keys gracefully.
    """

    print("Merging aggregated customer-level orders+transactions with behavioral data...")
    if "User ID" not in orders_txn_customer_df.columns:
        print("Warning: 'User ID' not found in orders+transactions customer-level dataset. Skipping merge.")
        return orders_txn_customer_df

    if "Customer ID" not in behavioral_agg_df.columns:
        print("Warning: 'Customer ID' not found in aggregated behavioral dataset. Skipping merge.")
        return behavioral_agg_df

    merged_df = orders_txn_customer_df.merge(
        behavioral_agg_df,
        left_on="User ID",
        right_on="Customer ID",
        how="left"
    )
    print(merged_df.info())
    return merged_df

def cap_outliers_by_quantile(
    df: pd.DataFrame,
    columns_to_cap: list[str],
    cap_multiplier: float = 3.0,
    quantile: float = 0.75
) -> Tuple[pd.DataFrame, dict[str, float]]:
    """
    Applies upper-limit capping based on a quantile multiplier (e.g., 3 * Q3).

    Args:
        df: Input DataFrame.
        columns_to_cap: List of columns to apply capping to.
        cap_multiplier: Multiplier for the quantile (e.g., 3.0 for 3 * Q3).
        quantile: The quantile to use (e.g., 0.75 for 75th percentile).

    Returns:
        Tuple of (Capped DataFrame, Dictionary of applied cap values).
    """
    df_capped = df.copy()
    cap_values = {}
    
    print(f"[INFO] Applying outlier capping to {len(columns_to_cap)} columns...")

    for col in columns_to_cap:
        if col in df_capped.columns and pdt.is_numeric_dtype(df_capped[col]):
            # Filter out NaNs for quantile calculation
            valid_series = df_capped[col].dropna()
            
            if valid_series.empty:
                print(f"[WARN] Column '{col}' is empty after dropping NaNs. Skipping.")
                continue

            q_val = valid_series.quantile(quantile)
            cap_value = cap_multiplier * q_val
            cap_values[col] = cap_value

            values_above_cap = df_capped[df_capped[col] > cap_value].shape[0]
            
            if values_above_cap > 0:
                # Apply the capping
                df_capped[col] = df_capped[col].clip(upper=cap_value)
                print(f"[INFO] Capped {values_above_cap} values in column '{col}' at cap: {cap_value:.2f}")
            else:
                print(f"[INFO] Column '{col}': No values found above cap of {cap_value:.2f}.")
        else:
            print(f"[WARN] Column '{col}' skipped (not found or not numeric).")

    return df_capped, cap_values