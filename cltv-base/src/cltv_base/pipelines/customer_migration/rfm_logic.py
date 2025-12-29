from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

# --- 7-bucket segment mapping (your spec) ---
def assign_segment(row):
    r = row['r_score']
    fm = row['fm_score']
    #r = int(row['r_score']); fm = int(row['fm_score'])
    # 1. Champions: Highest Recency and FM score.

    if r >= 4 and fm == 5:
        return 'Champions'

    # 2. Potential Champions: High Recency and FM, but not the top tier.

    elif (r >= 4 and fm >= 4):
        return 'Potential Champions'

    # 3. Customers Needing Attention: Medium Recency, high FM.

    elif (r >= 4 and fm == 3) or (r == 3 and fm >= 4):
        return 'Customers Needing Attention'

    # 4. Recent Customers: High Recency, lower FM.

    elif r >= 4 and fm <= 2:
        return 'Activated/Reactived'

    # 5. Loyal Lapsers: Low Recency, but high FM.

    elif r <= 3 and fm >= 3:
        return 'Loyal Lapsers'

    # 6. About to Sleep: Low Recency and low to medium FM.
    
    elif (r <= 3 and fm <= 2) or (r <= 2 and fm <= 2):
        return 'About to Sleep'

    # 7. Lost: All remaining combinations, which have low Recency and low FM.


    elif r <= 2 and fm <= 2:
        return 'Lost'
    else:
        return 'Unclassified'
# --- helpers ---
def safe_qscore(s: pd.Series, q: int = 5, labels=None) -> pd.Series:
    if labels is None:
        labels = list(range(1, q + 1))
    s = s.astype(float)
    if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
        return pd.Series([int(np.ceil(q / 2))] * len(s), index=s.index, dtype=object)
    try:
        ranks = s.rank(method='first')
        return pd.qcut(ranks, q, labels=labels)
    except Exception:
        return pd.cut(s, bins=q, labels=labels, include_lowest=True)

def add_period_and_end(df: pd.DataFrame, date_col: str, period: str) -> pd.DataFrame:
    out = df.copy()
    out['__period'] = out[date_col].dt.to_period(period)
    period_end_map = (
        out[['__period']].drop_duplicates()
        .assign(period_end=lambda x: x['__period'].dt.to_timestamp(how='end'))
    )
    return out.merge(period_end_map, on='__period', how='left')

# --- RFM per period ---
def compute_rfm_for_period(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    order_col: str,
    amount_col: str,
    period: str = 'M'
) -> pd.DataFrame:
    w = df.copy()
    w[date_col] = pd.to_datetime(w[date_col], errors='coerce')
    w[amount_col] = pd.to_numeric(w[amount_col], errors='coerce').fillna(0)
    w = add_period_and_end(w, date_col=date_col, period=period)

    g = (w.groupby([customer_col, '__period', 'period_end'])
        .agg(last_purchase=(date_col, 'max'),
                Frequency=(order_col, 'nunique'),
                Monetary=(amount_col, 'sum'))
        .reset_index())

    g['Recency'] = (g['period_end'] - g['last_purchase']).dt.days
    
    # ⭐ FIX: Recency Score Inversion (Uses Recency column)
    g['r_score'] = pd.qcut(
        g['Recency'].rank(method='first', ascending=False), # Use capital 'Recency' column
        5, 
        labels=[5, 4, 3, 2, 1]
    ).astype(object) 
    
    # FIX: Frequency and Monetary Scores
    g['f_score'] = safe_qscore(g['Frequency'], 5, labels=[1,2,3,4,5]) # Uses capital 'Frequency'
    g['m_score'] = safe_qscore(g['Monetary'],  5, labels=[1,2,3,4,5]) # Uses capital 'Monetary'
    
    g['fm_score'] = ((g['f_score'].astype(int) + g['m_score'].astype(int))/2).round().astype(int)
    g['Segment']  = g.apply(assign_segment, axis=1)

    period_col = 'month' if period == 'M' else 'quarter'
    g = g.rename(columns={'__period': period_col})
    
    # The columns defined here are correct: Recency, Frequency, Monetary
    cols = ['Recency','Frequency','Monetary','r_score','f_score','m_score','fm_score','Segment']
    return g[[customer_col, period_col, 'period_end'] + cols]

# --- Migrations (per pair) ---
def build_migration_by_period_pair(rfm_period_df: pd.DataFrame, customer_col: str, period_col: str):
    df = rfm_period_df.sort_values([customer_col, period_col]).copy()
    
    
    df['transactions'] = df['Frequency']
    df['aov'] = df['Monetary'] / df['Frequency']
    
    
    df['next_period']       = df.groupby(customer_col)[period_col].shift(-1)
    df['next_segment']      = df.groupby(customer_col)['Segment'].shift(-1)
    df['next_transactions'] = df.groupby(customer_col)['transactions'].shift(-1)
    df['next_monetary']     = df.groupby(customer_col)['Monetary'].shift(-1)
    df['next_aov']          = df.groupby(customer_col)['aov'].shift(-1)
    
    
    trans = df.dropna(subset=['next_period', 'next_segment'])
    
    out = {}
    for (cp, np_), g in trans.groupby([period_col, 'next_period']):
        counts = pd.crosstab(g['Segment'], g['next_segment'])
        perc   = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).round(4)
        start_period_metrics = g.groupby('Segment').agg(
            transactions_count=('transactions', 'sum'),
            total_monetary=('Monetary', 'sum')
        ).assign(aov=lambda x: x['total_monetary'] / x['transactions_count'])

        # Correct calculation for end period metrics using 'next' columns
        end_period_metrics = g.groupby('next_segment').agg(
            transactions_count=('next_transactions', 'sum'),
            total_monetary=('next_monetary', 'sum')
        ).assign(aov=lambda x: x['total_monetary'] / x['transactions_count'])

        # Calculate percentages for transactions
        start_total_transactions = start_period_metrics['transactions_count'].sum()
        start_period_metrics['transactions_percent'] = (start_period_metrics['transactions_count'] / start_total_transactions * 100).round(2)
        
        end_total_transactions = end_period_metrics['transactions_count'].sum()
        end_period_metrics['transactions_percent'] = (end_period_metrics['transactions_count'] / end_total_transactions * 100).round(2)

        out[(cp, np_)] = {
            'counts': counts, 
            'percent': perc, 
            'start_period_metrics': start_period_metrics,
            'end_period_metrics': end_period_metrics
        }
    return out

def list_available_period_pairs(migration_by_pair, period_freq: str):
    pairs = []
    for (cp, np_) in migration_by_pair.keys():
        cpp = cp if isinstance(cp, pd.Period) else pd.Period(str(cp), period_freq)
        npp = np_ if isinstance(np_, pd.Period) else pd.Period(str(np_), period_freq)
        pairs.append((str(cpp), str(npp)))
    pairs.sort(key=lambda t: (pd.Period(t[0], period_freq), pd.Period(t[1], period_freq)))
    return pairs