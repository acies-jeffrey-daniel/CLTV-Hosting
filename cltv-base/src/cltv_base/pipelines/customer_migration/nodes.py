from typing import Dict, Tuple
import pandas as pd
from .rfm_logic import compute_rfm_for_period, build_migration_by_period_pair

def compute_monthly_quarterly_rfm(
    transactions: pd.DataFrame,
    customer_col: str,
    date_col: str,
    order_col: str,
    amount_col: str
) -> Dict[str, pd.DataFrame]:
    monthly   = compute_rfm_for_period(transactions, customer_col, date_col, order_col, amount_col, period='M')
    quarterly = compute_rfm_for_period(transactions, customer_col, date_col, order_col, amount_col, period='Q')
    return {"monthly_rfm": monthly, "quarterly_rfm": quarterly}

def build_all_migrations(
    monthly_rfm: pd.DataFrame, quarterly_rfm: pd.DataFrame, customer_col: str
) -> Dict[str, Dict[Tuple[object, object], Dict[str, pd.DataFrame]]]:
    m_pairs = build_migration_by_period_pair(monthly_rfm, customer_col=customer_col, period_col='month')
    q_pairs = build_migration_by_period_pair(quarterly_rfm, customer_col=customer_col, period_col='quarter')
    
    return {"monthly_pair_migrations": m_pairs, "quarterly_pair_migrations": q_pairs}