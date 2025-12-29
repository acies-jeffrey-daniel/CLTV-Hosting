from kedro.pipeline import Pipeline, node, pipeline
from .nodes import compute_monthly_quarterly_rfm, build_all_migrations

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_monthly_quarterly_rfm,
            inputs=dict(
                transactions="transactions_typed",
                customer_col="params:customer_col",
                date_col="params:date_col",
                order_col="params:order_col",
                amount_col="params:amount_col",
            ),
            outputs=dict(
                monthly_rfm="monthly_rfm",
                quarterly_rfm="quarterly_rfm",
            ),
            name="compute_rfm_both",
        ),
        node(
            func=build_all_migrations,
            inputs=dict(
                monthly_rfm="monthly_rfm",
                quarterly_rfm="quarterly_rfm",
                customer_col="params:customer_col",
            ),
            outputs=dict(
                monthly_pair_migrations="monthly_pair_migrations",
                quarterly_pair_migrations="quarterly_pair_migrations",
            ),
            name="build_migrations",
        ),
    ])