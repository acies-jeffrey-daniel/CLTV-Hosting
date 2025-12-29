from kedro.pipeline import Pipeline, node
from .nodes import (
    calculate_all_distribution_thresholds,
    calculate_user_value_threshold,
    calculate_ml_based_threshold,
    label_churned_customers,
    get_churn_features_labels,
    train_churn_prediction_model,
    predict_churn_probabilities,
    get_customers_at_risk,
    assign_predicted_churn_labels,
    train_cox_survival_model
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Churn modeling pipeline:
    - Calculates threshold (distribution/user/ML/metric) based on params
    - Flags at-risk customers
    - Labels churned customers
    - Trains churn model
    - Predicts churn probabilities and labels
    - Trains survival model for churn time prediction
    """
    return Pipeline([
        
        node(
            func=calculate_all_distribution_thresholds,
            inputs=["historical_cltv_customers", "params:threshold_metric"],
            outputs="calculated_distribution_threshold",
            name="calculate_distribution_threshold"
        ),
        node(
            func=calculate_user_value_threshold,
            inputs=["params:threshold_metric", "params:user_threshold_value"],
            outputs="calculated_user_value_threshold",
            name="calculate_user_value_threshold"
        ),

        node(
            func=calculate_ml_based_threshold,
            inputs=["historical_cltv_customers", "params:threshold_metric"],
            outputs="calculated_ml_threshold",
            name="calculate_ml_based_threshold"
        ),
        
        node(
            func=label_churned_customers,
            inputs=["historical_cltv_customers", "params:threshold_metric", "calculated_distribution_threshold"],
            outputs="churn_labeled_customers",
            name="label_churned_customers"
        ),

        node(
            func=get_customers_at_risk,
            inputs="churn_labeled_customers",
            outputs="customers_at_risk_df",
            name="get_customers_at_risk"
        ),

        node(
            func=get_churn_features_labels,
            inputs="churn_labeled_customers",
            outputs=["churn_features", "churn_labels"],
            name="get_churn_features_labels"
        ),
        node(
            func=train_churn_prediction_model,
            inputs=[
                "churn_features", "churn_labels",
                "params:churn_model_n_estimators", "params:churn_model_random_state"
            ],
            outputs=[
                "churn_prediction_model", "churn_classification_report",
                "churn_feature_importances", "churn_features_test", "churn_labels_test"
            ],
            name="train_churn_prediction_model"
        ),
        node(
            func=predict_churn_probabilities,
            inputs=["churn_prediction_model", "churn_features"],
            outputs="predicted_churn_probabilities",
            name="predict_churn_probabilities"
        ),
        node(
            func=assign_predicted_churn_labels,
            inputs=["predicted_churn_probabilities", "params:predicted_churn_probability_threshold"],
            outputs="predicted_churn_labels",
            name="assign_predicted_churn_labels"
        ),
        node(
            func=train_cox_survival_model,
            inputs=["churn_labeled_customers", "params:cox_feature_cols"],
            outputs=["cox_survival_model", "cox_predicted_active_days"],
            name="train_cox_survival_model"
        ),
    ])
