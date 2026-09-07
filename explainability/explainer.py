import shap
import pandas as pd


def create_explainer(model):
    return shap.TreeExplainer(model)


def explain_prediction(explainer, X):
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def get_feature_importance(X, shap_values, top_n=10):
    importance = pd.DataFrame({
        "feature": X.columns,
        "shap_value": shap_values[0]
    })

    importance["impact"] = importance["shap_value"].abs()

    importance = importance.sort_values(
        "impact",
        ascending=False
    )

    return importance.head(top_n)


def get_human_explanation(X, shap_values, top_n=5):
    importance = get_feature_importance(
        X,
        shap_values,
        top_n
    )

    explanation = []

    for _, row in importance.iterrows():
        direction = "increased" if row["shap_value"] > 0 else "decreased"

        explanation.append({
            "feature": row["feature"],
            "direction": direction,
            "impact": abs(row["shap_value"])
        })

    return explanation