import joblib
import shap

from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model():
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict(model, X):
    probabilities = model.predict_proba(X)[:, 1]

    bands = []

    for probability in probabilities:
        if probability < 0.3:
            bands.append("Low")
        elif probability < 0.6:
            bands.append("Medium")
        else:
            bands.append("High")

    return probabilities, bands


def explain_prediction(model, X_instance):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_instance)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values