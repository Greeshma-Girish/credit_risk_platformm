import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model():
    logger.info("Loading model...")

    model = joblib.load(MODEL_PATH)

    logger.info("Loading test data...")

    df = load_data(is_train=True)
    df = df.sample(n=20000, random_state=42)

    X, y = preprocess_data(df)

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info("Generating predictions...")

    y_probability = model.predict_proba(X_val)[:, 1]
    y_pred = (y_probability >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_probability)
    pr_auc = average_precision_score(y_val, y_probability)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"PR-AUC: {pr_auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")

    print("\nModel Evaluation")
    print("----------------------------")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report")
    print("----------------------------")
    print(classification_report(
        y_val,
        y_pred,
        target_names=["No Default", "Default"],
        zero_division=0
    ))

    cm = confusion_matrix(y_val, y_pred)

    print("Confusion Matrix")
    print("----------------------------")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Default", "Default"]
    ).plot()

    plt.title("Credit Risk Model - Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()