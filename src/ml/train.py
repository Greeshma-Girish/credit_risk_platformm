import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.data.preprocessor import preprocess_data
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model():
    logger.info("Loading training data...")

    df = load_data(is_train=True)
    logger.info(f"Original dataset shape: {df.shape}")

    df = df.sample(n=20000, random_state=42)
    logger.info(f"Training dataset shape: {df.shape}")

    logger.info("Preprocessing data...")

    X, y = preprocess_data(df)

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    logger.info("Splitting data into train and validation sets...")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Validation samples: {X_val.shape[0]}")

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    logger.info(f"Negative samples: {negative_count}")
    logger.info(f"Positive samples: {positive_count}")
    logger.info(f"Scale positive weight: {scale_pos_weight:.2f}")

    logger.info("Creating LightGBM model...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    logger.info("Training LightGBM model...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        eval_names=[
            "training",
            "validation"
        ],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=10)
        ]
    )

    logger.info(f"Saving model to {MODEL_PATH}")

    joblib.dump(model, MODEL_PATH)

    logger.info("Model saved successfully.")

    shap_background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    shap_background.to_csv(
        "data/shap_background.csv",
        index=False
    )

    logger.info("SHAP background dataset saved successfully.")
    logger.info(f"Best iteration: {model.best_iteration_}")
    logger.info(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()