import pandas as pd
import sqlite3
from src.utils.config import TRAIN_DATA_PATH, TEST_DATA_PATH, DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(is_train=True):
    path = TRAIN_DATA_PATH if is_train else TEST_DATA_PATH
    logger.info(f"Loading data from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def init_sqlite_db():
    logger.info(f"Initializing SQLite database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:

        df = pd.read_csv(TRAIN_DATA_PATH)
        df.to_sql("application_train", conn, if_exists="replace", index=False)
        logger.info("Successfully initialized SQLite database with application_train table.")
    except Exception as e:
        logger.error(f"Error initializing SQLite DB: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    init_sqlite_db()
