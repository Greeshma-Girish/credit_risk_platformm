import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "credit_risk.db"

# Data files
TRAIN_DATA_PATH = DATA_DIR / "application_train.csv"
TEST_DATA_PATH = DATA_DIR / "application_test.csv"

# Model file
MODEL_PATH = MODELS_DIR / "lgb_model.pkl"

# LLM Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
