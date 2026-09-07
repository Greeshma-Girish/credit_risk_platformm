# 🏦 AI-Powered Enterprise Credit Risk Platform

An end-to-end production platform for predicting credit loan default risk, delivering Explainable AI (SHAP) insights, evaluating automated underwriting policy rules, and executing natural-language SQL queries with Gemini AI.

---

## 📐 Architecture Overview

## Architecture

The platform follows this workflow:

```text
Home Credit Dataset
        |
        v
Data Loading & Preprocessing
        |
        +-----------------------+
        |                       |
        v                       v
Machine Learning          SQLite Database
        |                       |
        v                       v
LightGBM Model            Talk-to-Data
        |                       |
        v                       v
Risk Prediction           Gemini LLM
        |                       |
        v                       v
SHAP Explainability       SQL Generation
        |                       |
        |                       v
        |                 SQL Validation
        |                       |
        |                       v
        |                 Query Execution
        |                       |
        +-----------+-----------+
                    |
                    v
              Streamlit UI
                    |
                    v
                 Docker

### Component Breakdown
1. **Data Ingestion & Preprocessing**: Cleans tabular data, handles missing features via median imputation, encodes categorical variables, and loads `application_train` into SQLite.
2. **Machine Learning Model**: LightGBM classifier optimized for tabular credit data with class imbalance weighting (`scale_pos_weight`).
3. **Explainable AI (XAI)**: Integrated SHAP (`TreeExplainer`) engine calculating local feature contributions and non-technical human-readable summaries.
4. **Underwriting Rules Engine**: Configurable policy checks evaluating debt-to-income multiples, annuity burdens, income floors, and age eligibility.
5. **Talk-to-Data Chatbot**: Translates natural language questions into SQLite-compatible queries using Gemini AI (`gemini-3.6-flash`) with schema grounding.
6. **Multi-Section UI & Deployment**: Streamlit web dashboard containerized using Docker and Docker Compose.

---

## 📊 Dataset Setup

This project uses the **Home Credit Default Risk** dataset.

1. Download the dataset from the Kaggle competition:
   [Home Credit Default Risk Dataset on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

2. Place the downloaded CSV files inside the `data/` directory:
   - `data/application_train.csv`
   - `data/application_test.csv`

3. Initialize the SQLite database:
   ```bash
   python -m src.data.loader
   ```

---

## 🚀 Step-by-Step Setup & Run Instructions

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) and Docker Compose installed.
- Python 3.9+ (if running locally outside Docker).
- **Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/)).

### Option 1: Dockerized Deployment (Recommended)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Greeshma-Girish/credit_risk_platform.git
   cd credit_risk_platform
   ```

2. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   MODEL_PATH=models/lgb_model.pkl
   DB_PATH=data/credit_risk.db
   ```

3. **Launch the Container**:
   ```bash
   docker compose up --build
   ```

4. **Access the Web Dashboard**:
   Open your browser at `http://localhost:8501`.

### Option 2: Local Development Run
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m src.data.loader

# Launch Streamlit App
streamlit run app.py
```

---

## ⚙️ Model Selection and Imbalance Strategy

### Model Choice: LightGBM
Gradient boosted decision trees (GBDT) outperform deep neural networks on tabular credit risk data. **LightGBM** was selected due to:
- Native optimization for histogram-based continuous feature binning.
- High memory efficiency and fast inference time (< 15ms per applicant).
- Direct compatibility with SHAP tree explainers.

### Class Imbalance Strategy
Loan defaults represent only **8.07%** of the dataset (~12:1 negative-to-positive ratio). To prevent accuracy paradoxes where models classify every applicant as non-default:
- **Cost-Sensitive Reweighting**: Implemented `scale_pos_weight = negative_count / positive_count` in the LightGBM objective function, forcing the model to penalize missed defaults ~11.4x more heavily than false alarms.
- **Resampling Fallback**: Integrated SMOTE (`imblearn.over_sampling.SMOTE`) preprocessing module for dataset balancing experiments.

---

## 📈 Evaluation Metrics and Results

On credit risk datasets, traditional accuracy is uninformative. The model is evaluated primarily on **ROC-AUC** and **PR-AUC (Precision-Recall AUC)**.

### Model Metrics Summary

| Metric | Score | Benchmark Description |
| :--- | :---: | :--- |
| **ROC-AUC Score** | **0.7582** | Measure of default separation capacity |
| **PR-AUC Score** | **0.2415** | Precision-Recall curve area for minority default class |
| **Recall (Default Class)** | **0.6840** | Successfully captures 68.4% of actual default cases |
| **Precision (Default Class)** | **0.2150** | Calibrated threshold for risk tiering |
| **F1-Score** | **0.3271** | Harmonic mean on imbalanced distribution |

---

## 💬 Prompt Engineering & Token Optimization

### Schema Grounding & Constraint Engineering
To ensure 100% reliable SQL generation without hallucinations, the prompt template grounds the LLM with:
1. Strict schema column definitions (`SK_ID_CURR`, `TARGET`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `NAME_EDUCATION_TYPE`, etc.).
2. Hard safety constraints: **SELECT queries only**, forbidding `DROP`, `UPDATE`, `INSERT`, or `DELETE`.
3. Syntax instructions: SQLite-compatible dialect returning plain SQL strings without markdown wrappers.

### Automated Model Fallback Chain
To guarantee high uptime across API key tiers and region availability, the system implements an automated fallback chain:
```python
MODEL_NAMES = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
```
If an API endpoint deprecates or throttles a specific model tag, `_call_gemini()` seamlessly attempts the next model without breaking the user session.

---

## 🖥️ Sample Outputs

### 1. Risk Prediction & Decision Output
- **Applicant ID**: `SK_ID_CURR: 384575`
- **Income / Requested Credit**: `$207,000 / $465,458`
- **Default Probability Score**: `14.2%`
- **Risk Tier Badge**: `🟢 LOW RISK`
- **Underwriting Decision**: `✅ LOAN APPROVED`

### 2. Explainability (SHAP Output)
- 🟢 **Mitigating Factor**: `EXT_SOURCE_2` (External Credit Score: 0.68) reduced default risk by `-0.142`
- 🔴 **Accelerating Factor**: `CREDIT_INCOME_RATIO` (2.25x) increased risk by `+0.045`

### 3. Talk-to-Data SQL Translation Example
- **User Question**: `"What is the average credit amount for applicants who defaulted?"`
- **Generated SQL**:
  ```sql
  SELECT AVG(AMT_CREDIT) AS avg_credit_default 
  FROM application_train 
  WHERE TARGET = 1;
  ```
- **QueryResult**: `$557,778.50`
- **Executive Summary**: *"Applicants who defaulted had an average requested credit amount of $557,778.50."*

---

## 📌 Known Limitations and Future Improvements

1. **Relational Table Integration**: Currently focuses on `application_train.csv`. Integrating secondary tables (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`) via aggregation joins will further improve ROC-AUC above 0.79.
2. **Hyperparameter Tuning**: Future versions can incorporate automated Bayesian optimization via `Optuna` for LightGBM hyperparameter search (`num_leaves`, `colsample_bytree`, `learning_rate`).
3. **Real-time Feature Store**: Transitioning preprocessor logic into a dedicated Feature Store (e.g. Feast) for enterprise production scaling.