# AI-Powered Credit Risk Platform

This platform provides an end-to-end solution for predicting credit default risk, explaining predictions, and querying loan data using natural language.

## Architecture Overview
1. **Data Pipeline**: Loads and preprocesses data, handles class imbalance, and initializes a SQLite DB for Talk-to-Data.
2. **Machine Learning**: A LightGBM model trained to predict the probability of default.
3. **Explainable AI (XAI)**: SHAP TreeExplainer provides local explanations for each prediction.
4. **Talk-to-Data**: Gemini Pro LLM converts natural language questions to SQL and provides business insights.
5. **User Interface**: A Streamlit multi-page app combining EDA, Risk Prediction, and Talk-to-Data.
6. **Deployment**: Dockerized with Docker Compose.

## Setup and Run Instructions

### Prerequisites
- Docker and Docker Compose installed.
- Gemini API Key (get one from Google AI Studio).

### Steps
1. Clone this repository.
2. Rename `.env.example` to `.env` and insert your API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
3. Build and run the containers:
   ```bash
   docker-compose up --build
   ```
4. Access the platform at `http://localhost:8501`.

## ML Strategy
- **Algorithm**: LightGBM was chosen for its speed and native handling of tabular data.
- **Class Imbalance**: Used `scale_pos_weight` rather than SMOTE to maintain fast training times and avoid synthetic data generation overhead.

## Evaluation
- Metrics used: ROC-AUC and Precision-Recall, as Accuracy is misleading on highly imbalanced data.

## Talk-to-Data Approach
- Prompt templates are used to constrain the LLM to output valid SQL.
- We supply a schema definition within the prompt to ground the LLM's understanding of the table structure.