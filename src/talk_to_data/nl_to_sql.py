import pandas as pd
from google import genai

from src.talk_to_data.prompt_templates import (
    SQL_SYSTEM_PROMPT,
    BUSINESS_INSIGHT_PROMPT
)

from src.talk_to_data.query_runner import execute_query
from src.utils.config import GEMINI_API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=GEMINI_API_KEY)


MODEL_NAMES = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-flash", "gemini-1.5-flash"]



def _call_gemini(prompt):
    last_error = None
    for model_name in MODEL_NAMES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini call with model {model_name} failed: {e}")
            last_error = e
    logger.error(f"All Gemini models failed. Last error: {last_error}")
    return None

def generate_sql(question):
    prompt = f"""
{SQL_SYSTEM_PROMPT}

User question:
{question}

Return only the SQL query.
"""

    try:
        raw_text = _call_gemini(prompt)
        if not raw_text:
            return None

        sql = raw_text.replace("```sql", "").replace("```", "").strip()
        return sql

    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        return None


def get_business_insight(question, sql, result):
    prompt = BUSINESS_INSIGHT_PROMPT.format(
        question=question,
        query=sql,
        result=result.to_string(index=False)
    )

    try:
        insight = _call_gemini(prompt)
        return insight if insight else "Unable to generate business insight."

    except Exception as e:
        logger.error(f"Error generating business insight: {e}")
        return "Unable to generate business insight."



def ask_data(question):
    sql = generate_sql(question)

    if not sql:
        return None, pd.DataFrame(), "Unable to generate SQL."

    try:
        result = execute_query(sql)

        if "Error" in result.columns:
            return sql, result, "The SQL query could not be executed."

        insight = get_business_insight(
            question,
            sql,
            result
        )

        return sql, result, insight

    except Exception as e:
        logger.error(f"Talk-to-Data error: {e}")

        return (
            sql,
            pd.DataFrame(),
            "Unable to process the question."
        )