import sqlite3
import pandas as pd

from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_sql(sql_query):
    sql = sql_query.strip().lower()

    if not sql.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    forbidden = ["insert","update","delete","drop","alter","create","replace","truncate","attach","pragma"]

    for keyword in forbidden:
        if keyword in sql:
            raise ValueError(
                f"Forbidden SQL operation detected: {keyword}"
            )

    return True


def execute_query(sql_query):
    logger.info(f"Executing SQL query: {sql_query}")

    try:
        validate_sql(sql_query)

        conn = sqlite3.connect(DB_PATH)

        try:
            df = pd.read_sql_query(sql_query, conn)
        finally:
            conn.close()

        return df

    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return pd.DataFrame({"Error": [str(e)]})