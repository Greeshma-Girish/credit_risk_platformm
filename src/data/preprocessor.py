import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def preprocess_data(df, is_train=True):
    # Copy dataframe
    df_copy = df.copy()

    # Separate features and target
    if "TARGET" in df_copy.columns:
        y = df_copy["TARGET"]
        X = df_copy.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    else:
        y = None
        X = df_copy.drop(columns=["SK_ID_CURR"], errors="ignore")

    # Handle missing values
    num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
    cat_cols = list(X.select_dtypes(include=['object', 'category']).columns)

    for col in num_cols:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col] = X[col].fillna(median_val)


    # Encode categorical variables
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna('missing')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


def balance_data(X, y):
    # Optional: balance data using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
