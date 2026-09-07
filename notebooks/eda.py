#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk — Exploratory Data Analysis
# 
# ## Objective
# Understand the Home Credit dataset, identify data quality issues,
# analyse applicant demographics and financial characteristics, and
# derive business insights relevant to credit risk.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

print("Libraries loaded successfully.")


# 2. Load the dataset

# In[6]:


import os
import pandas as pd

DATA_PATH = "../data"

train_path = os.path.join(DATA_PATH, "application_train.csv")

df = pd.read_csv(train_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# 3. Understand the dataset

# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df.info()


# In[10]:


print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# 4. Dataset summary

# In[11]:


summary = pd.DataFrame({
    "Rows": [df.shape[0]],
    "Columns": [df.shape[1]],
    "Duplicate Rows": [df.duplicated().sum()],
    "Total Missing Values": [df.isnull().sum().sum()]
})

summary


# 5. Target variable — default distribution
# 
# 
#     TARGET means:
# 
#     0 = Loan repaid / no default
#     1 = Loan default

# In[12]:


target_counts = df["TARGET"].value_counts()

print(target_counts)


# In[13]:


target_percentage = df["TARGET"].value_counts(normalize=True) * 100

print(target_percentage.round(2))


# In[14]:


plt.figure(figsize=(7, 5))

df["TARGET"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Loan Default Distribution")
plt.xlabel("Target (0 = No Default, 1 = Default)")
plt.ylabel("Number of Applicants")
plt.xticks(rotation=0)

plt.show()


# In[15]:


default_rate = df["TARGET"].mean() * 100

print(f"Overall default rate: {default_rate:.2f}%")


# 6. Data types

# In[16]:


dtype_summary = df.dtypes.value_counts()

dtype_summary


# In[17]:


categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("Categorical columns:", len(categorical_columns))
print("Numeric columns:", len(numeric_columns))


# 7. Missing-value analysis

# In[18]:


missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": df.isnull().mean() * 100
})

missing = missing.sort_values(
    by="Missing Percentage",
    ascending=False
)

missing.head(20)


# In[19]:


missing_plot = missing[
    missing["Missing Percentage"] > 0
].head(20)

plt.figure(figsize=(10, 7))

missing_plot["Missing Percentage"].sort_values().plot(
    kind="barh"
)

plt.title("Top 20 Features by Missing Percentage")
plt.xlabel("Missing Values (%)")
plt.ylabel("Feature")

plt.show()


# 8. Duplicate analysis

# In[20]:


duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# 9. Numerical feature statistics

# In[21]:


df.describe().T


# 10. Applicant demographics

# In[22]:


gender_summary = pd.crosstab(
    df["CODE_GENDER"],
    df["TARGET"],
    normalize="index"
) * 100

gender_summary


# In[23]:


gender_summary.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Default Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.legend(
    ["No Default", "Default"],
    title="Loan Outcome"
)

plt.show()


# 11. Age analysis

# In[24]:


df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

df["AGE_YEARS"].describe()


# In[25]:


plt.figure(figsize=(9, 5))

plt.hist(
    df["AGE_YEARS"],
    bins=30
)

plt.title("Applicant Age Distribution")
plt.xlabel("Age (Years)")
plt.ylabel("Number of Applicants")

plt.show()


# 12. Age vs default

# In[26]:


df["AGE_GROUP"] = pd.cut(
    df["AGE_YEARS"],
    bins=[18, 25, 35, 45, 55, 65, 100],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "65+"
    ]
)

age_default = df.groupby(
    "AGE_GROUP",
    observed=True
)["TARGET"].mean() * 100

age_default


# In[27]:


plt.figure(figsize=(9, 5))

age_default.plot(kind="bar")

plt.title("Default Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=0)

plt.show()


# 13. Income analysis

# In[28]:


df["AMT_INCOME_TOTAL"].describe()


# In[29]:


plt.figure(figsize=(9, 5))

plt.hist(
    df["AMT_INCOME_TOTAL"],
    bins=50
)

plt.title("Applicant Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of Applicants")

plt.show()


# Because income has extreme outliers, also look at the 99th percentile.

# In[30]:


income_99 = df["AMT_INCOME_TOTAL"].quantile(0.99)

print("99th percentile:", income_99)

plt.figure(figsize=(9, 5))

plt.hist(
    df[df["AMT_INCOME_TOTAL"] <= income_99]["AMT_INCOME_TOTAL"],
    bins=50
)

plt.title("Applicant Income Distribution - Up to 99th Percentile")
plt.xlabel("Income")
plt.ylabel("Number of Applicants")

plt.show()


# 14. Income vs default

# In[31]:


df["INCOME_GROUP"] = pd.qcut(
    df["AMT_INCOME_TOTAL"],
    q=5,
    duplicates="drop"
)

income_default = df.groupby(
    "INCOME_GROUP",
    observed=True
)["TARGET"].mean() * 100

income_default


# In[32]:


plt.figure(figsize=(10, 5))

income_default.plot(kind="bar")

plt.title("Default Rate by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45)

plt.show()


# 15. Credit amount analysis
# 
# Important columns:
# 
# AMT_CREDIT
# AMT_ANNUITY
# AMT_GOODS_PRICE

# In[33]:


financial_features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]

df[financial_features].describe().T


# In[34]:


plt.figure(figsize=(9, 5))

plt.hist(
    df["AMT_CREDIT"],
    bins=50
)

plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount")
plt.ylabel("Number of Applicants")

plt.show()


# 16. Loan-to-income ratio

# In[35]:


df["CREDIT_INCOME_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_INCOME_TOTAL"]
)

df["CREDIT_INCOME_RATIO"].describe()


# In[36]:


df["CREDIT_INCOME_GROUP"] = pd.qcut(
    df["CREDIT_INCOME_RATIO"],
    q=5,
    duplicates="drop"
)

credit_income_default = df.groupby(
    "CREDIT_INCOME_GROUP",
    observed=True
)["TARGET"].mean() * 100

credit_income_default


# In[37]:


plt.figure(figsize=(10, 5))

credit_income_default.plot(kind="bar")

plt.title("Default Rate by Credit-to-Income Ratio")
plt.xlabel("Credit-to-Income Group")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45)

plt.show()


# 17. Education
# 
# Column:
# 
# NAME_EDUCATION_TYPE

# In[38]:


education_default = pd.crosstab(
    df["NAME_EDUCATION_TYPE"],
    df["TARGET"],
    normalize="index"
) * 100

education_default.sort_values(
    by=1,
    ascending=False
)


# In[39]:


default_by_education = (
    df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(10, 6))

default_by_education.plot(kind="bar")

plt.title("Default Rate by Education Level")
plt.xlabel("Education")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.show()


# 18. Employment
# 
# Column:
# 
# DAYS_EMPLOYED
# 
# Again, convert to years.

# In[40]:


df["EMPLOYMENT_YEARS"] = (
    -df["DAYS_EMPLOYED"] / 365
)

df["EMPLOYMENT_YEARS"].describe()


# In[41]:


plt.figure(figsize=(9, 5))

plt.hist(
    df[
        (df["EMPLOYMENT_YEARS"] >= 0) &
        (df["EMPLOYMENT_YEARS"] <= 50)
    ]["EMPLOYMENT_YEARS"],
    bins=40
)

plt.title("Employment Duration")
plt.xlabel("Employment Years")
plt.ylabel("Applicants")

plt.show()


# 19. Housing
# 
# Column:
# 
# NAME_HOUSING_TYPE

# In[42]:


housing_default = (
    df.groupby("NAME_HOUSING_TYPE")["TARGET"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

housing_default


# In[43]:


plt.figure(figsize=(10, 6))

housing_default.plot(kind="bar")

plt.title("Default Rate by Housing Type")
plt.xlabel("Housing Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.show()


# 20. Family status

# In[44]:


family_default = (
    df.groupby("NAME_FAMILY_STATUS")["TARGET"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

family_default


# In[45]:


plt.figure(figsize=(10, 5))

family_default.plot(kind="bar")

plt.title("Default Rate by Family Status")
plt.xlabel("Family Status")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45, ha="right")

plt.show()


# 21. Correlation with TARGET
# 

# In[46]:


numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()["TARGET"].sort_values(
    ascending=False
)

correlation.head(15)


# In[47]:


correlation.tail(15)


# 22. Top numerical correlations

# In[48]:


top_positive = correlation.drop("TARGET").head(10)
top_negative = correlation.drop("TARGET").tail(10)

print("Top positive correlations:")
display(top_positive)

print("\nTop negative correlations:")
display(top_negative)


# # Key Business Insights
# 
# Based on the exploratory analysis:
# 
# 1. **Class imbalance:** The majority of applicants do not default, while
#    default cases represent a much smaller portion of the dataset. This
#    creates a class-imbalance challenge for the ML model.
# 
# 2. **Age and credit risk:** Default rates vary across age groups,
#    indicating that applicant age contains useful predictive information.
# 
# 3. **Income and risk:** Default behaviour differs across income groups,
#    suggesting that income is an important variable for credit-risk
#    assessment.
# 
# 4. **Credit-to-income relationship:** The amount of credit requested
#    relative to applicant income shows a relationship with default risk.
#    This motivates creating a credit-to-income feature during model
#    development.
# 
# 5. **Education and risk:** Default rates differ between education
#    categories, indicating that demographic characteristics may contain
#    predictive information.
# 
# 6. **Employment stability:** Employment duration shows variation across
#    applicants and can provide useful information about financial stability.
# 
# 7. **Data quality:** Several variables contain substantial missing values,
#    requiring careful preprocessing before model training.
# 
# Important: after you run the notebook, adjust these statements to match your actual results. Don't claim a relationship that your charts don't show.
# 
# 24. Final EDA summary

# In[49]:


print("=" * 50)
print("EDA SUMMARY")
print("=" * 50)

print(f"Applicants: {df.shape[0]:,}")
print(f"Features: {df.shape[1]:,}")
print(f"Default Rate: {df['TARGET'].mean() * 100:.2f}%")
print(f"Duplicate Rows: {df.duplicated().sum():,}")
print(
    f"Features with Missing Values: "
    f"{(df.isnull().sum() > 0).sum()}"
)

