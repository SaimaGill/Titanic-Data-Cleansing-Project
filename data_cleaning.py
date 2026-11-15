"""
TITANIC DATA CLEANSING PROJECT
Dataset source: Kaggle Titanic - Machine Learning from Disaster
Download train.csv from:
https://www.kaggle.com/competitions/titanic/data
Place it in the Dataset folder, then run:
    python data_cleaning.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Dataset" / "train.csv"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        print("ERROR: train.csv not found.")
        print("Download train.csv from Kaggle and place it in the Dataset folder.")
        return

    df = pd.read_csv(DATA_FILE)
    original_rows, original_cols = df.shape
    report = []

    report.append("TITANIC DATA CLEANSING - EXECUTION REPORT")
    report.append("=" * 55)
    report.append(f"Original shape: {original_rows} rows x {original_cols} columns")
    report.append(f"Exact duplicate rows before cleaning: {df.duplicated().sum()}")
    report.append("\nMissing values before cleaning:")
    report.append(df.isna().sum().to_string())

    # 1. Standardize text values: trim whitespace and normalize text columns.
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
    report.append("\nStep 1 - Text whitespace cleaned.")

    # 2. Remove exact duplicate records.
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    report.append(f"Step 2 - Duplicate rows removed: {duplicates_removed}")

    # 3. Correct / validate data types.
    if "PassengerId" in df:
        df["PassengerId"] = pd.to_numeric(df["PassengerId"], errors="coerce").astype("Int64")
    if "Survived" in df:
        df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce").astype("Int64")
    if "Pclass" in df:
        df["Pclass"] = pd.to_numeric(df["Pclass"], errors="coerce").astype("Int64")
    for col in ["Age", "Fare"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["SibSp", "Parch"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    report.append("Step 3 - Numeric columns converted to appropriate numeric types.")

    # 4. Handle invalid values (values outside known logical ranges).
    if "Age" in df:
        invalid_age = int((df["Age"] < 0).sum())
        df.loc[df["Age"] < 0, "Age"] = np.nan
    else:
        invalid_age = 0
    if "Fare" in df:
        invalid_fare = int((df["Fare"] < 0).sum())
        df.loc[df["Fare"] < 0, "Fare"] = np.nan
    else:
        invalid_fare = 0
    report.append(f"Step 4 - Invalid negative Age values corrected: {invalid_age}; negative Fare values corrected: {invalid_fare}.")

    # 5. Handle missing values.
    missing_before = int(df.isna().sum().sum())
    if "Age" in df:
        # Group median preserves differences by passenger class and sex.
        if {"Pclass", "Sex"}.issubset(df.columns):
            df["Age"] = df.groupby(["Pclass", "Sex"], dropna=False)["Age"].transform(
                lambda s: s.fillna(s.median())
            )
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df:
        mode = df["Embarked"].mode(dropna=True)
        if not mode.empty:
            df["Embarked"] = df["Embarked"].fillna(mode.iloc[0])
    # Cabin has a very high proportion of missing entries in the standard Kaggle training file.
    # Retain the information as a binary indicator, then remove the sparse original column.
    if "Cabin" in df:
        df["CabinKnown"] = df["Cabin"].notna().astype("int64")
        df = df.drop(columns=["Cabin"])
    # Fill any remaining missing numeric values with median and categorical values with mode.
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                modes = df[col].mode(dropna=True)
                if not modes.empty:
                    df[col] = df[col].fillna(modes.iloc[0])
    missing_after = int(df.isna().sum().sum())
    report.append(f"Step 5 - Missing cells before: {missing_before}; after: {missing_after}.")
    report.append("Age: group median (Passenger Class and Sex), then overall median if needed.")
    report.append("Embarked: mode. Cabin: converted to CabinKnown indicator because of extensive missingness.")

    # 6. Handle inconsistent categorical values.
    for col in ["Sex", "Embarked"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip()
    if "Sex" in df:
        df["Sex"] = df["Sex"].str.lower()
    if "Embarked" in df:
        df["Embarked"] = df["Embarked"].str.upper()
    report.append("Step 6 - Categorical text values standardized (Sex lowercase; Embarked uppercase).")

    # 7. Detect outliers using IQR; flag them rather than automatically deleting valid passengers.
    outlier_summary = {}
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        if col in df and pd.api.types.is_numeric_dtype(df[col]):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            outlier_summary[col] = int(mask.sum())
            df[f"{col}_OutlierFlag"] = mask.astype("int64")
    report.append("Step 7 - Outliers detected with the IQR rule and flagged, not deleted.")
    for col, count in outlier_summary.items():
        report.append(f"  {col}: {count} potential outlier(s)")

    # 8. Validate key constraints and final missing/duplicate checks.
    validation = {
        "No duplicate rows": not df.duplicated().any(),
        "No missing cells": not df.isna().any().any(),
        "Age non-negative": ("Age" not in df) or bool((df["Age"] >= 0).all()),
        "Fare non-negative": ("Fare" not in df) or bool((df["Fare"] >= 0).all()),
        "Survived values are 0/1": ("Survived" not in df) or bool(df["Survived"].dropna().isin([0, 1]).all()),
    }
    report.append("\nStep 8 - Data validation:")
    for check, passed in validation.items():
        report.append(f"  {check}: {'PASS' if passed else 'CHECK'}")

    output_csv = OUTPUT_DIR / "cleaned_titanic.csv"
    df.to_csv(output_csv, index=False)
    report.append(f"\nCleaned dataset saved to: {output_csv}")
    report.append(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    report.append("\nMissing values after cleaning:")
    report.append(df.isna().sum().to_string())

    report_path = OUTPUT_DIR / "cleaning_execution_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print("\nSUCCESS: Cleaned CSV and execution report saved in Output folder.")

if __name__ == "__main__":
    main()
