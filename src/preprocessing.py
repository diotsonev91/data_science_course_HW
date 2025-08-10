"""Preprocessing pipeline for the asthma dataset.

This module loads the raw CSV, applies reproducible cleaning steps,
and writes a cleaned file to ``data/processed/asthma_clean.csv`` by default.

CLI
---
Run from the repo root:

    python src/preprocessing.py \
        --in data/raw/asthma_disease_data.csv \
        --out data/processed/asthma_clean.csv

Flags:
    --no-fe         Disable feature engineering (FEV1/FVC ratio).
    --no-winsor     Disable winsorizing continuous features.
    --winsor-lower  Lower quantile for winsorizing (default 0.01).
    --winsor-upper  Upper quantile for winsorizing (default 0.99).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------- core loaders/savers ----------
def load_raw_data(path: str = "data/raw/asthma_disease_data.csv") -> pd.DataFrame:
    """Load the raw asthma dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the raw CSV.

    Returns
    -------
    pandas.DataFrame
        Raw dataframe as read from disk.
    """
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/asthma_clean.csv") -> None:
    """Save a processed dataframe to CSV, creating the folder if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to save.
    path : str
        Destination CSV path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------- basic cleaning ----------
def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID-like columns that could cause leakage.

    Currently drops ``PatientID`` if present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy without identifier columns.
    """
    return df.drop(columns=["PatientID"], errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (numeric→median, categorical→mode).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy with missing values imputed.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=np.number).columns
    cat_cols = out.select_dtypes(exclude=np.number).columns

    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    for c in cat_cols:
        out[c] = out[c].fillna(out[c].mode().iloc[0])
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns (binary → label encode, else one-hot).

    - If ``Gender`` exists and is binary, label-encode it to {0,1}.
    - All remaining non-numeric columns are one-hot encoded with ``drop_first=True``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy with categorical columns encoded numerically.
    """
    out = df.copy()

    if "Gender" in out.columns and out["Gender"].nunique() == 2:
        le = LabelEncoder()
        out["Gender"] = le.fit_transform(out["Gender"])

    cat_cols = out.select_dtypes(exclude=np.number).columns
    if len(cat_cols):
        out = pd.get_dummies(out, columns=list(cat_cols), drop_first=True)
    return out


# ---------- “better cleaning” knobs (still model-agnostic) ----------
def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple engineered features (safe, model-agnostic).

    Adds:
    - ``FEV1_FVC_ratio`` = FEV1 / FVC (guarding against zeros/NaNs).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Copy with engineered features.
    """
    out = df.copy()
    if {"LungFunctionFEV1", "LungFunctionFVC"}.issubset(out.columns):
        safe_fvc = out["LungFunctionFVC"].replace(0, np.nan)
        ratio = out["LungFunctionFEV1"] / safe_fvc
        out["FEV1_FVC_ratio"] = ratio.fillna(ratio.median())
    return out


def winsorize_continuous(
    df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99, min_unique: int = 11
) -> pd.DataFrame:
    """Cap continuous features at given quantiles (winsorize).

    Binary/low-cardinality columns are left untouched.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    lower : float, default 0.01
        Lower quantile for clipping.
    upper : float, default 0.99
        Upper quantile for clipping.
    min_unique : int, default 11
        Minimum unique values to consider a column "continuous".

    Returns
    -------
    pandas.DataFrame
        Copy with continuous columns clipped to quantile bounds.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=np.number).columns
    cont_cols = [c for c in num_cols if c != "Diagnosis" and out[c].nunique() >= min_unique]
    for c in cont_cols:
        ql, qh = out[c].quantile([lower, upper])
        out[c] = out[c].clip(ql, qh)
    return out


# ---------- master pipeline ----------
def clean_data(
    df: pd.DataFrame,
    *,
    do_feature_engineering: bool = True,
    do_winsorize: bool = True,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Steps:
    1) Drop identifier columns
    2) Impute missing values (median/mode)
    3) Encode categoricals (binary label-encode, else one-hot)
    4) (Optional) Add simple engineered features
    5) (Optional) Winsorize continuous features

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataframe.
    do_feature_engineering : bool, default True
        Whether to add engineered features (e.g., FEV1/FVC ratio).
    do_winsorize : bool, default True
        Whether to winsorize continuous columns.
    winsor_lower : float, default 0.01
        Lower quantile for winsorizing.
    winsor_upper : float, default 0.99
        Upper quantile for winsorizing.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe ready for EDA/feature building.
    """
    out = df.copy()
    out = drop_identifier_columns(out)
    out = handle_missing_values(out)
    out = encode_categoricals(out)
    if do_feature_engineering:
        out = add_feature_engineering(out)
    if do_winsorize:
        out = winsorize_continuous(out, lower=winsor_lower, upper=winsor_upper)
    return out


# ---------- CLI entrypoint ----------
def main() -> None:
    """Parse CLI args, run the cleaning pipeline, and save to disk."""
    p = argparse.ArgumentParser(description="Preprocess asthma dataset.")
    p.add_argument("--in", dest="in_path", default="data/raw/asthma_disease_data.csv",
                   help="Path to raw CSV.")
    p.add_argument("--out", dest="out_path", default="data/processed/asthma_clean.csv",
                   help="Destination for cleaned CSV.")
    p.add_argument("--no-fe", dest="no_fe", action="store_true",
                   help="Disable feature engineering (FEV1/FVC ratio).")
    p.add_argument("--no-winsor", dest="no_winsor", action="store_true",
                   help="Disable winsorizing continuous features.")
    p.add_argument("--winsor-lower", type=float, default=0.01,
                   help="Lower quantile for winsorizing.")
    p.add_argument("--winsor-upper", type=float, default=0.99,
                   help="Upper quantile for winsorizing.")
    args = p.parse_args()

    df_raw = load_raw_data(args.in_path)
    df_clean = clean_data(
        df_raw,
        do_feature_engineering=not args.no_fe,
        do_winsorize=not args.no_winsor,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    save_processed_data(df_clean, args.out_path)
    print(f"Saved processed dataset: {args.out_path}  shape={df_clean.shape}")


if __name__ == "__main__":
    main()

