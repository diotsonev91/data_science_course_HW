import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------- core loaders/savers ----------
def load_raw_data(path="data/raw/asthma_disease_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def save_processed_data(df: pd.DataFrame, path="data/processed/asthma_clean.csv") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ---------- basic cleaning ----------
def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["PatientID"], errors="ignore")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric with median and categorical with mode."""
    out = df.copy()
    num_cols = out.select_dtypes(include=np.number).columns
    cat_cols = out.select_dtypes(exclude=np.number).columns

    out[num_cols] = out[num_cols].fillna(out[num_cols].median())
    for c in cat_cols:
        out[c] = out[c].fillna(out[c].mode().iloc[0])
    return out

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary categoricals; one-hot the rest."""
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
    """Example: add FEV1/FVC ratio (safe divide)."""
    out = df.copy()
    if {"LungFunctionFEV1", "LungFunctionFVC"}.issubset(out.columns):
        safe_fvc = out["LungFunctionFVC"].replace(0, np.nan)
        ratio = out["LungFunctionFEV1"] / safe_fvc
        out["FEV1_FVC_ratio"] = ratio.fillna(ratio.median())
    return out

def winsorize_continuous(
    df: pd.DataFrame, lower=0.01, upper=0.99, min_unique=11
) -> pd.DataFrame:
    """
    Cap continuous columns at given quantiles.
    Keeps binary/low-cardinality columns untouched.
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
def main():
    p = argparse.ArgumentParser(description="Preprocess asthma dataset.")
    p.add_argument("--in", dest="in_path", default="data/raw/asthma_disease_data.csv")
    p.add_argument("--out", dest="out_path", default="data/processed/asthma_clean.csv")
    p.add_argument("--no-fe", dest="no_fe", action="store_true",
                   help="Disable feature engineering (FEV1/FVC ratio).")
    p.add_argument("--no-winsor", dest="no_winsor", action="store_true",
                   help="Disable winsorizing continuous features.")
    p.add_argument("--winsor-lower", type=float, default=0.01)
    p.add_argument("--winsor-upper", type=float, default=0.99)
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

