"""Feature engineering utilities for the asthma project.

This module turns a cleaned dataframe into a rectangular, all-numeric
feature table ready for modeling. It also provides simple engineering
steps (ratios, aggregates, symptom counts) and helpers to save outputs.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature building.

    Parameters
    ----------
    target_col : str
        Name of the target column.
    exposure_cols : tuple of str
        Column names that represent environmental exposures.
    symptom_cols : tuple of str
        Binary symptom flags to aggregate into a count.
    continuous_min_unique : int
        Minimum distinct values to treat a numeric feature as "continuous"
        (used when deciding what to scale).
    """

    target_col: str = "Diagnosis"
    exposure_cols: Tuple[str, ...] = ("DustExposure", "PollenExposure", "PollutionExposure")
    symptom_cols: Tuple[str, ...] = (
        "Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing", "NighttimeSymptoms", "ExerciseInduced"
    )
    continuous_min_unique: int = 11


def split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features X and target y.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe that includes the target column.
    target_col : str
        Name of the target.

    Returns
    -------
    X : pandas.DataFrame
        Features (all columns except the target).
    y : pandas.Series
        Target labels (cast to int).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataframe.")
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def add_feature_engineering(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Add simple engineered features in a vectorized way.

    Adds:
    - ``FEV1_FVC_ratio`` if FEV1 and FVC exist.
    - ``ExposureMean`` as the mean of exposure columns (if all present).
    - ``SymptomCount`` as the row-wise sum of binary symptom flags.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe.
    cfg : FeatureConfig
        Column configuration.

    Returns
    -------
    pandas.DataFrame
        Dataframe with engineered columns added (idempotent).
    """
    out = df.copy()

    if {"LungFunctionFEV1", "LungFunctionFVC"}.issubset(out.columns) and "FEV1_FVC_ratio" not in out.columns:
        ratio = out["LungFunctionFEV1"] / out["LungFunctionFVC"].replace(0, np.nan)
        out["FEV1_FVC_ratio"] = ratio.fillna(ratio.median())

    if all(c in out.columns for c in cfg.exposure_cols):
        out["ExposureMean"] = out[list(cfg.exposure_cols)].mean(axis=1)

    present_symptoms = [c for c in cfg.symptom_cols if c in out.columns]
    if present_symptoms:
        out["SymptomCount"] = out[present_symptoms].sum(axis=1)

    return out


def select_continuous_columns(df: pd.DataFrame, cfg: FeatureConfig) -> List[str]:
    """Return numeric columns that look continuous (not binary).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    cfg : FeatureConfig
        Threshold settings.

    Returns
    -------
    list of str
        Column names to treat as continuous.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in num_cols if df[c].nunique() >= cfg.continuous_min_unique]


def standardize(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Z-score standardize selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    cols : iterable of str
        Columns to standardize.

    Returns
    -------
    pandas.DataFrame
        Copy with standardized columns.
    """
    out = df.copy()
    cols = [c for c in cols if c in out.columns]
    if cols:
        means = out[cols].mean()
        stds = out[cols].std(ddof=0).replace(0, 1.0)
        out[cols] = (out[cols] - means) / stds
    return out


def build_feature_table(
    df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
    scale_continuous: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """Produce an all-numeric feature table and the target.

    Assumes the dataframe has already been cleaned/encoded.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe containing the target column.
    cfg : FeatureConfig, optional
        Column configuration; defaults will be used if omitted.
    scale_continuous : bool, default True
        Whether to standardize continuous columns only.

    Returns
    -------
    X : pandas.DataFrame
        Numeric features (engineered + original).
    y : pandas.Series
        Target labels (0/1).

    Raises
    ------
    ValueError
        If non-numeric columns remain in X.
    """
    cfg = cfg or FeatureConfig()
    df_eng = add_feature_engineering(df, cfg)
    X, y = split_xy(df_eng, cfg.target_col)

    if X.select_dtypes(exclude=np.number).shape[1] > 0:
        raise ValueError("Non-numeric columns found in X; ensure preprocessing encoded everything.")

    if scale_continuous:
        cont_cols = select_continuous_columns(X, cfg)
        X = standardize(X, cont_cols)

    return X, y


def save_features(X: pd.DataFrame, y: pd.Series, out_dir: str = "data/processed") -> None:
    """Save features and target to CSV files.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target labels.
    out_dir : str, default "data/processed"
        Output directory to write ``features.csv`` and ``target.csv``.
    """
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    X.to_csv(od / "features.csv", index=False)
    y.to_csv(od / "target.csv", index=False, header=True)

