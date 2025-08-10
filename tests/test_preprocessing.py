import numpy as np
import pandas as pd
from src import preprocessing as pp

def test_drop_identifier_columns(raw_df_small):
    out = pp.drop_identifier_columns(raw_df_small)
    assert "PatientID" not in out.columns

def test_handle_missing_values_imputes(raw_df_small):
    out = pp.handle_missing_values(raw_df_small)
    assert out["BMI"].isna().sum() == 0
    assert out["Gender"].isna().sum() == 0

def test_encode_categoricals_makes_numeric(raw_df_small):
    # ensure cats exist then encode
    df = pp.handle_missing_values(raw_df_small)
    out = pp.encode_categoricals(df)
    # gender should now be numeric (label encoded) or one-hot removed
    assert out.select_dtypes(exclude=np.number).empty

def test_add_feature_engineering_ratio(cleaned_df_small):
    # your clean_data already adds FEV1_FVC_ratio when enabled
    assert "FEV1_FVC_ratio" in cleaned_df_small.columns
    # no inf/NaN produced by zero-div issues
    s = cleaned_df_small["FEV1_FVC_ratio"]
    assert np.isfinite(s).all()

def test_winsorize_only_continuous():
    df = pd.DataFrame({
        "Diagnosis": [0]*50 + [1]*50,
        "binary_col": [0,1]*50,
        "cont": np.r_[np.linspace(0, 1, 99), 50.0],  # big outlier at end
    })
    out = pp.winsorize_continuous(df, lower=0.05, upper=0.95, min_unique=11)
    # binary untouched (still only 0/1)
    assert set(out["binary_col"].unique()) <= {0, 1}
    # outlier clipped below 50
    assert out["cont"].max() < 50

