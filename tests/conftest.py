# tests/conftest.py
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def raw_df_small():
    """Tiny raw-like frame (with PatientID, some NaNs, mixed dtypes)."""
    df = pd.DataFrame({
        "PatientID": [1001, 1002, 1003, 1004],
        "Diagnosis": [0, 1, 0, 0],
        "Gender": ["M", "F", "M", np.nan],            # cat with a NaN
        "Age": [40, 55, 33, 60],
        "BMI": [26.1, np.nan, 31.2, 22.9],            # numeric with a NaN
        "LungFunctionFEV1": [2.1, 3.0, 2.5, 2.8],
        "LungFunctionFVC": [4.1, 5.0, 0.0, 3.9],      # one zero to test safe divide
        "DustExposure": [7, 6, 9, 4],
        "PollenExposure": [5, 8, 6, 7],
        "PollutionExposure": [6, 7, 8, 5],
        "Wheezing": [1, 0, 1, 0],
        "Coughing": [0, 1, 0, 1],
    })
    return df

@pytest.fixture
def cleaned_df_small(raw_df_small):
    """Run your actual cleaning pipeline to get a small cleaned frame."""
    from src.preprocessing import clean_data
    return clean_data(raw_df_small, do_feature_engineering=True, do_winsorize=False)

