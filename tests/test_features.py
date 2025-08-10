import numpy as np
from src.features import FeatureConfig, build_feature_table, save_features

def test_build_feature_table_shapes(cleaned_df_small, tmp_path):
    X, y = build_feature_table(cleaned_df_small, cfg=FeatureConfig(), scale_continuous=True)
    # rows match; target is 0/1
    assert X.shape[0] == y.shape[0]
    assert set(y.unique()) <= {0,1}
    # no non-numeric columns remain
    assert X.select_dtypes(exclude=np.number).empty

    # save and check files created
    outdir = tmp_path / "out"
    save_features(X, y, out_dir=str(outdir))
    assert (outdir / "features.csv").exists()
    assert (outdir / "target.csv").exists()

