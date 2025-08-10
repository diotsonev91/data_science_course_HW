"""CLI to build model-ready features from the cleaned dataset.

Usage
-----
python src/build_features.py --in data/processed/asthma_clean.csv --outdir data/processed
"""

import argparse
from pathlib import Path

import pandas as pd

from src.features import FeatureConfig, build_feature_table, save_features


def main() -> None:
    """Parse args, build features, and save to disk."""
    ap = argparse.ArgumentParser(description="Build ML-ready feature table.")
    ap.add_argument("--in", dest="in_path", default="data/processed/asthma_clean.csv",
                    help="Path to cleaned CSV (output of preprocessing).")
    ap.add_argument("--outdir", dest="out_dir", default="data/processed",
                    help="Directory to store features.csv and target.csv.")
    ap.add_argument("--no-scale", dest="no_scale", action="store_true",
                    help="Disable standardization of continuous columns.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_path)
    X, y = build_feature_table(df, cfg=FeatureConfig(), scale_continuous=not args.no_scale)
    save_features(X, y, args.out_dir)
    print(f"Saved features to {args.out_dir}/features.csv and {args.out_dir}/target.csv  | X:{X.shape} y:{y.shape}")


if __name__ == "__main__":
    main()

