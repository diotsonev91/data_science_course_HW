# Asthma Data Science Project

## Overview
This project demonstrates a complete data science workflow for predicting factors leading to a positive asthma diagnosis.
It follows a professional project structure with separate directories for data, notebooks, source code, and tests.

## Project Structure
```
data_science_course_HW/
│
├── data/
│   ├── raw/            # Unmodified datasets
│   └── processed/      # Cleaned, preprocessed datasets
│
├── notebooks/          # Jupyter notebooks for EDA, preprocessing, modeling
│
├── src/                # Python modules for reusable logic
│   └── preprocessing.py
│
├── tests/              # Unit tests
│
├── environment.yml     # Conda environment definition
├── setup.py            # Makes src/ installable as a package
└── README.md
```

## Setup

### 1. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate asthma-lab
```

### 2. Install the project in editable mode
This ensures that modules inside `src/` can be imported anywhere.
```bash
pip install -e .
```

### 3. Run Jupyter Notebooks
```bash
jupyter notebook
```
Open any notebook from the `notebooks/` directory.

## Usage
Example import from within a notebook or Python script:
```python
from src.preprocessing import load_raw_data, clean_data, save_processed_data

df_raw = load_raw_data("data/raw/asthma_disease_data.csv")
df_clean = clean_data(df_raw)
save_processed_data(df_clean, "data/processed/asthma_clean.csv")
```

## License
This project is for educational purposes only.

