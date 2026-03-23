import numpy as np
import pandas as pd
import yaml
import os
import logging

from sklearn.feature_extraction.text import CountVectorizer

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE   = "feature_engineering.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)          # Master level — handlers filter further

# Console handler — INFO and above printed to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# File handler — DEBUG and above written to a log file for full traceability
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

# Attach both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ─────────────────────────────────────────────────────────────────────────────


def load_params(params_path: str) -> int:
    """Load max_features parameter from a YAML config file."""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)

        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features={max_features} from '{params_path}'")
        return max_features

    except FileNotFoundError:
        logger.error(f"Config file not found: '{params_path}'")
        raise
    except KeyError as e:
        logger.error(f"Missing expected key in config: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file '{params_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading params: {e}")
        raise


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        logger.info(f"Train data loaded from '{train_path}' — shape: {train_data.shape}")

        test_data = pd.read_csv(test_path)
        logger.info(f"Test data loaded  from '{test_path}'  — shape: {test_data.shape}")

        return train_data, test_data

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


def validate_columns(df: pd.DataFrame, required_cols: list[str], label: str) -> None:
    """Check that all required columns exist in the DataFrame."""
    try:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {label} data: {missing}")
        logger.debug(f"Column validation passed for {label} data.")
    except KeyError as e:
        logger.error(f"Column validation failed: {e}")
        raise


def fill_missing_values(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaN values with empty strings to avoid vectorizer errors."""
    try:
        train_nulls = train_data['content'].isna().sum()
        test_nulls  = test_data['content'].isna().sum()

        train_data.fillna('', inplace=True)
        test_data.fillna('',  inplace=True)

        logger.debug(f"Filled {train_nulls} NaN(s) in train data.")
        logger.debug(f"Filled {test_nulls} NaN(s) in test data.")
        logger.info("Missing values filled with empty strings.")

        return train_data, test_data

    except Exception as e:
        logger.error(f"Unexpected error while filling missing values: {e}")
        raise


def extract_features(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    """Apply Bag of Words vectorization to train and test content."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test  = test_data['content'].values
        y_test  = test_data['sentiment'].values

        logger.info(f"Applying CountVectorizer with max_features={max_features}.")

        vectorizer = CountVectorizer(max_features=max_features)

        # Fit on training data and transform — never fit on test data
        X_train_bow = vectorizer.fit_transform(X_train)
        logger.info(f"Train BoW matrix shape: {X_train_bow.shape}")

        # Transform test data using the already-fitted vectorizer
        X_test_bow = vectorizer.transform(X_test)
        logger.info(f"Test  BoW matrix shape: {X_test_bow.shape}")

        return X_train_bow, y_train, X_test_bow, y_test

    except KeyError as e:
        logger.error(f"Missing column during feature extraction: {e}")
        raise
    except ValueError as e:
        logger.error(f"ValueError during vectorization (check max_features or empty content): {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during feature extraction: {e}")
        raise


def build_dataframes(X_train_bow, y_train, X_test_bow, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert sparse BoW matrices to DataFrames and attach labels."""
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        logger.debug(f"Train DataFrame built — shape: {train_df.shape}")

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        logger.debug(f"Test  DataFrame built — shape: {test_df.shape}")

        return train_df, test_df

    except ValueError as e:
        logger.error(f"Shape mismatch while building DataFrames: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while building DataFrames: {e}")
        raise


def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save BoW feature DataFrames as CSV files to the given directory."""
    try:
        os.makedirs(data_path, exist_ok=True)       # Fix: original had no exist_ok, crashes on re-run

        train_out = os.path.join(data_path, "train_bow.csv")
        test_out  = os.path.join(data_path, "test_bow.csv")

        train_df.to_csv(train_out, index=False)
        logger.info(f"Train BoW features saved to '{train_out}' — shape: {train_df.shape}")

        test_df.to_csv(test_out, index=False)
        logger.info(f"Test  BoW features saved to '{test_out}'  — shape: {test_df.shape}")

    except OSError as e:
        logger.error(f"Failed to create directory or write files at '{data_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving feature data: {e}")
        raise


def main():
    try:
        logger.info("=" * 60)
        logger.info("Feature engineering pipeline started.")

        # Load config
        max_features = load_params('params.yaml')

        # Load processed data
        train_data, test_data = load_data(
            train_path='./data/processed/train_processed.csv',
            test_path ='./data/processed/test_processed.csv'
        )

        # Validate required columns exist
        validate_columns(train_data, ['content', 'sentiment'], label='train')
        validate_columns(test_data,  ['content', 'sentiment'], label='test')

        # Handle missing values
        train_data, test_data = fill_missing_values(train_data, test_data)

        # Extract BoW features
        X_train_bow, y_train, X_test_bow, y_test = extract_features(
            train_data, test_data, max_features
        )

        # Build final DataFrames with labels
        train_df, test_df = build_dataframes(X_train_bow, y_train, X_test_bow, y_test)

        # Save to data/features
        data_path = os.path.join("data", "features")
        save_data(data_path, train_df, test_df)

        logger.info("Feature engineering pipeline completed successfully.")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Pipeline aborted: a required file was not found.")
    except KeyError as e:
        logger.error(f"Pipeline aborted: missing column or config key — {e}")
    except ValueError as e:
        logger.error(f"Pipeline aborted: data validation error — {e}")
    except Exception as e:
        logger.error(f"Pipeline aborted due to an unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()