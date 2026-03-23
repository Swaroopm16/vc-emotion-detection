import numpy as np
import pandas as pd
import yaml
import os
import logging

from sklearn.model_selection import train_test_split

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE   = "data_ingestion.log"

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


def load_params(params_path: str) -> float:
    """Load test_size parameter from a YAML config file."""
    try:
        with open(params_path, 'r') as f:                          # Fix: was hardcoded string 'params_path'
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.info(f"Loaded test_size={test_size} from '{params_path}'")
        return test_size

    except FileNotFoundError:
        logger.error(f"Config file not found: '{params_path}'")
        raise
    except KeyError as e:
        logger.error(f"Missing expected key in config: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file '{params_path}': {e}")
        raise


def read_data(url: str) -> pd.DataFrame:
    """Read CSV data from a local path or remote URL."""
    try:
        df = pd.read_csv(url)
        logger.info(f"Data loaded successfully from '{url}' — shape: {df.shape}")
        return df

    except FileNotFoundError:
        logger.error(f"Data file not found: '{url}'")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Data file is empty: '{url}'")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file '{url}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading data from '{url}': {e}")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, filter sentiments, and encode labels."""
    try:
        if 'tweet_id' not in df.columns:
            logger.warning("Column 'tweet_id' not found — skipping drop step.")
        else:
            df.drop(columns=['tweet_id'], inplace=True)

        if 'sentiment' not in df.columns:
            raise KeyError("Required column 'sentiment' is missing from the DataFrame.")

        final_df = df[df['sentiment'].isin(['neutral', 'sadness'])].copy()

        if final_df.empty:
            raise ValueError("No rows remain after filtering for 'neutral' and 'sadness' sentiments.")

        final_df['sentiment'].replace({'neutral': 0, 'sadness': 1}, inplace=True)
        logger.info(f"Data processed successfully — shape after filtering: {final_df.shape}")
        return final_df

    except KeyError as e:
        logger.error(f"Column error during processing: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error during processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data processing: {e}")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save train and test DataFrames as CSV files to the given directory."""
    try:
        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.csv")
        test_path  = os.path.join(data_path, "test.csv")

        # Save training data into a CSV file
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"Train data saved to '{train_path}' — shape: {train_data.shape}")
        logger.info(f"Test data saved  to '{test_path}'  — shape: {test_data.shape}")

    except OSError as e:
        logger.error(f"Failed to create directory or write files at '{data_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise


def main():
    try:
        test_size  = load_params('params.yaml')
        df         = read_data('src/tweet_emotions.csv')
        final_df   = process_data(df)                              # Fix: was final_data (undefined in split call)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )

        # Makes two nested folders: data/raw
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion pipeline completed successfully.")

    except FileNotFoundError:
        logger.error("Pipeline aborted: a required file was not found.")
    except KeyError:
        logger.error("Pipeline aborted: a required config key or column was missing.")
    except ValueError:
        logger.error("Pipeline aborted: a data validation check failed.")
    except Exception as e:
        logger.error(f"Pipeline aborted due to an unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()