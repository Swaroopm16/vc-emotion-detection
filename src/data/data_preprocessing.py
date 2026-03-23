import pandas as pd
import numpy as np
import os
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE   = "data_preprocessing.log"

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


def download_nltk_resources() -> None:
    """Download required NLTK resources with error handling."""
    resources = ['wordnet', 'stopwords', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.debug(f"NLTK resource '{resource}' downloaded/verified successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource '{resource}': {e}")
            raise


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSV files from the given paths."""
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


# ── Text transformation functions ─────────────────────────────────────────────

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logger.warning(f"Lemmatization failed for text snippet '{text[:40]}...': {e}")
        return text                      # Return original text as fallback


def remove_stop_words(text: str) -> str:
    """Remove English stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in str(text).split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logger.warning(f"Stop word removal failed for text snippet '{text[:40]}...': {e}")
        return text


def removing_numbers(text: str) -> str:
    """Strip all digit characters from the text."""
    try:
        return ''.join([ch for ch in text if not ch.isdigit()])
    except Exception as e:
        logger.warning(f"Number removal failed for text snippet '{text[:40]}...': {e}")
        return text


def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logger.warning(f"Lowercase conversion failed for text snippet '{text[:40]}...': {e}")
        return text


def removing_punctuations(text: str) -> str:
    """Remove punctuation characters and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        logger.warning(f"Punctuation removal failed for text snippet '{text[:40]}...': {e}")
        return text


def removing_urls(text: str) -> str:
    """Remove http/https and www URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.warning(f"URL removal failed for text snippet '{text[:40]}...': {e}")
        return text


def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentences with fewer than 3 words with NaN."""
    try:
        # Fix: use .loc to avoid SettingWithCopyWarning; original used iloc assignment
        mask = df['content'].apply(lambda x: len(str(x).split()) < 3)
        df.loc[mask, 'content'] = np.nan
        removed = mask.sum()
        logger.debug(f"Removed {removed} short sentence(s) (< 3 words) — set to NaN.")
        return df
    except KeyError:
        logger.error("Column 'content' not found while removing small sentences.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in remove_small_sentences: {e}")
        raise


# ── Main normalization pipeline ───────────────────────────────────────────────

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text cleaning pipeline to the 'content' column."""
    try:
        if 'content' not in df.columns:
            raise KeyError("Column 'content' is missing from the DataFrame.")

        original_count = len(df)
        logger.info(f"Starting text normalization on {original_count} rows.")

        steps = [
            ("lower_case",             lower_case),
            ("remove_stop_words",      remove_stop_words),
            ("removing_numbers",       removing_numbers),
            ("removing_punctuations",  removing_punctuations),
            ("removing_urls",          removing_urls),
            ("lemmatization",          lemmatization),
        ]

        for step_name, func in steps:
            try:
                df['content'] = df['content'].apply(func)
                logger.debug(f"Step '{step_name}' applied successfully.")
            except Exception as e:
                logger.error(f"Step '{step_name}' failed: {e}")
                raise

        logger.info("Text normalization completed successfully.")
        return df

    except KeyError as e:
        logger.error(f"Column error during normalization: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during text normalization: {e}")
        raise


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save processed train and test DataFrames to the given directory."""
    try:
        os.makedirs(data_path, exist_ok=True)           # Fix: original had no exist_ok, would crash on re-run

        train_out = os.path.join(data_path, "train_processed.csv")
        test_out  = os.path.join(data_path, "test_processed.csv")

        train_data.to_csv(train_out, index=False)
        logger.info(f"Processed train data saved to '{train_out}' — shape: {train_data.shape}")

        test_data.to_csv(test_out, index=False)
        logger.info(f"Processed test data saved  to '{test_out}'  — shape: {test_data.shape}")

    except OSError as e:
        logger.error(f"Failed to create directory or write files at '{data_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving processed data: {e}")
        raise


def main():
    try:
        logger.info("=" * 60)
        logger.info("Data preprocessing pipeline started.")

        download_nltk_resources()

        train_data, test_data = load_data(
            train_path='./data/raw/train.csv',
            test_path ='./data/raw/test.csv'
        )

        train_processed = normalize_text(train_data)
        test_processed  = normalize_text(test_data)

        data_path = os.path.join("data", "processed")
        save_data(data_path, train_processed, test_processed)

        logger.info("Data preprocessing pipeline completed successfully.")
        logger.info("=" * 60)

    except FileNotFoundError:
        logger.error("Pipeline aborted: a required data file was not found.")
    except KeyError as e:
        logger.error(f"Pipeline aborted: missing column or key — {e}")
    except Exception as e:
        logger.error(f"Pipeline aborted due to an unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()