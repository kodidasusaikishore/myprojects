"""
preprocessor.py — NLTK-based Text Preprocessing Pipeline

Cleans, tokenizes, and normalizes text for the intent classifier.
No LLMs or Generative AI APIs used.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ── Ensure NLTK data is available ──────────────────────────────────────────
def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_data()

# ── Module-level objects (initialized once) ────────────────────────────────
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words('english'))

# Keep domain-critical words that NLTK would normally strip
_KEEP_WORDS = {
    'not', 'no', 'need', 'help', 'how', 'what', 'which', 'why',
    'can', 'should', 'would', 'could',
}
_stop_words -= _KEEP_WORDS


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs and emails
      3. Remove punctuation (keep hyphens in compound words)
      4. Tokenize with NLTK
      5. Remove stopwords
      6. Apply Porter stemming
      7. Return cleaned string

    Args:
        text: Raw user input string.

    Returns:
        Cleaned, stemmed, space-joined string ready for TF-IDF.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers that aren't part of words (keep e.g. "pytest3")
    text = re.sub(r'\b\d+\b', '', text)

    # Remove punctuation except hyphens in compound words
    text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and stem
    processed = [
        _stemmer.stem(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]

    return ' '.join(processed)


def preprocess_batch(texts: list) -> list:
    """
    Apply preprocessing to a list of texts.

    Args:
        texts: List of raw text strings.

    Returns:
        List of preprocessed strings.
    """
    return [preprocess(t) for t in texts]
