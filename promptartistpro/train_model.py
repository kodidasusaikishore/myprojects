"""
train_model.py — Intent Classifier Training Script

Trains an SVM (LinearSVC) classifier on TF-IDF features to map
user queries to intent categories. Pure ML — no LLMs or APIs.

Usage:
    python train_model.py

Output:
    models/intent_classifier.joblib  — serialized pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessor import preprocess_batch

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'training_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'intent_classifier.joblib')


def load_data(path: str) -> tuple:
    """Load and validate training data from CSV."""
    print(f"[DATA] Loading training data from: {path}")
    df = pd.read_csv(path)

    # Validate columns
    assert 'text' in df.columns and 'intent' in df.columns, \
        "CSV must have 'text' and 'intent' columns."

    # Drop any rows with missing values
    df.dropna(subset=['text', 'intent'], inplace=True)

    print(f"  [OK] Loaded {len(df)} samples across {df['intent'].nunique()} categories")
    print(f"\n[STATS] Class Distribution:")
    print(df['intent'].value_counts().to_string())
    print()

    return df['text'].tolist(), df['intent'].tolist()


def build_pipeline() -> Pipeline:
    """
    Build the ML pipeline:
      TF-IDF Vectorizer -> LinearSVC Classifier

    TF-IDF params are tuned for short-text classification:
      - ngram_range=(1,2): captures bigrams like "test case", "delta hedge"
      - max_features=5000: limits vocabulary for generalization
      - sublinear_tf=True: applies log normalization to TF
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
        )),
        ('classifier', LinearSVC(
            C=1.0,
            class_weight='balanced',  # handles class imbalance
            max_iter=10000,
            random_state=42,
        )),
    ])
    return pipeline


def train_and_evaluate(texts: list, labels: list) -> tuple:
    """
    Train the pipeline and evaluate with stratified cross-validation.

    Returns:
        (pipeline, label_encoder) -- trained and ready to serialize.
    """
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Preprocess all texts
    print("[PREP] Preprocessing texts...")
    X_processed = preprocess_batch(texts)

    # Build pipeline
    pipeline = build_pipeline()

    # -- Cross-Validation -------------------------------------------------------
    print("[CV] Running 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_processed, y, cv=cv, scoring='accuracy')

    print(f"   Fold Accuracies : {np.round(scores, 3)}")
    print(f"   Mean Accuracy   : {scores.mean():.3f} (+/-{scores.std():.3f})")
    print()

    # -- Train on full dataset --------------------------------------------------
    print("[TRAIN] Training final model on full dataset...")
    pipeline.fit(X_processed, y)

    # Classification report on training data (for quick sanity check)
    y_pred = pipeline.predict(X_processed)
    print("\n[REPORT] Classification Report (on training data):")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # Confusion matrix
    print("[MATRIX] Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    # Pretty print with labels
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    print()

    return pipeline, le


def save_model(pipeline, label_encoder, path: str):
    """Serialize the trained pipeline and label encoder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
    }
    joblib.dump(artifact, path)
    file_size = os.path.getsize(path) / 1024
    print(f"[SAVE] Model saved to: {path} ({file_size:.1f} KB)")


def main():
    print("=" * 60)
    print("  PROMPTING TECHNIQUES ENGINE -- Model Training")
    print("  Pure ML/NLP | No LLMs | No API calls")
    print("=" * 60)
    print()

    # Load data
    texts, labels = load_data(DATA_PATH)

    # Train & evaluate
    pipeline, label_encoder = train_and_evaluate(texts, labels)

    # Save
    save_model(pipeline, label_encoder, MODEL_PATH)

    print("\n[DONE] Training complete! You can now run the engine:")
    print("   streamlit run app.py")
    print()


if __name__ == '__main__':
    main()
