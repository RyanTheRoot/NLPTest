"""Bootstrap script to download models and train TF-IDF artifacts during Docker build."""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def download_transformer_models():
    """Download and cache HuggingFace transformer models."""
    print("=" * 60)
    print("Downloading transformer models...")
    print("=" * 60)
    
    from transformers import pipeline
    
    # Download sentiment model
    print("\n[1/2] Downloading sentiment model (distilbert-sst-2)...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    # Test it
    result = sentiment_pipeline("This is a test")
    print(f"✓ Sentiment model loaded: {result}")
    
    # Download toxicity model
    print("\n[2/2] Downloading toxicity model (toxic-bert)...")
    toxicity_pipeline = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=-1
    )
    # Test it
    result = toxicity_pipeline("This is a test")
    print(f"✓ Toxicity model loaded: {result}")
    
    print("\n✓ All transformer models downloaded successfully")


def train_tfidf_models():
    """Train TF-IDF + LinearSVC models on sample data."""
    print("=" * 60)
    print("Training TF-IDF models...")
    print("=" * 60)
    
    # Load training data
    data_path = Path("data/sample_tfidf.csv")
    if not data_path.exists():
        print(f"✗ Training data not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    print(f"\n✓ Loaded {len(df)} training examples")
    
    # Prepare data
    X = df["text"].values
    y = df["label"].values
    
    # Convert toxic to negative for binary classification
    y = ["negative" if label == "toxic" else label for label in y]
    
    print(f"  - Positive examples: {sum(1 for label in y if label == 'positive')}")
    print(f"  - Negative examples: {sum(1 for label in y if label == 'negative')}")
    
    # Train TF-IDF vectorizer
    print("\nTraining TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X)
    
    # Train LinearSVC classifier
    print("Training LinearSVC classifier...")
    classifier = LinearSVC(random_state=42, max_iter=1000)
    classifier.fit(X_vectorized, y)
    
    # Test the model
    test_texts = ["I love this", "This is terrible"]
    test_vectorized = vectorizer.transform(test_texts)
    predictions = classifier.predict(test_vectorized)
    print(f"\n✓ Model test predictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"  '{text}' -> {pred}")
    
    # Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    vectorizer_path = models_dir / "vectorizer.joblib"
    classifier_path = models_dir / "linear_svc.joblib"
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)
    
    print(f"\n✓ Saved vectorizer to {vectorizer_path}")
    print(f"✓ Saved classifier to {classifier_path}")


def main():
    """Main bootstrap function."""
    print("\n" + "=" * 60)
    print("MODEL BOOTSTRAP STARTING")
    print("=" * 60)
    
    # Set HF_HOME if not already set
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/opt/models"
    
    print(f"\nHF_HOME: {os.environ.get('HF_HOME')}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Download transformer models
        download_transformer_models()
        
        # Train TF-IDF models
        train_tfidf_models()
        
        print("\n" + "=" * 60)
        print("✓ BOOTSTRAP COMPLETE")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Bootstrap failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

