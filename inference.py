"""Inference backends for sentiment and toxicity analysis."""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any
import joblib
import numpy as np


class InferenceService(ABC):
    """Base class for inference services."""
    
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for sentiment and toxicity.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment (label, score) and toxicity score
        """
        pass
    
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        pass
    
    @classmethod
    def from_env(cls) -> "InferenceService":
        """Factory method to create backend from environment."""
        backend = os.getenv("MODEL_BACKEND", "transformer").lower()
        
        if backend == "tfidf":
            return TfidfBackend()
        else:
            return TransformerBackend()


class TransformerBackend(InferenceService):
    """Transformer-based inference using HuggingFace models."""
    
    def __init__(self):
        """Initialize both sentiment and toxicity pipelines."""
        from transformers import pipeline
        
        print("Loading transformer models...")
        
        # Load sentiment model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU only
        )
        
        # Load toxicity model
        self.toxicity_pipeline = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1  # CPU only
        )
        
        # Warm up pipelines
        print("Warming up pipelines...")
        self.sentiment_pipeline("test")
        self.toxicity_pipeline("test")
        print("Transformer backend ready")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text using transformer models."""
        # Get sentiment
        sentiment_result = self.sentiment_pipeline(text)[0]
        
        # Get toxicity
        toxicity_result = self.toxicity_pipeline(text)[0]
        
        # toxic-bert returns "toxic" or "non-toxic"
        toxicity_score = (
            toxicity_result["score"] 
            if toxicity_result["label"] == "toxic" 
            else 1.0 - toxicity_result["score"]
        )
        
        return {
            "sentiment": {
                "label": sentiment_result["label"],
                "score": sentiment_result["score"]
            },
            "toxicity": toxicity_score
        }
    
    def backend_name(self) -> str:
        return "transformer"


class TfidfBackend(InferenceService):
    """TF-IDF based fallback inference."""
    
    def __init__(self):
        """Load pre-trained TF-IDF artifacts."""
        print("Loading TF-IDF models...")
        
        self.vectorizer = joblib.load("models/vectorizer.joblib")
        self.classifier = joblib.load("models/linear_svc.joblib")
        
        # Toxic keyword dictionary for toxicity scoring
        self.toxic_keywords = {
            "stupid", "idiot", "hate", "kill", "moron", "dumb", 
            "trash", "horrible", "awful", "terrible", "worst",
            "useless", "pathetic", "loser", "fool", "jerk"
        }
        
        # Warm up
        print("Warming up TF-IDF pipeline...")
        self.analyze("test")
        print("TF-IDF backend ready")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text using TF-IDF classifier."""
        # Get sentiment prediction
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.classifier.predict(text_vectorized)[0]
        decision_scores = self.classifier.decision_function(text_vectorized)[0]
        
        # Convert decision score to probability-like score
        # Using sigmoid-like transformation
        prob = 1 / (1 + np.exp(-decision_scores))
        
        # Map prediction to label
        if prediction == "positive":
            label = "POSITIVE"
            score = float(prob)
        else:
            label = "NEGATIVE"
            score = float(1.0 - prob)
        
        # Calculate toxicity based on keyword presence
        text_lower = text.lower()
        toxic_count = sum(1 for word in self.toxic_keywords if word in text_lower)
        
        # Sigmoid mapping: more toxic keywords = higher toxicity
        toxicity_raw = toxic_count / 3.0
        toxicity_score = float(1 / (1 + np.exp(-toxicity_raw)))
        
        return {
            "sentiment": {
                "label": label,
                "score": score
            },
            "toxicity": toxicity_score
        }
    
    def backend_name(self) -> str:
        return "tfidf"

