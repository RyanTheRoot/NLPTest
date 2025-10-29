# Sentiment & Toxicity Analysis API

A self-contained offline API for analyzing text sentiment and toxicity. Runs entirely within a single Docker container with no external dependencies or network access required at runtime.

## Features

- **Sentiment Analysis**: Classifies text as POSITIVE or NEGATIVE with confidence scores
- **Toxicity Detection**: Scores text toxicity from 0 (clean) to 1 (toxic)
- **Dual Backend**: Transformer models (default) or TF-IDF fallback
- **Fully Offline**: All model weights baked into the Docker image
- **Fast**: Cold start under 3s, inference under 100ms for short text
- **Compact**: Image size under 1.5 GB

## Quick Start

### Build the Image

```bash
make build
```

This downloads HuggingFace models and trains the TF-IDF fallback during the build process.

### Run the API

```bash
make run
```

The API will be available at `http://localhost:8000`

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Analyze text
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Or use the convenient target:

```bash
make analyze
```

### Run Tests

```bash
make test
```

### Stop the Container

```bash
make stop
```

## API Reference

### POST /analyze

Analyzes text for sentiment and toxicity.

**Request:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.9998
  },
  "toxicity": 0.0156,
  "model_backend": "transformer",
  "latency_ms": 45.2
}
```

**Fields:**
- `sentiment.label`: POSITIVE or NEGATIVE
- `sentiment.score`: Confidence score (0-1)
- `toxicity`: Toxicity score (0-1), higher is more toxic
- `model_backend`: transformer or tfidf
- `latency_ms`: Processing time in milliseconds

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Model Backends

### Transformer (Default)

Uses state-of-the-art HuggingFace models:
- **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Toxicity**: `unitary/toxic-bert`

More accurate but slightly larger and slower.

### TF-IDF Fallback

Uses scikit-learn TF-IDF vectorizer + LinearSVC trained on a small dataset.

Switch backends with the `MODEL_BACKEND` environment variable:

```bash
docker run -e MODEL_BACKEND=tfidf -p 8000:8000 sentiment-toxicity-api
```

## Offline Guarantee

This API requires **zero network access** at runtime:

1. All transformer model weights are downloaded during `docker build`
2. The TF-IDF model is trained during `docker build` on bundled data
3. `TRANSFORMERS_OFFLINE=1` environment variable enforces offline mode
4. No external API calls or database connections

You can verify by running the container with network disabled:

```bash
docker run --network none -p 8000:8000 sentiment-toxicity-api
```

## Performance Tips

- The first request after container start may be slower due to model loading
- Both backends warm up on startup with a dummy inference
- Use `--workers` flag with uvicorn for multi-process handling (increases memory usage)
- For batch processing, consider reusing the same container

## Development

### Project Structure

```
.
├── app.py                      # FastAPI application
├── inference.py                # Backend implementations
├── models/
│   ├── bootstrap_models.py     # Model download and training script
│   ├── vectorizer.joblib       # TF-IDF artifacts (generated)
│   └── linear_svc.joblib       # TF-IDF artifacts (generated)
├── data/
│   └── sample_tfidf.csv        # Training data for TF-IDF
├── tests/
│   └── test_api.py             # Pytest smoke tests
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build instructions
├── Makefile                    # Common tasks
└── README.md                   # This file
```

### Requirements

- Docker
- Make (optional, for convenience targets)

### Local Development

To run outside Docker (requires Python 3.11):

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (first time only)
python models/bootstrap_models.py

# Run the API
uvicorn app:app --reload
```

## License

This project is provided as-is for demonstration purposes.

