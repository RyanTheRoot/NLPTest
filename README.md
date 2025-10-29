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

This downloads HuggingFace models and trains the TF-IDF fallback during the build process. The build captures the current git SHA for version tracking.

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

### Verify Offline Operation

Prove the API works with **zero network access**:

```bash
make offline
```

This runs the container with `--network none` and successfully makes requests, demonstrating that all models are truly baked into the image.

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

### GET /version

Version and model information.

**Response:**
```json
{
  "version": "1.0.0",
  "git_sha": "a1b2c3d",
  "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
  "toxicity_model": "unitary/toxic-bert"
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

You can verify with the offline test:

```bash
make offline
```

This proves the API works with `--network none` - a strong guarantee for airgapped or restricted environments.

## Performance Tips

- The first request after container start may be slower due to model loading
- Both backends warm up on startup with a dummy inference
- Use `--workers` flag with uvicorn for multi-process handling (increases memory usage)
- For batch processing, consider reusing the same container
- Request body size is capped at 1 MB to prevent abuse

## Security

- Container runs as non-root user (UID 1000)
- Request body size limited to 1 MB
- No external network dependencies at runtime
- Docker healthcheck monitors container status
- All dependencies pinned to specific versions

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Model Licenses

The pre-trained models used in this project have their own licenses:
- **DistilBERT** (sentiment): Apache 2.0
- **Toxic-BERT** (toxicity): Apache 2.0

Both models are freely available from HuggingFace Hub and compatible with commercial use. See [LICENSE](LICENSE) for full attribution.

