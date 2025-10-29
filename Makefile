.PHONY: build run test analyze offline stop clean help

IMAGE_NAME := sentiment-toxicity-api
CONTAINER_NAME := sentiment-toxicity-api-container
PORT := 8000
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")

help:
	@echo "Available targets:"
	@echo "  build    - Build the Docker image with git SHA"
	@echo "  run      - Run the container on localhost:8000"
	@echo "  test     - Run pytest tests inside container"
	@echo "  analyze  - Send example curl request to /analyze"
	@echo "  offline  - Prove offline execution with --network none"
	@echo "  stop     - Stop and remove the container"
	@echo "  clean    - Remove container and image"

build:
	@echo "Building Docker image with git SHA: $(GIT_SHA)"
	docker build --build-arg GIT_SHA=$(GIT_SHA) -t $(IMAGE_NAME) .

run:
	@echo "Starting container on port $(PORT)..."
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 $(IMAGE_NAME)
	@echo "Container running at http://localhost:$(PORT)"
	@echo "Health check: curl http://localhost:$(PORT)/health"

test:
	@echo "Running tests in container..."
	docker run --rm -e PYTHONPATH=/app $(IMAGE_NAME) pytest tests/test_api.py -v

analyze:
	@echo "Sending example requests..."
	@echo ""
	@echo "Test 1: JSON endpoint - Positive sentiment"
	curl -X POST http://localhost:$(PORT)/analyze \
		-H "Content-Type: application/json" \
		-d '{"text": "I love this product, it works great!"}' \
		| python3 -m json.tool
	@echo ""
	@echo "Test 2: JSON endpoint - Negative/toxic text"
	curl -X POST http://localhost:$(PORT)/analyze \
		-H "Content-Type: application/json" \
		-d '{"text": "This is terrible and you are stupid"}' \
		| python3 -m json.tool
	@echo ""
	@echo "Test 3: Form data endpoint - Multi-line text"
	@printf "This is a great product!\n\nI highly recommend it to everyone." | \
		curl -X POST http://localhost:$(PORT)/analyze/text \
		-F "text=<-" \
		| python3 -m json.tool

offline:
	@echo "=========================================="
	@echo "Proving offline execution with --network none"
	@echo "=========================================="
	@echo ""
	@echo "Starting container with NO network access..."
	docker run --rm --network none --name $(CONTAINER_NAME)-offline -d $(IMAGE_NAME)
	@echo "Waiting for startup (models loading)..."
	@sleep 8
	@echo ""
	@echo "✓ Container running without network"
	@echo ""
	@echo "Testing from INSIDE container (only way with --network none):"
	@echo ""
	@echo "1. Health check..."
	@docker exec $(CONTAINER_NAME)-offline python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
	@echo ""
	@echo "2. Version endpoint..."
	@docker exec $(CONTAINER_NAME)-offline python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/version').read().decode())"
	@echo ""
	@echo "3. Analyze endpoint (proves models work offline)..."
	@docker exec $(CONTAINER_NAME)-offline python -c "import urllib.request, json; req = urllib.request.Request('http://localhost:8000/analyze', data=json.dumps({'text': 'This works offline!'}).encode(), headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode())"
	@echo ""
	@echo "✓ All requests succeeded with --network none"
	@echo "✓ Offline guarantee verified: models loaded from baked-in cache"
	@docker stop $(CONTAINER_NAME)-offline
	@echo ""
	@echo "=========================================="

stop:
	@echo "Stopping container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean: stop
	@echo "Removing image..."
	-docker rmi $(IMAGE_NAME)

