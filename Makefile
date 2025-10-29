.PHONY: build run test analyze stop clean help

IMAGE_NAME := sentiment-toxicity-api
CONTAINER_NAME := sentiment-toxicity-api-container
PORT := 8000

help:
	@echo "Available targets:"
	@echo "  build    - Build the Docker image"
	@echo "  run      - Run the container on localhost:8000"
	@echo "  test     - Run pytest tests inside container"
	@echo "  analyze  - Send example curl request to /analyze"
	@echo "  stop     - Stop and remove the container"
	@echo "  clean    - Remove container and image"

build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Starting container on port $(PORT)..."
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 $(IMAGE_NAME)
	@echo "Container running at http://localhost:$(PORT)"
	@echo "Health check: curl http://localhost:$(PORT)/health"

test:
	@echo "Running tests in container..."
	docker run --rm -e PYTHONPATH=/app $(IMAGE_NAME) pytest tests/test_api.py -v

analyze:
	@echo "Sending example request to /analyze..."
	@echo ""
	@echo "Test 1: Positive sentiment"
	curl -X POST http://localhost:$(PORT)/analyze \
		-H "Content-Type: application/json" \
		-d '{"text": "I love this product, it works great!"}' \
		| python3 -m json.tool
	@echo ""
	@echo "Test 2: Negative/toxic text"
	curl -X POST http://localhost:$(PORT)/analyze \
		-H "Content-Type: application/json" \
		-d '{"text": "This is terrible and you are stupid"}' \
		| python3 -m json.tool

stop:
	@echo "Stopping container..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

clean: stop
	@echo "Removing image..."
	-docker rmi $(IMAGE_NAME)

