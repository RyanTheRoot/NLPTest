# Contributing to Sentiment & Toxicity Analysis API

Thank you for your interest in contributing! This project is designed to be a clean, self-contained ML API reference implementation.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-toxicity-api.git
cd sentiment-toxicity-api

# Build the Docker image
make build

# Run tests
make test

# Start the API
make run

# Test manually
make analyze
```

## Project Structure

- `app.py` - FastAPI application with endpoints
- `inference.py` - Model backends (Transformer and TF-IDF)
- `models/bootstrap_models.py` - Model download and training
- `data/sample_tfidf.csv` - Training data for TF-IDF fallback
- `tests/test_api.py` - Test suite
- `Dockerfile` - Container build definition
- `Makefile` - Common development tasks

## Making Changes

### Before You Start

1. Check existing issues and PRs to avoid duplicates
2. For major changes, open an issue first to discuss
3. Keep changes focused and atomic

### Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style (black formatting, type hints)
   - Add tests for new functionality
   - Update README.md if adding features

3. **Test your changes**
   ```bash
   make test
   make analyze
   make offline
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat: add support for batch processing"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New features
   - `fix:` - Bug fixes
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

5. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- **Python**: Follow PEP 8, use type hints
- **Line length**: 100 characters max
- **Imports**: Group by stdlib, third-party, local
- **Docstrings**: Use Google style for functions and classes

## Testing Guidelines

- All new features must include tests
- Maintain test coverage above 80%
- Tests should run offline (no network mocking needed)
- Use descriptive test names: `test_feature_scenario_expected_outcome`

## What to Contribute

### Good First Issues

- Improve documentation
- Add more test cases
- Fix typos or formatting
- Add example scripts

### Feature Ideas

- Batch processing endpoint
- Additional model backends
- Performance optimizations
- Enhanced logging
- Prometheus metrics endpoint

### What We're Not Looking For

- External dependencies without strong justification
- Features requiring network access at runtime
- Breaking changes to the core API
- Platform-specific code

## Reporting Issues

### Bug Reports

Include:
- Steps to reproduce
- Expected vs actual behavior
- Docker image version or git SHA
- Error messages or logs
- Environment (OS, Docker version)

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered
- Willingness to implement

## Code Review Process

1. All PRs require at least one review
2. Tests must pass
3. Documentation must be updated
4. Commit history should be clean
5. PR description should explain the "why"

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

## Recognition

Contributors will be acknowledged in the README and release notes.

---

**Thank you for contributing!** 

