# Running Tests

## Prerequisites

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_encoder.py -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=encoder_lean --cov-report=html
```

## Test Structure

- `tests/test_encoder.py` - Comprehensive tests for the LeanEncoder class
  - Tests for encoding single strings and lists
  - Tests for different pooling strategies
  - Tests for device handling
  - Tests for edge cases
