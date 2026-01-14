# Encoder for Lean

An LLM-based encoder for Lean theorem prover code. This project provides tools to encode Lean code into high-dimensional embeddings using pre-trained transformer models, enabling downstream tasks such as code similarity, search, and machine learning on formal proofs.

## Features

- 🚀 **Pre-trained Models**: Uses state-of-the-art code language models (CodeBERT by default)
- 🔄 **Flexible Pooling**: Multiple pooling strategies (CLS, mean, max)
- 🖥️ **GPU Support**: Automatic GPU detection with manual override options
- 📦 **Easy to Use**: Simple API for encoding single or batch Lean code
- ⚡ **Efficient**: Batch processing support for encoding multiple examples

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd encoder-lean
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Command Line Usage

Run the example script:

```bash
# Auto-detect device (uses GPU if available)
python run_example.py

# Explicitly use GPU
python run_example.py --gpu

# Force CPU
python run_example.py --cpu

# Specify device explicitly
python run_example.py --device cuda:0
```

### Python API Usage

```python
from encoder_lean.encoder import LeanEncoder

# Initialize the encoder (auto-detects GPU if available)
encoder = LeanEncoder()

# Encode a single Lean code snippet
lean_code = """
def add (a b : Nat) : Nat :=
  a + b
"""
embeddings = encoder.encode(lean_code)
print(f"Embedding shape: {embeddings.shape}")

# Encode multiple examples
lean_examples = [
    "def add (a b : Nat) : Nat := a + b",
    "theorem add_zero (n : Nat) : n + 0 = n := rfl"
]
batch_embeddings = encoder.encode(lean_examples)
print(f"Batch embedding shape: {batch_embeddings.shape}")
```

## GPU Options

The encoder supports GPU acceleration with flexible device selection:

### Command Line Options

- `--gpu`: Use GPU (CUDA) if available, fallback to CPU
- `--cpu`: Force CPU usage
- `--device <device>`: Specify device explicitly (e.g., `cuda`, `cuda:0`, `cpu`)

### Python API Options

```python
# Use GPU
encoder = LeanEncoder(device="cuda")

# Use specific GPU
encoder = LeanEncoder(device="cuda:0")

# Force CPU
encoder = LeanEncoder(device="cpu")

# Auto-detect (default)
encoder = LeanEncoder()  # Uses GPU if available
```

## API Reference

### `LeanEncoder`

Main encoder class for Lean code.

#### `__init__(model_name: str = "microsoft/codebert-base", device: Optional[str] = None)`

Initialize the encoder.

**Parameters:**
- `model_name` (str): Name of the pre-trained model to use (default: "microsoft/codebert-base")
- `device` (str, optional): Device to run the model on ('cuda', 'cpu', or None for auto-detection)

#### `encode(lean_code: Union[str, List[str]], max_length: int = 512, return_tensors: bool = False) -> torch.Tensor`

Encode Lean code into embeddings using the CLS token.

**Parameters:**
- `lean_code` (str | List[str]): Single Lean code string or list of strings
- `max_length` (int): Maximum sequence length (default: 512)
- `return_tensors` (bool): Whether to return PyTorch tensors (default: False, returns numpy arrays)

**Returns:**
- Embeddings as numpy array or PyTorch tensor

#### `encode_with_pooling(lean_code: Union[str, List[str]], max_length: int = 512, pooling: str = "mean") -> np.ndarray`

Encode with different pooling strategies.

**Parameters:**
- `lean_code` (str | List[str]): Single Lean code string or list of strings
- `max_length` (int): Maximum sequence length (default: 512)
- `pooling` (str): Pooling strategy - 'mean', 'max', or 'cls' (default: "mean")

**Returns:**
- Embeddings as numpy array

## Examples

See the `examples/` directory for more detailed usage examples:

- `basic_usage.py`: Basic encoding examples
- `run_example.py`: Command-line example with GPU options

## Project Structure

```
encoder-lean/
├── encoder_lean/          # Main package
│   ├── __init__.py
│   └── encoder.py         # Core encoder implementation
├── examples/              # Example scripts
│   └── basic_usage.py
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup configuration
├── run_example.py        # Quick start example script
└── README.md             # This file
```

## Requirements

- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.30.0` - Hugging Face transformers library
- `tokenizers>=0.13.0` - Fast tokenization
- `numpy>=1.24.0` - Numerical operations
- `datasets>=2.14.0` - Dataset handling (optional, for future features)
- `tqdm>=4.65.0` - Progress bars

## Model Information

By default, this project uses **CodeBERT** (`microsoft/codebert-base`), a pre-trained model for programming languages. You can use other models by specifying the `model_name` parameter:

```python
encoder = LeanEncoder(model_name="microsoft/codebert-base-mlm")
```

Other compatible models include:
- `microsoft/codebert-base`
- `microsoft/codebert-base-mlm`
- Any BERT-based model compatible with the transformers library

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Acknowledgments

- Uses [CodeBERT](https://github.com/microsoft/CodeBERT) by Microsoft
- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
