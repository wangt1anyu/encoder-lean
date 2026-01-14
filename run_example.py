"""Simple script to run the encoder example from the root directory."""

import argparse
import logging
import torch
from encoder_lean.encoder import LeanEncoder
from encoder_lean.logger import setup_logger, get_default_log_file

def main():
    parser = argparse.ArgumentParser(description="Run Lean encoder example")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU (cuda) if available"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specify device explicitly (e.g., 'cuda', 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/encoder_lean_TIMESTAMP.log)"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging to file"
    )
    args = parser.parse_args()
    
    # Set up logging
    log_file = None if args.no_log else (args.log_file or get_default_log_file())
    if log_file:
        setup_logger("encoder_lean", log_file=log_file, console_output=True)
        logger = logging.getLogger("encoder_lean")
        logger.info("=" * 60)
        logger.info("Starting Lean Encoder Example")
        logger.info("=" * 60)
    else:
        logger = logging.getLogger("encoder_lean")
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
    
    # Determine device
    if args.device:
        device = args.device
    elif args.cpu:
        device = "cpu"
    elif args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning("GPU requested but CUDA is not available. Falling back to CPU.")
            device = "cpu"
    else:
        device = None  # Auto-detect (will use GPU if available)
    
    # Initialize the encoder
    logger.info("Loading encoder model (this may take a moment on first run)...")
    encoder = LeanEncoder(device=device, log_file=log_file)
    
    # Example Lean code snippets
    lean_examples = [
        """
        def add (a b : Nat) : Nat :=
          a + b
        """,
        """
        theorem add_zero (n : Nat) : n + 0 = n :=
          rfl
        """,
        """
        inductive List (α : Type) where
          | nil : List α
          | cons : α → List α → List α
        """
    ]
    
    # Encode the Lean code
    logger.info("Encoding Lean code...")
    embeddings = encoder.encode(lean_examples)
    
    logger.info(f"✓ Encoded {len(lean_examples)} examples")
    logger.info(f"✓ Embedding shape: {embeddings.shape}")
    logger.info(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Encode with different pooling strategies
    logger.info("Trying different pooling strategies...")
    mean_embeddings = encoder.encode_with_pooling(lean_examples[0], pooling="mean")
    logger.info(f"✓ Mean pooling shape: {mean_embeddings.shape}")
    
    if log_file:
        logger.info("=" * 60)
        logger.info(f"Example completed. Log saved to: {log_file}")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()
