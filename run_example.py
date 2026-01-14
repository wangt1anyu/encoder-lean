"""Simple script to run the encoder example from the root directory."""

import argparse
import torch
from encoder_lean.encoder import LeanEncoder

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
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    elif args.cpu:
        device = "cpu"
    elif args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("⚠ Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            device = "cpu"
    else:
        device = None  # Auto-detect (will use GPU if available)
    
    # Initialize the encoder
    print("Loading encoder model (this may take a moment on first run)...")
    encoder = LeanEncoder(device=device)
    
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
    print("\nEncoding Lean code...")
    embeddings = encoder.encode(lean_examples)
    
    print(f"✓ Encoded {len(lean_examples)} examples")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Encode with different pooling strategies
    print("\nTrying different pooling strategies...")
    mean_embeddings = encoder.encode_with_pooling(lean_examples[0], pooling="mean")
    print(f"✓ Mean pooling shape: {mean_embeddings.shape}")

if __name__ == "__main__":
    main()
