"""Basic usage example for the Lean encoder."""

from encoder_lean.encoder import LeanEncoder

def main():
    # Initialize the encoder
    encoder = LeanEncoder()
    
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
    print("Encoding Lean code...")
    embeddings = encoder.encode(lean_examples)
    
    print(f"Encoded {len(lean_examples)} examples")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Encode with different pooling strategies
    print("\nTrying different pooling strategies...")
    mean_embeddings = encoder.encode_with_pooling(lean_examples[0], pooling="mean")
    print(f"Mean pooling shape: {mean_embeddings.shape}")

if __name__ == "__main__":
    main()
