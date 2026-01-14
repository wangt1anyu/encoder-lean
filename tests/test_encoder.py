"""Tests for the LeanEncoder class."""

import pytest
import numpy as np
import torch
from encoder_lean.encoder import LeanEncoder


class TestLeanEncoder:
    """Test suite for LeanEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create a LeanEncoder instance for testing."""
        # Use CPU for testing to avoid GPU requirements
        return LeanEncoder(device="cpu")
    
    @pytest.fixture
    def sample_lean_code(self):
        """Sample Lean code for testing."""
        return "def add (a b : Nat) : Nat := a + b"
    
    @pytest.fixture
    def sample_lean_codes(self):
        """Multiple Lean code samples for testing."""
        return [
            "def add (a b : Nat) : Nat := a + b",
            "theorem add_zero (n : Nat) : n + 0 = n := rfl",
            "inductive List (α : Type) where | nil : List α | cons : α → List α → List α"
        ]
    
    def test_encoder_initialization(self, encoder):
        """Test that encoder initializes correctly."""
        assert encoder is not None
        assert encoder.device == "cpu"
        assert encoder.tokenizer is not None
        assert encoder.model is not None
    
    def test_encoder_cpu_device(self):
        """Test encoder initialization with CPU device."""
        encoder = LeanEncoder(device="cpu")
        assert encoder.device == "cpu"
    
    def test_encode_single_string(self, encoder, sample_lean_code):
        """Test encoding a single Lean code string."""
        embeddings = encoder.encode(sample_lean_code)
        
        # Check that embeddings are numpy array
        assert isinstance(embeddings, np.ndarray)
        
        # Check shape: should be (1, embedding_dim)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0  # Should have some embedding dimension
    
    def test_encode_list(self, encoder, sample_lean_codes):
        """Test encoding a list of Lean code strings."""
        embeddings = encoder.encode(sample_lean_codes)
        
        # Check that embeddings are numpy array
        assert isinstance(embeddings, np.ndarray)
        
        # Check shape: should be (batch_size, embedding_dim)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == len(sample_lean_codes)
        assert embeddings.shape[1] > 0
    
    def test_encode_return_tensors(self, encoder, sample_lean_code):
        """Test encoding with return_tensors=True."""
        embeddings = encoder.encode(sample_lean_code, return_tensors=True)
        
        # Check that embeddings are torch tensors
        assert isinstance(embeddings, torch.Tensor)
        
        # Check shape
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1
    
    def test_encode_with_pooling_mean(self, encoder, sample_lean_code):
        """Test encode_with_pooling with mean pooling."""
        embeddings = encoder.encode_with_pooling(sample_lean_code, pooling="mean")
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0
    
    def test_encode_with_pooling_max(self, encoder, sample_lean_code):
        """Test encode_with_pooling with max pooling."""
        embeddings = encoder.encode_with_pooling(sample_lean_code, pooling="max")
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1
    
    def test_encode_with_pooling_cls(self, encoder, sample_lean_code):
        """Test encode_with_pooling with cls pooling."""
        embeddings = encoder.encode_with_pooling(sample_lean_code, pooling="cls")
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1
    
    def test_encode_with_pooling_invalid(self, encoder, sample_lean_code):
        """Test encode_with_pooling with invalid pooling strategy."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder.encode_with_pooling(sample_lean_code, pooling="invalid")
    
    def test_encode_max_length(self, encoder, sample_lean_code):
        """Test encoding with custom max_length."""
        embeddings = encoder.encode(sample_lean_code, max_length=256)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
    
    def test_encode_batch_consistency(self, encoder, sample_lean_codes):
        """Test that batch encoding produces consistent results."""
        # Encode as batch
        batch_embeddings = encoder.encode(sample_lean_codes)
        
        # Encode individually
        individual_embeddings = [encoder.encode(code) for code in sample_lean_codes]
        individual_embeddings = np.vstack(individual_embeddings)
        
        # Shapes should match
        assert batch_embeddings.shape == individual_embeddings.shape
        
        # Values should be the same (or very close due to floating point)
        np.testing.assert_array_almost_equal(batch_embeddings, individual_embeddings, decimal=5)
    
    def test_encode_empty_string(self, encoder):
        """Test encoding an empty string."""
        embeddings = encoder.encode("")
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
    
    def test_encode_long_code(self, encoder):
        """Test encoding longer Lean code."""
        long_code = """
        def fibonacci (n : Nat) : Nat :=
          match n with
          | 0 => 0
          | 1 => 1
          | n + 2 => fibonacci n + fibonacci (n + 1)
        """
        embeddings = encoder.encode(long_code)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1


class TestDeviceHandling:
    """Test device handling in LeanEncoder."""
    
    def test_auto_device_selection(self):
        """Test automatic device selection."""
        encoder = LeanEncoder()
        # Should select a valid device
        assert encoder.device in ["cpu", "cuda"]
    
    def test_explicit_cpu(self):
        """Test explicit CPU device selection."""
        encoder = LeanEncoder(device="cpu")
        assert encoder.device == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explicit_cuda(self):
        """Test explicit CUDA device selection (only if CUDA is available)."""
        encoder = LeanEncoder(device="cuda")
        assert encoder.device.startswith("cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
