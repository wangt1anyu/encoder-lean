"""Main encoder module for Lean code."""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional


class LeanEncoder:
    """Encoder for Lean code using pre-trained language models."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None
    ):
        """
        Initialize the Lean encoder.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device '{device}' requested but CUDA is not available")
        
        # Load tokenizer and model
        print(f"Loading model '{model_name}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def encode(
        self,
        lean_code: Union[str, List[str]],
        max_length: int = 512,
        return_tensors: bool = False
    ) -> torch.Tensor:
        """
        Encode Lean code into embeddings.
        
        Args:
            lean_code: Single Lean code string or list of strings
            max_length: Maximum sequence length
            return_tensors: Whether to return tensors or numpy arrays
            
        Returns:
            Encoded representations of the Lean code
        """
        # Handle single string or list
        if isinstance(lean_code, str):
            lean_code = [lean_code]
        
        # Tokenize
        encoded = self.tokenizer(
            lean_code,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding (first token) or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        if not return_tensors:
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def encode_with_pooling(
        self,
        lean_code: Union[str, List[str]],
        max_length: int = 512,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Encode with different pooling strategies.
        
        Args:
            lean_code: Single Lean code string or list of strings
            max_length: Maximum sequence length
            pooling: Pooling strategy ('mean', 'max', 'cls')
            
        Returns:
            Encoded representations
        """
        if isinstance(lean_code, str):
            lean_code = [lean_code]
        
        encoded = self.tokenizer(
            lean_code,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state
            
            if pooling == "cls":
                embeddings = hidden_states[:, 0, :]
            elif pooling == "mean":
                # Mean pooling (excluding padding tokens)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            elif pooling == "max":
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).max(1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return embeddings.cpu().numpy()
