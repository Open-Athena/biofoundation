import pytest
import torch
from biofoundation.model.base import EmbeddingModel


def test_embedding_model_is_abstract():
    """Test that EmbeddingModel cannot be instantiated."""
    with pytest.raises(TypeError):
        EmbeddingModel()


def test_concrete_embedding_model():
    """Test a concrete implementation of EmbeddingModel."""

    class DummyEmbedding(EmbeddingModel):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.hidden_dim = hidden_dim

        def forward(self, input_ids):
            B, L = input_ids.shape
            return torch.randn(B, L, self.hidden_dim)

    model = DummyEmbedding(hidden_dim=256)
    input_ids = torch.randint(0, 100, (4, 10))
    embeddings = model(input_ids)

    assert embeddings.shape == (4, 10, 256)


def test_embedding_model_output_shape():
    """Test that EmbeddingModel returns correct output shape."""

    class SimpleEmbedding(EmbeddingModel):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 64)

        def forward(self, input_ids):
            return self.embed(input_ids)

    model = SimpleEmbedding()

    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 5),  # Single sequence, short
        (4, 10),  # Small batch
        (8, 128),  # Larger batch and sequence
    ]

    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        embeddings = model(input_ids)
        assert embeddings.shape == (batch_size, seq_len, 64)
