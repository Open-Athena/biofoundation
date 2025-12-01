from jaxtyping import Float, Int
from torch import Tensor
from typing import Any, cast

from biofoundation.model.base import CausalLM, MaskedLM


class GLMExperimentsMaskedLM(MaskedLM):
    """Adapter for glm-experiments MLM (Masked Language Model) PyTorch Lightning modules.

    This adapter wraps models from the glm-experiments repository that follow the
    MLMLitModule pattern. The adapter expects the extracted PyTorch model (the .net
    attribute of the Lightning module), not the Lightning module itself.

    Usage:
        from glm_experiments.models.lm_lit_module import MLMLitModule
        from biofoundation.model.adapters.glm_experiments import GLMExperimentsMaskedLM
        from biofoundation.model.adapters.hf import HFTokenizer
        from transformers import AutoTokenizer

        # Load checkpoint using PyTorch Lightning
        lit_module = MLMLitModule.load_from_checkpoint("path/to/checkpoint.ckpt")

        # Extract and wrap the PyTorch model
        model = GLMExperimentsMaskedLM(lit_module.net)

        # Load tokenizer separately
        tokenizer = HFTokenizer(AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm"))

        # Use with biofoundation inference functions
        from biofoundation.inference import run_llr_mlm
        scores = run_llr_mlm(model, tokenizer, dataset, genome, window_size=512)

    Args:
        model: The PyTorch model (.net attribute) from a loaded MLMLitModule.
            Must have a get_logits(input_ids) method that returns logits with
            shape [batch, seq_len, vocab_size].

    Note:
        This adapter requires the glm-experiments package to be installed.
        Install dependencies with: uv pip install -e .[glm-experiments]
        Then install glm-experiments: uv pip install git+https://github.com/Open-Athena/glm-experiments
    """

    def __init__(self, model: Any) -> None:
        # Check if user accidentally passed Lightning module instead of .net model
        if hasattr(model, "net"):
            raise TypeError(
                "GLMExperimentsMaskedLM expects the PyTorch model, not the Lightning module. "
                "Use: GLMExperimentsMaskedLM(lit_module.net)"
            )

        # Validate model has required method
        if not hasattr(model, "get_logits"):
            raise AttributeError(
                f"Model {type(model).__name__} does not have 'get_logits' method. "
                "GLMExperimentsMaskedLM requires models with get_logits(input_ids) method."
            )

        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs with shape [batch, seq_len]

        Returns:
            Logits with shape [batch, seq_len, vocab_size]
        """
        return cast(Float[Tensor, "B L V"], self._model.get_logits(input_ids))


class GLMExperimentsCausalLM(CausalLM):
    """Adapter for glm-experiments CLM (Causal Language Model) PyTorch Lightning modules.

    This adapter wraps models from the glm-experiments repository that follow the
    CLMLitModule pattern. The adapter expects the extracted PyTorch model (the .net
    attribute of the Lightning module), not the Lightning module itself.

    Usage:
        from glm_experiments.models.lm_lit_module import CLMLitModule
        from biofoundation.model.adapters.glm_experiments import GLMExperimentsCausalLM
        from biofoundation.model.adapters.hf import HFTokenizer
        from transformers import AutoTokenizer

        # Load checkpoint using PyTorch Lightning
        lit_module = CLMLitModule.load_from_checkpoint("path/to/checkpoint.ckpt")

        # Extract and wrap the PyTorch model
        model = GLMExperimentsCausalLM(lit_module.net)

        # Load tokenizer separately
        tokenizer = HFTokenizer(AutoTokenizer.from_pretrained("songlab/tokenizer-dna-mlm"))

        # Use with biofoundation inference functions
        from biofoundation.inference import run_llr_clm
        scores = run_llr_clm(model, tokenizer, dataset, genome, window_size=512)

    Args:
        model: The PyTorch model (.net attribute) from a loaded CLMLitModule.
            Must have a get_logits(input_ids) method that returns logits with
            shape [batch, seq_len, vocab_size].

    Note:
        This adapter requires the glm-experiments package to be installed.
        Install dependencies with: uv pip install -e .[glm-experiments]
        Then install glm-experiments: uv pip install git+https://github.com/Open-Athena/glm-experiments
    """

    def __init__(self, model: Any) -> None:
        # Check if user accidentally passed Lightning module instead of .net model
        if hasattr(model, "net"):
            raise TypeError(
                "GLMExperimentsCausalLM expects the PyTorch model, not the Lightning module. "
                "Use: GLMExperimentsCausalLM(lit_module.net)"
            )

        # Validate model has required method
        if not hasattr(model, "get_logits"):
            raise AttributeError(
                f"Model {type(model).__name__} does not have 'get_logits' method. "
                "GLMExperimentsCausalLM requires models with get_logits(input_ids) method."
            )

        super().__init__()
        self._model = model

    def forward(self, input_ids: Int[Tensor, "B L"]) -> Float[Tensor, "B L V"]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs with shape [batch, seq_len]

        Returns:
            Logits with shape [batch, seq_len, vocab_size]
        """
        return cast(Float[Tensor, "B L V"], self._model.get_logits(input_ids))
