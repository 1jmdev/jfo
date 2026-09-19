"""JFO: optimized Jacobi forcing conversion.

Converts a pretrained autoregressive model into a causal parallel decoder in
four offline phases: trajectory collection, noise-schedule packing, SVD-aligned
subspace training, and merge with decoding validation.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
