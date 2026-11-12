"""PMAE: proportionally masked autoencoders for tabular data imputation (AAAI 2025)."""
from .imputer import PMAE
from .masking import mask_rate

__version__ = "0.1.0"
__all__ = ["PMAE", "mask_rate"]
