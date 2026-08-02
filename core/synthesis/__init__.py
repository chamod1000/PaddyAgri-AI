"""
Synthesis Package Initialization - Version 3.0 Architecture
"""

from core.synthesis.ui_contracts import WeatherCardContract, FertilizerTableContract, MarketGridContract
from core.synthesis.text_sanitizer import coerce_scalar, summarize_snippet
from core.synthesis.hybrid_response_builder import HybridResponseBuilder

__all__ = [
    "WeatherCardContract",
    "FertilizerTableContract",
    "MarketGridContract",
    "HybridResponseBuilder",
    "coerce_scalar",
    "summarize_snippet"
]
