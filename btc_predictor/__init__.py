"""
Modern BTC Predictor - Production-grade Bitcoin price prediction
"""

from .predictor import (
    generate_production_prediction,
    get_live_price,
    get_real_15m_data,
    add_indicators,
    build_deep_lstm,
    build_prophet_model,
    get_prediction_window,
    calculate_confidence,
    Config,
)

__version__ = "2.0.0"
__author__ = "Sean McDonnell (NextEleven)"
__all__ = [
    "generate_production_prediction",
    "get_live_price",
    "get_real_15m_data",
    "add_indicators",
    "build_deep_lstm",
    "build_prophet_model",
    "get_prediction_window",
    "calculate_confidence",
    "Config",
]
