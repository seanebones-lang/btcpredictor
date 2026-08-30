"""
Hydra/OmegaConf configuration loader for BTC Predictor.
"""

import os
from typing import Any, Dict, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig


@dataclass
class DataSourceConfig:
    """Data source configuration."""
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_rate_limit: int = 50
    binance_base_url: str = "https://api.binance.com/api/v3"
    binance_futures_url: str = "https://fapi.binance.com/fapi/v1"
    binance_rate_limit: int = 1200
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 60
    cache_ttl_historical: int = 3600


@dataclass
class ModelConfig:
    """Model configuration."""
    lstm_sequence_length: int = 96
    lstm_n_features: int = 12
    lstm_units: list = None
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.1
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_validation_split: float = 0.2
    lstm_early_stopping_patience: int = 10
    lstm_n_ensemble: int = 3
    
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    prophet_holidays_prior_scale: float = 10.0
    prophet_daily_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_yearly_seasonality: bool = False
    prophet_interval_width: float = 0.90
    
    ensemble_lstm_weight: float = 0.6
    ensemble_prophet_weight: float = 0.4
    ensemble_use_conformal: bool = True
    ensemble_conformal_alpha: float = 0.10
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    technical_indicators: list = None
    momentum_windows: list = None
    enriched_funding_rate: bool = True
    enriched_open_interest: bool = True
    enriched_liquidations: bool = True
    enriched_order_book_imbalance: bool = False
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = [
                "sma_20", "sma_50", "ema_12", "ema_26", "rsi_14",
                "macd", "bollinger_bands", "atr_14", "obv", "vwap"
            ]
        if self.momentum_windows is None:
            self.momentum_windows = [1, 4, 16, 96]


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit: str = "60/minute"
    cors_origins: list = None
    auth_enabled: bool = True
    api_key_env: str = "API_KEY"
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class WorkerConfig:
    """Worker configuration."""
    prediction_interval_minutes: int = 15
    retrain_interval_hours: int = 24
    telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    telegram_chat_id_env: str = "TELEGRAM_CHAT_ID"
    health_check_interval_minutes: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    data_sources: DataSourceConfig = None
    model: ModelConfig = None
    features: FeatureConfig = None
    api: APIConfig = None
    worker: WorkerConfig = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = DataSourceConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.worker is None:
            self.worker = WorkerConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
    
    Returns:
        AppConfig object with all settings.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "conf", "config.yaml"
        )
    
    # Load base config
    if os.path.exists(config_path):
        cfg: DictConfig = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({})
    
    # Override with environment variables
    env_overrides = {
        "data_sources.redis_url": os.getenv("REDIS_URL"),
        "api.api_key_env": os.getenv("API_KEY"),
        "worker.telegram_bot_token_env": os.getenv("TELEGRAM_BOT_TOKEN"),
        "worker.telegram_chat_id_env": os.getenv("TELEGRAM_CHAT_ID"),
    }
    
    for key, value in env_overrides.items():
        if value is not None:
            OmegaConf.update(cfg, key, value)
    
    # Convert to dataclass
    app_config = OmegaConf.to_object(cfg, structured_class=AppConfig)
    
    return app_config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get global config instance (singleton)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig):
    """Set global config instance."""
    global _config
    _config = config
