"""
Unit tests for BTC Predictor core modules.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Import modules to test
from btc_predictor.predictor_async import (
    EnrichedConfig,
    get_prediction_window,
    generate_async_prediction,
)
from btc_predictor.data_sources import (
    DataSourceConfig,
    AsyncDataFetcher,
)
from btc_predictor.config_loader import (
    DataSourceConfig as LoaderDataSourceConfig,
    ModelConfig,
    FeatureConfig,
    APIConfig,
    WorkerConfig,
    AppConfig,
    load_config,
    get_config,
)


class TestEnrichedConfig:
    """Test EnrichedConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = EnrichedConfig()
        
        assert config.sequence_length == 96
        assert config.lstm_units == [128, 64, 32]
        assert config.dropout == 0.2
        assert config.prophet_changepoint_prior_scale == 0.05
        assert config.ensemble_lstm_weight == 0.6
        assert config.ensemble_prophet_weight == 0.4
        assert config.use_conformal is True
        assert config.conformal_alpha == 0.10


class TestGetPredictionWindow:
    """Test prediction window calculation."""
    
    def test_window_calculation(self):
        """Test 15-minute window boundaries."""
        # Mock current time
        with patch('btc_predictor.predictor_async.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 30, 12, 7, 30)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            start, end = get_prediction_window()
            
            # Should round to next 15-min boundary
            assert start == "12:15"
            assert end == "12:30"


class TestDataSourceConfig:
    """Test DataSourceConfig dataclass."""
    
    def test_default_values(self):
        """Test default data source configuration."""
        config = DataSourceConfig()
        
        assert "coingecko.com" in config.COINGECKO_BASE_URL
        assert "binance.com" in config.BINANCE_BASE_URL
        assert config.REDIS_URL == "redis://localhost:6379/0"
        assert config.CACHE_TTL_SECONDS == 60
        assert config.CACHE_TTL_HISTORICAL == 3600


class TestAsyncDataFetcher:
    """Test AsyncDataFetcher class."""
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation."""
        fetcher = AsyncDataFetcher()
        key = fetcher._cache_key("price", "coingecko", "btc")
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        fetcher = AsyncDataFetcher()
        await fetcher.__aenter__()
        assert fetcher._session is not None
        await fetcher.__aexit__(None, None, None)


class TestConfigLoader:
    """Test configuration loader with Hydra/OmegaConf."""
    
    def test_dataclass_defaults(self):
        """Test all config dataclasses have sensible defaults."""
        ds_config = LoaderDataSourceConfig()
        assert ds_config.coingecko_base_url == "https://api.coingecko.com/api/v3"
        
        model_config = ModelConfig()
        assert model_config.lstm_sequence_length == 96
        assert model_config.lstm_units == [128, 64, 32]
        
        feature_config = FeatureConfig()
        assert "rsi_14" in feature_config.technical_indicators
        assert feature_config.momentum_windows == [1, 4, 16, 96]
        
        api_config = APIConfig()
        assert api_config.host == "0.0.0.0"
        assert api_config.port == 8000
        
        worker_config = WorkerConfig()
        assert worker_config.prediction_interval_minutes == 15
    
    def test_app_config_composition(self):
        """Test AppConfig composes all sub-configs."""
        app_config = AppConfig()
        
        assert isinstance(app_config.data_sources, LoaderDataSourceConfig)
        assert isinstance(app_config.model, ModelConfig)
        assert isinstance(app_config.features, FeatureConfig)
        assert isinstance(app_config.api, APIConfig)
        assert isinstance(app_config.worker, WorkerConfig)
    
    def test_load_config_from_file(self):
        """Test loading config from YAML file."""
        # This tests the file exists and is valid YAML
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.model.lstm_sequence_length == 96
    
    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestPredictionOutput:
    """Test prediction output format."""
    
    @pytest.mark.asyncio
    async def test_prediction_structure(self):
        """Test prediction returns expected structure."""
        # This is a basic smoke test
        with patch('btc_predictor.predictor_async.get_fetcher') as mock_fetcher:
            mock_fetcher_instance = AsyncMock()
            mock_fetcher_instance.__aenter__ = AsyncMock(return_value=mock_fetcher_instance)
            mock_fetcher_instance.__aexit__ = AsyncMock()
            mock_fetcher.return_value = mock_fetcher_instance
            
            with patch('btc_predictor.predictor_async.get_prediction_with_intervals') as mock_pred:
                mock_pred.return_value = {
                    "window_start": "12:00",
                    "window_end": "12:15",
                    "current_price": 50000.0,
                    "next_price": 50100.0,
                    "change_pct": 0.2,
                    "direction": "📈 UP",
                    "confidence": 85,
                    "ensemble_interval": [50050.0, 50150.0],
                    "prophet_interval": [50000.0, 50200.0],
                    "lstm_std": 25.0,
                    "enriched_features": {
                        "funding_rate": 0.0001,
                        "open_interest": 1000000,
                        "liquidations_count": 5
                    },
                    "model": "Test Ensemble"
                }
                
                result = await generate_async_prediction()
                
                assert "BTC 15-min Prediction" in result
                assert "$50,000.00" in result
                assert "📈 UP 0.2%" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
