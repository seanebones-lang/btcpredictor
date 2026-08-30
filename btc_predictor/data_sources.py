"""
Data sources module with async fetching, multiple providers, and Redis caching.
"""

import asyncio
import time
import json
import hashlib
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiohttp
import redis.asyncio as redis
import pandas as pd
import numpy as np

from .predictor import Config


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    # CoinGecko (primary)
    COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
    COINGECKO_RATE_LIMIT: int = 50  # requests per minute
    
    # Binance (secondary - public API)
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
    BINANCE_FUTURES_URL: str = "https://fapi.binance.com/fapi/v1"
    BINANCE_RATE_LIMIT: int = 1200  # requests per minute
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 60  # 1 minute cache for prices
    CACHE_TTL_HISTORICAL: int = 3600  # 1 hour for historical data


class AsyncDataFetcher:
    """Async data fetcher with multiple providers and Redis caching."""
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        self.config = config or DataSourceConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis: Optional[redis.Redis] = None
        self._coingecko_semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self._binance_semaphore = asyncio.Semaphore(20)
        
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            connector=aiohttp.TCPConnector(limit=50)
        )
        try:
            self._redis = redis.from_url(self.config.REDIS_URL, decode_responses=True)
            await self._redis.ping()
        except Exception:
            self._redis = None
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
        if self._redis:
            await self._redis.close()
    
    def _cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments."""
        key_str = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._redis:
            return None
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None
    
    async def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Set value in Redis cache."""
        if not self._redis:
            return
        try:
            await self._redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass
    
    async def _fetch_with_retry(
        self, 
        url: str, 
        params: Optional[Dict] = None, 
        semaphore: Optional[asyncio.Semaphore] = None,
        retries: int = 3
    ) -> Optional[Dict]:
        """Fetch with retry logic and rate limiting."""
        sem = semaphore or self._coingecko_semaphore
        async with sem:
            for attempt in range(retries):
                try:
                    async with self._session.get(url, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return None
                except Exception:
                    await asyncio.sleep(1)
            return None

    async def get_json(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Generic JSON fetch for external APIs."""
        return await self._fetch_with_retry(url, params=params)
    
    # ============================================================
    # CoinGecko API (Primary)
    # ============================================================
    
    async def get_live_price(self) -> Optional[float]:
        """Get current BTC price from CoinGecko with caching."""
        cache_key = self._cache_key("price", "coingecko", "btc")
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return float(cached)
        
        data = await self._fetch_with_retry(
            f"{Config.COINGECKO_BASE_URL}/simple/price",
            {"ids": "bitcoin", "vs_currencies": "usd"}
        )
        if data and "bitcoin" in data:
            price = float(data["bitcoin"]["usd"])
            await self._set_cached(cache_key, price, self.config.CACHE_TTL_SECONDS)
            return price
        return None
    
    async def get_historical_15m(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch 15-minute historical data from CoinGecko."""
        cache_key = self._cache_key("historical", "coingecko", "btc", "15m", days)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            df["ds"] = pd.to_datetime(df["ds"])
            return df
        
        data = await self._fetch_with_retry(
            f"{Config.COINGECKO_BASE_URL}/coins/bitcoin/market_chart",
            {"vs_currency": "usd", "days": days, "interval": "hourly"}
        )
        if not data or "prices" not in data:
            return None
        
        prices = []
        volumes = []
        for i, point in enumerate(data["prices"]):
            prices.append({
                "ds": datetime.fromtimestamp(point[0] / 1000),
                "y": point[1]
            })
            if i < len(data.get("total_volumes", [])):
                volumes.append(data["total_volumes"][i][1])
        
        df = pd.DataFrame(prices)
        df["volume"] = volumes[:len(df)]
        df = df.set_index("ds").resample("15min").interpolate(method="linear").reset_index()
        
        await self._set_cached(cache_key, df.to_dict("records"), self.config.CACHE_TTL_HISTORICAL)
        return df
    
    # ============================================================
    # Binance API (Secondary - Futures data for funding rates, OI, etc.)
    # ============================================================
    
    async def get_binance_funding_rate(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get current funding rate from Binance Futures."""
        cache_key = self._cache_key("funding", "binance", symbol)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return float(cached)
        
        data = await self._fetch_with_retry(
            f"{self.config.BINANCE_FUTURES_URL}/premiumIndex",
            {"symbol": symbol},
            self._binance_semaphore
        )
        if data and "lastFundingRate" in data:
            rate = float(data["lastFundingRate"])
            await self._set_cached(cache_key, rate, 300)  # 5 min cache
            return rate
        return None
    
    async def get_binance_open_interest(self, symbol: str = "BTCUSDT") -> Optional[float]:
        """Get open interest from Binance Futures."""
        cache_key = self._cache_key("oi", "binance", symbol)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            return float(cached)
        
        data = await self._fetch_with_retry(
            f"{self.config.BINANCE_FUTURES_URL}/openInterest",
            {"symbol": symbol},
            self._binance_semaphore
        )
        if data and "openInterest" in data:
            oi = float(data["openInterest"])
            await self._set_cached(cache_key, oi, 60)
            return oi
        return None
    
    async def get_binance_liquidations(self, symbol: str = "BTCUSDT", limit: int = 100) -> Optional[List[Dict]]:
        """Get recent liquidations from Binance Futures."""
        data = await self._fetch_with_retry(
            f"{self.config.BINANCE_FUTURES_URL}/forceOrders",
            {"symbol": symbol, "limit": limit},
            self._binance_semaphore
        )
        return data if data else None
    
    async def get_binance_klines(
        self, 
        symbol: str = "BTCUSDT", 
        interval: str = "15m", 
        limit: int = 500
    ) -> Optional[pd.DataFrame]:
        """Get klines/candlestick data from Binance."""
        cache_key = self._cache_key("klines", "binance", symbol, interval, limit)
        cached = await self._get_cached(cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            df["ds"] = pd.to_datetime(df["ds"])
            return df
        
        data = await self._fetch_with_retry(
            f"{self.config.BINANCE_BASE_URL}/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
            self._binance_semaphore
        )
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["ds"] = pd.to_datetime(df["open_time"], unit="ms")
        df["y"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df[["ds", "y", "volume"]].copy()
        
        await self._set_cached(cache_key, df.to_dict("records"), 300)
        return df
    
    # ============================================================
    # Macro Data (FRED, etc. - placeholder for future)
    # ============================================================
    
    async def get_macro_data(self) -> Dict[str, Optional[float]]:
        """Get macro indicators (DXY, yields, VIX). Placeholder for now."""
        return {
            "dxy": None,
            "us10y": None,
            "vix": None,
            "btc_dominance": None,
        }
    
    # ============================================================
    # Unified Interface with Fallback
    # ============================================================
    
    async def get_best_price(self) -> Optional[float]:
        """Get price from primary, fallback to secondary."""
        price = await self.get_live_price()
        if price:
            return price
        
        # Fallback to Binance spot
        try:
            data = await self._fetch_with_retry(
                f"{self.config.BINANCE_BASE_URL}/ticker/price",
                {"symbol": "BTCUSDT"},
                self._binance_semaphore
            )
            if data and "price" in data:
                return float(data["price"])
        except Exception:
            pass
        return None
    
    async def get_best_historical(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data from primary, fallback to Binance."""
        df = await self.get_historical_15m(days)
        if df is not None and len(df) > 50:
            return df
        
        # Fallback to Binance klines
        return await self.get_binance_klines("BTCUSDT", "15m", min(days * 96, 1000))
    
    async def get_enriched_features(self) -> Dict[str, Any]:
        """Get all enriched features for model input."""
        results = await asyncio.gather(
            self.get_best_price(),
            self.get_binance_funding_rate(),
            self.get_binance_open_interest(),
            self.get_binance_liquidations(),
            self.get_macro_data(),
            return_exceptions=True
        )
        
        price, funding, oi, liquidations, macro = results
        
        return {
            "price": price if not isinstance(price, Exception) else None,
            "funding_rate": funding if not isinstance(funding, Exception) else None,
            "open_interest": oi if not isinstance(oi, Exception) else None,
            "liquidations": liquidations if not isinstance(liquidations, Exception) else None,
            "macro": macro if not isinstance(macro, Exception) else {},
        }


# Singleton instance for reuse
_fetcher_instance: Optional[AsyncDataFetcher] = None


async def get_fetcher() -> AsyncDataFetcher:
    """Get or create the global fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = AsyncDataFetcher()
        await _fetcher_instance.__aenter__()
    return _fetcher_instance


async def close_fetcher() -> None:
    """Close the global fetcher instance."""
    global _fetcher_instance
    if _fetcher_instance:
        await _fetcher_instance.__aexit__(None, None, None)
        _fetcher_instance = None
