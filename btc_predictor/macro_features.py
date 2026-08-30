"""
Macro feature engineering: DXY, yields, VIX, correlation regime detection.
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class MacroConfig:
    """Configuration for macro features."""
    # FRED API (requires API key for production)
    fred_api_key: Optional[str] = None
    fred_base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    
    # Alpha Vantage (free tier available)
    alphavantage_api_key: Optional[str] = None
    alphavantage_base_url: str = "https://www.alphavantage.co/query"
    
    # Yahoo Finance (via yfinance)
    use_yfinance: bool = True
    
    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour for macro data
    
    # Series IDs for FRED
    series_ids: Dict[str, str] = None
    
    def __post_init__(self):
        if self.series_ids is None:
            self.series_ids = {
                "DXY": "DTWEXBGS",      # Dollar Index
                "US10Y": "DGS10",       # 10-Year Treasury
                "US2Y": "DGS2",         # 2-Year Treasury
                "VIX": "VIXCLS",        # VIX
                "SP500": "SP500",       # S&P 500
                "GOLD": "GOLDPMGBD228NLBM",  # Gold Price
                "OIL": "DCOILWTICO",    # WTI Crude Oil
            }


class MacroFeatureEngine:
    """Fetch and compute macro features for BTC prediction."""
    
    def __init__(self, config: MacroConfig = None):
        self.config = config or MacroConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (timestamp, value)
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    def _is_cached(self, key: str) -> bool:
        """Check if key is cached and not expired."""
        if key not in self._cache:
            return False
        timestamp, _ = self._cache[key]
        return (datetime.now().timestamp() - timestamp) < self.config.cache_ttl_seconds
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if self._is_cached(key):
            return self._cache[key][1]
        return None
    
    def _set_cached(self, key: str, value: Any):
        """Set cached value."""
        self._cache[key] = (datetime.now().timestamp(), value)
    
    async def _fetch_fred(self, series_id: str) -> Optional[pd.Series]:
        """Fetch data from FRED API."""
        if not self.config.fred_api_key:
            return None
        
        cache_key = f"fred_{series_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        params = {
            "series_id": series_id,
            "api_key": self.config.fred_api_key,
            "file_type": "json",
            "observation_start": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        }
        
        try:
            async with self._session.get(self.config.fred_base_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    observations = data.get("observations", [])
                    
                    df = pd.DataFrame(observations)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna().set_index("date")["value"]
                    
                    self._set_cached(cache_key, df)
                    return df
        except Exception as e:
            print(f"FRED fetch error for {series_id}: {e}")
        
        return None
    
    async def _fetch_yfinance(self, symbol: str, period: str = "1y") -> Optional[pd.Series]:
        """Fetch data from Yahoo Finance."""
        cache_key = f"yf_{symbol}_{period}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            if not hist.empty:
                series = hist["Close"]
                self._set_cached(cache_key, series)
                return series
        except Exception as e:
            print(f"yfinance fetch error for {symbol}: {e}")
        
        return None
    
    async def _fetch_alphavantage(self, function: str, symbol: str) -> Optional[pd.Series]:
        """Fetch data from Alpha Vantage."""
        if not self.config.alphavantage_api_key:
            return None
        
        cache_key = f"av_{function}_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.config.alphavantage_api_key,
            "outputsize": "compact",
        }
        
        try:
            async with self._session.get(self.config.alphavantage_base_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Parse based on function
                    key = [k for k in data.keys() if "Time Series" in k]
                    if key:
                        ts_data = data[key[0]]
                        df = pd.DataFrame.from_dict(ts_data, orient="index")
                        df.index = pd.to_datetime(df.index)
                        df.columns = [c.split(". ")[1] for c in df.columns]
                        series = df["close"].astype(float).sort_index()
                        self._set_cached(cache_key, series)
                        return series
        except Exception as e:
            print(f"Alpha Vantage fetch error: {e}")
        
        return None
    
    async def get_dxy(self) -> Optional[float]:
        """Get current Dollar Index (DXY) value."""
        # Try FRED first
        series = await self._fetch_fred(self.config.series_ids["DXY"])
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        # Fallback to Yahoo Finance (DX-Y.NYB)
        series = await self._fetch_yfinance("DX-Y.NYB", "1mo")
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        return None
    
    async def get_treasury_yields(self) -> Dict[str, Optional[float]]:
        """Get current US Treasury yields."""
        yields = {}
        
        # 10-Year
        series = await self._fetch_fred(self.config.series_ids["US10Y"])
        yields["US10Y"] = float(series.iloc[-1]) if series is not None and len(series) > 0 else None
        
        # 2-Year
        series = await self._fetch_fred(self.config.series_ids["US2Y"])
        yields["US2Y"] = float(series.iloc[-1]) if series is not None and len(series) > 0 else None
        
        # Calculate spread
        if yields["US10Y"] is not None and yields["US2Y"] is not None:
            yields["spread_10y_2y"] = yields["US10Y"] - yields["US2Y"]
        else:
            yields["spread_10y_2y"] = None
        
        return yields
    
    async def get_vix(self) -> Optional[float]:
        """Get current VIX value."""
        series = await self._fetch_fred(self.config.series_ids["VIX"])
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        # Fallback to Yahoo Finance
        series = await self._fetch_yfinance("^VIX", "1mo")
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        return None
    
    async def get_sp500(self) -> Optional[float]:
        """Get current S&P 500 value."""
        series = await self._fetch_fred(self.config.series_ids["SP500"])
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        # Fallback to Yahoo Finance
        series = await self._fetch_yfinance("^GSPC", "1mo")
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        return None
    
    async def get_gold(self) -> Optional[float]:
        """Get current Gold price."""
        series = await self._fetch_fred(self.config.series_ids["GOLD"])
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        # Fallback to Yahoo Finance
        series = await self._fetch_yfinance("GC=F", "1mo")
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        return None
    
    async def get_oil(self) -> Optional[float]:
        """Get current WTI Crude Oil price."""
        series = await self._fetch_fred(self.config.series_ids["OIL"])
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        # Fallback to Yahoo Finance
        series = await self._fetch_yfinance("CL=F", "1mo")
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        
        return None
    
    async def get_all_macro_features(self) -> Dict[str, Any]:
        """Get all macro features at once."""
        # Fetch concurrently
        dxy_task = self.get_dxy()
        yields_task = self.get_treasury_yields()
        vix_task = self.get_vix()
        sp500_task = self.get_sp500()
        gold_task = self.get_gold()
        oil_task = self.get_oil()
        
        dxy, yields, vix, sp500, gold, oil = await asyncio.gather(
            dxy_task, yields_task, vix_task, sp500_task, gold_task, oil_task
        )
        
        features = {
            "dxy": dxy,
            "vix": vix,
            "sp500": sp500,
            "gold": gold,
            "oil": oil,
            **yields,
        }
        
        return features
    
    async def compute_regime_features(
        self,
        btc_returns: pd.Series,
        lookback_days: int = 30,
    ) -> Dict[str, float]:
        """Compute correlation regime features."""
        features = {}
        
        try:
            # Get macro data for correlation
            macro_data = await self.get_all_macro_features()
            
            # For regime detection, we need historical macro data
            # This is a simplified version - in production, fetch full histories
            
            # DXY regime
            dxy = macro_data.get("dxy")
            if dxy is not None:
                features["dxy_level"] = dxy
                # DXY trend (simplified)
                features["dxy_above_100"] = 1.0 if dxy > 100 else 0.0
                features["dxy_above_105"] = 1.0 if dxy > 105 else 0.0
            
            # VIX regime
            vix = macro_data.get("vix")
            if vix is not None:
                features["vix_level"] = vix
                features["vix_high"] = 1.0 if vix > 30 else 0.0
                features["vix_low"] = 1.0 if vix < 15 else 0.0
            
            # Yield curve regime
            spread = macro_data.get("spread_10y_2y")
            if spread is not None:
                features["yield_spread"] = spread
                features["yield_curve_inverted"] = 1.0 if spread < 0 else 0.0
                features["yield_curve_steep"] = 1.0 if spread > 1.5 else 0.0
            
            # Risk-on/risk-off
            sp500 = macro_data.get("sp500")
            gold = macro_data.get("gold")
            if sp500 is not None and gold is not None:
                # Simple risk-on proxy: SP500 up, Gold down
                features["risk_on_proxy"] = 1.0  # Would need historical for real calc
            
            # Correlation regime (BTC vs macro)
            # This requires historical macro data - placeholder
            features["btc_sp500_corr"] = 0.0
            features["btc_dxy_corr"] = 0.0
            features["btc_gold_corr"] = 0.0
            
        except Exception as e:
            print(f"Regime feature computation error: {e}")
        
        return features


class CorrelationRegimeDetector:
    """Detect correlation regimes between BTC and macro assets."""
    
    def __init__(self, window: int = 60):
        self.window = window
        self.regimes = {
            "risk_on": "Risk-on: BTC correlates with equities",
            "risk_off": "Risk-off: BTC decouples from equities",
            "flight_to_safety": "Flight to safety: BTC correlates with gold",
            "dollar_driven": "Dollar-driven: BTC anti-correlates with DXY",
            "neutral": "Neutral: No clear regime",
        }
    
    def detect_regime(
        self,
        btc_returns: pd.Series,
        macro_returns: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        """Detect current correlation regime."""
        if len(btc_returns) < self.window:
            return {"regime": "insufficient_data", "confidence": 0.0}
        
        recent_btc = btc_returns.tail(self.window)
        correlations = {}
        
        for name, series in macro_returns.items():
            if len(series) >= self.window:
                recent_macro = series.tail(self.window)
                # Align indices
                aligned_btc, aligned_macro = recent_btc.align(recent_macro, join="inner")
                if len(aligned_btc) >= 20:
                    corr = aligned_btc.corr(aligned_macro)
                    correlations[name] = corr
        
        # Determine regime
        regime = "neutral"
        confidence = 0.5
        
        sp500_corr = correlations.get("sp500", 0)
        dxy_corr = correlations.get("dxy", 0)
        gold_corr = correlations.get("gold", 0)
        
        if sp500_corr > 0.5 and dxy_corr < -0.3:
            regime = "risk_on"
            confidence = min(sp500_corr, abs(dxy_corr))
        elif sp500_corr < -0.3 and gold_corr > 0.3:
            regime = "flight_to_safety"
            confidence = min(abs(sp500_corr), gold_corr)
        elif dxy_corr < -0.5:
            regime = "dollar_driven"
            confidence = abs(dxy_corr)
        elif sp500_corr < -0.3:
            regime = "risk_off"
            confidence = abs(sp500_corr)
        
        return {
            "regime": regime,
            "description": self.regimes.get(regime, "Unknown regime"),
            "confidence": float(confidence),
            "correlations": correlations,
        }
    
    def get_regime_features(self, regime_result: Dict[str, Any]) -> Dict[str, float]:
        """Convert regime detection to features."""
        regime = regime_result.get("regime", "neutral")
        confidence = regime_result.get("confidence", 0.5)
        
        features = {
            "regime_risk_on": 1.0 if regime == "risk_on" else 0.0,
            "regime_risk_off": 1.0 if regime == "risk_off" else 0.0,
            "regime_flight_to_safety": 1.0 if regime == "flight_to_safety" else 0.0,
            "regime_dollar_driven": 1.0 if regime == "dollar_driven" else 0.0,
            "regime_neutral": 1.0 if regime == "neutral" else 0.0,
            "regime_confidence": confidence,
        }
        
        # Add correlation features
        for name, corr in regime_result.get("correlations", {}).items():
            features[f"corr_{name}"] = float(corr) if corr is not None else 0.0
        
        return features


# Integration function
async def enrich_with_macro(
    fetcher: 'AsyncDataFetcher',
    config: 'EnrichedConfig',
) -> Dict[str, Any]:
    """Enrich prediction with macro features."""
    macro_config = MacroConfig()
    
    async with MacroFeatureEngine(macro_config) as macro_engine:
        macro_features = await macro_engine.get_all_macro_features()
        
        # Also get regime features (would need historical data)
        regime_features = await macro_engine.compute_regime_features(
            pd.Series(dtype=float),  # Placeholder for BTC returns
        )
        
        return {
            **macro_features,
            **regime_features,
        }


if __name__ == "__main__":
    async def demo():
        config = MacroConfig()
        async with MacroFeatureEngine(config) as engine:
            features = await engine.get_all_macro_features()
            print("Macro Features:")
            for k, v in features.items():
                print(f"  {k}: {v}")
    
    asyncio.run(demo())
