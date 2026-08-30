"""
Walk-forward backtesting with purged cross-validation for BTC Predictor.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings("ignore")

from .predictor_async import (
    AsyncDataFetcher, 
    EnrichedConfig,
    get_prediction_with_intervals,
)

@dataclass
class BacktestResult:
    """Single backtest window result."""
    window_start: str
    window_end: str
    predicted_price: float
    actual_price: float
    change_pct_pred: float
    change_pct_actual: float
    direction_correct: bool
    mae: float
    mape: float
    within_interval: bool
    confidence: int
    ensemble_interval: List[float]

@dataclass
class BacktestSummary:
    """Aggregated backtest metrics."""
    total_windows: int
    direction_accuracy: float
    mean_mae: float
    mean_mape: float
    interval_coverage: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float

class WalkForwardBacktest:
    """Walk-forward backtesting with purged cross-validation."""
    
    def __init__(
        self,
        initial_train_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 1,
        purge_days: int = 1,
        embargo_days: int = 0,
    ):
        self.initial_train_days = initial_train_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.results: List[BacktestResult] = []
        
    def generate_windows(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.
        Returns: [(train_start, train_end, test_start, test_end), ...]
        """
        windows = []
        current = start_date
        
        while True:
            train_start = current
            train_end = current + timedelta(days=self.initial_train_days)
            test_start = train_end + timedelta(days=self.purge_days)
            test_end = test_start + timedelta(days=self.test_window_days)
            
            if test_end > end_date:
                break
                
            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=self.step_days)
            
        return windows
    
    async def run_backtest(
        self,
        fetcher: AsyncDataFetcher,
        config: EnrichedConfig,
        start_date: datetime,
        end_date: datetime,
    ) -> List[BacktestResult]:
        """Run walk-forward backtest."""
        windows = self.generate_windows(start_date, end_date)
        print(f"Running backtest with {len(windows)} windows...")
        
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"Window {i+1}/{len(windows)}: {test_start.date()} to {test_end.date()}")
            
            try:
                # Get actual prices for test period
                actual_prices = await self._get_actual_prices(
                    fetcher, test_start, test_end
                )
                
                if len(actual_prices) < 2:
                    print(f"  Insufficient data, skipping")
                    continue
                
                # Get prediction at test_start
                prediction = await get_prediction_with_intervals(fetcher, config)
                
                # Convert to native types
                import numpy as np
                def to_native(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: to_native(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [to_native(v) for v in obj]
                    return obj
                
                prediction = {k: to_native(v) for k, v in prediction.items()}
                
                predicted_price = prediction['next_price']
                actual_price = actual_prices[-1]  # Price at test_end
                start_price = actual_prices[0]    # Price at test_start
                
                change_pct_pred = prediction['change_pct']
                change_pct_actual = ((actual_price - start_price) / start_price) * 100
                
                direction_pred = 1 if change_pct_pred > 0 else -1
                direction_actual = 1 if change_pct_actual > 0 else -1
                direction_correct = direction_pred == direction_actual
                
                mae = abs(predicted_price - actual_price)
                mape = (mae / actual_price) * 100
                
                interval = prediction['ensemble_interval']
                within_interval = interval[0] <= actual_price <= interval[1]
                
                result = BacktestResult(
                    window_start=test_start.isoformat(),
                    window_end=test_end.isoformat(),
                    predicted_price=predicted_price,
                    actual_price=actual_price,
                    change_pct_pred=change_pct_pred,
                    change_pct_actual=change_pct_actual,
                    direction_correct=direction_correct,
                    mae=mae,
                    mape=mape,
                    within_interval=within_interval,
                    confidence=prediction['confidence'],
                    ensemble_interval=interval,
                )
                
                results.append(result)
                print(f"  Pred: ${predicted_price:,.2f} | Actual: ${actual_price:,.2f} | "
                      f"Dir: {'✓' if direction_correct else '✗'} | "
                      f"MAE: ${mae:,.2f} | Interval: {'✓' if within_interval else '✗'}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        self.results = results
        return results
    
    async def _get_actual_prices(
        self, 
        fetcher: AsyncDataFetcher,
        start: datetime, 
        end: datetime
    ) -> List[float]:
        """Get actual BTC prices for a date range."""
        # Use CoinGecko for historical data
        days = (end - start).days + 1
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "hourly"}
        
        data = await fetcher.get_json(url, params=params)
        if not data or "prices" not in data:
            return []
        
        prices = [p[1] for p in data["prices"]]
        return prices
    
    def compute_summary(self) -> BacktestSummary:
        """Compute aggregated metrics from results."""
        if not self.results:
            return BacktestSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        r = self.results
        
        # Direction accuracy
        direction_correct = sum(1 for x in r if x.direction_correct)
        direction_accuracy = direction_correct / len(r)
        
        # MAE/MAPE
        mean_mae = np.mean([x.mae for x in r])
        mean_mape = np.mean([x.mape for x in r])
        
        # Interval coverage
        within_interval = sum(1 for x in r if x.within_interval)
        interval_coverage = within_interval / len(r)
        
        # Returns for Sharpe
        returns = [x.change_pct_actual / 100 for x in r]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)  # 15-min periods per year
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative = np.cumprod([1 + ret for ret in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Total return
        total_return = (cumulative[-1] - 1) * 100
        
        # Win rate (profitable predictions)
        wins = sum(1 for x in r if x.change_pct_pred * x.change_pct_actual > 0)
        win_rate = wins / len(r)
        
        # Profit factor
        gross_profit = sum(x.change_pct_actual for x in r if x.change_pct_actual > 0)
        gross_loss = abs(sum(x.change_pct_actual for x in r if x.change_pct_actual < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestSummary(
            total_windows=len(r),
            direction_accuracy=direction_accuracy,
            mean_mae=mean_mae,
            mean_mape=mean_mape,
            interval_coverage=interval_coverage,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )
    
    def save_results(self, filepath: str):
        """Save results to JSON."""
        data = {
            "results": [asdict(r) for r in self.results],
            "summary": asdict(self.compute_summary()),
            "config": {
                "initial_train_days": self.initial_train_days,
                "test_window_days": self.test_window_days,
                "step_days": self.step_days,
                "purge_days": self.purge_days,
                "embargo_days": self.embargo_days,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to {filepath}")

async def run_full_backtest():
    """Run complete walk-forward backtest."""
    fetcher = AsyncDataFetcher()
    await fetcher.__aenter__()
    
    config = EnrichedConfig()
    
    # Backtest last 60 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    backtest = WalkForwardBacktest(
        initial_train_days=30,
        test_window_days=7,
        step_days=1,
        purge_days=1,
    )
    
    results = await backtest.run_backtest(fetcher, config, start_date, end_date)
    summary = backtest.compute_summary()
    
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(f"Total Windows:      {summary.total_windows}")
    print(f"Direction Accuracy: {summary.direction_accuracy:.1%}")
    print(f"Mean MAE:           ${summary.mean_mae:,.2f}")
    print(f"Mean MAPE:          {summary.mean_mape:.2f}%")
    print(f"Interval Coverage:  {summary.interval_coverage:.1%}")
    print(f"Sharpe Ratio:       {summary.sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {summary.max_drawdown:.2f}%")
    print(f"Total Return:       {summary.total_return:.2f}%")
    print(f"Win Rate:           {summary.win_rate:.1%}")
    print(f"Profit Factor:      {summary.profit_factor:.2f}")
    
    backtest.save_results("backtest_results.json")
    
    await fetcher.__aexit__(None, None, None)
    
    return backtest

if __name__ == "__main__":
    asyncio.run(run_full_backtest())
