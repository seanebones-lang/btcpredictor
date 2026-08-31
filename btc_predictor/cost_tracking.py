"""
Cost tracking for BTC Predictor - API calls, compute costs, prediction costs.
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import threading

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CostCategory(Enum):
    API_CALL = "api_call"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_STORAGE = "data_storage"
    COMPUTE = "compute"


@dataclass
class CostEntry:
    """Single cost entry."""
    timestamp: datetime
    category: CostCategory
    service: str
    operation: str
    cost_usd: float
    quantity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "service": self.service,
            "operation": self.operation,
            "cost_usd": self.cost_usd,
            "quantity": self.quantity,
            "metadata": self.metadata,
        }


@dataclass
class CostSummary:
    """Aggregated cost summary."""
    period_start: datetime
    period_end: datetime
    total_cost_usd: float
    by_category: Dict[str, float]
    by_service: Dict[str, float]
    by_operation: Dict[str, float]
    cost_per_prediction: float
    predictions_count: int


class CostTracker:
    """Tracks costs for API calls, compute, and predictions."""
    
    # Default pricing (update with actual rates)
    PRICING = {
        # API costs (per call)
        "coingecko": {"price": 0.0, "historical": 0.0},  # Free tier
        "binance": {"price": 0.0, "funding_rate": 0.0, "open_interest": 0.0},  # Free
        "fred": {"series": 0.0},  # Free with API key
        "alphavantage": {"call": 0.0},  # Free tier
        "yfinance": {"call": 0.0},  # Free
        
        # Compute costs (per hour)
        "cpu_hour": 0.05,  # $0.05/hour for CPU
        "gpu_hour": 0.50,  # $0.50/hour for GPU (e.g., T4)
        "memory_gb_hour": 0.01,  # $0.01/GB/hour
        
        # Storage costs (per GB/month)
        "storage_gb_month": 0.10,
        
        # Redis (if managed)
        "redis_gb_month": 0.20,
    }
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._local_buffer: List[CostEntry] = []
        self._lock = threading.Lock()
        self._session_costs: Dict[str, float] = defaultdict(float)
    
    async def __aenter__(self):
        if REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                self._redis = None
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._redis:
            await self._redis.close()
    
    def _record_local(self, entry: CostEntry):
        """Record to local buffer."""
        with self._lock:
            self._local_buffer.append(entry)
            # Flush if buffer gets large
            if len(self._local_buffer) >= 100:
                self._flush_local()
    
    async def _record_redis(self, entry: CostEntry):
        """Record to Redis."""
        if not self._redis:
            return
        
        try:
            key = f"costs:{entry.timestamp.strftime('%Y-%m-%d')}"
            await self._redis.lpush(key, json.dumps(entry.to_dict()))
            await self._redis.expire(key, 86400 * 30)  # 30 days
        except Exception:
            pass
    
    def record_api_call(self, service: str, operation: str, quantity: int = 1):
        """Record an API call cost."""
        cost_per_call = self.PRICING.get(service, {}).get(operation, 0.0)
        total_cost = cost_per_call * quantity
        
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            category=CostCategory.API_CALL,
            service=service,
            operation=operation,
            cost_usd=total_cost,
            quantity=quantity,
        )
        
        self._record_local(entry)
        if self._redis:
            asyncio.create_task(self._record_redis(entry))
    
    def record_training(self, model_type: str, duration_seconds: float, gpu_used: bool = False):
        """Record model training cost."""
        hours = duration_seconds / 3600
        rate = self.PRICING["gpu_hour"] if gpu_used else self.PRICING["cpu_hour"]
        total_cost = hours * rate
        
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            category=CostCategory.MODEL_TRAINING,
            service="compute",
            operation=f"train_{model_type}",
            cost_usd=total_cost,
            quantity=int(hours * 100),  # centi-hours
            metadata={"duration_hours": hours, "gpu": gpu_used},
        )
        
        self._record_local(entry)
        if self._redis:
            asyncio.create_task(self._record_redis(entry))
    
    def record_inference(self, model_type: str, duration_ms: float, batch_size: int = 1):
        """Record model inference cost."""
        # Estimate compute cost for inference
        hours = duration_ms / (1000 * 3600)
        rate = self.PRICING["cpu_hour"]  # Inference typically on CPU
        total_cost = hours * rate * batch_size
        
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            category=CostCategory.MODEL_INFERENCE,
            service="compute",
            operation=f"infer_{model_type}",
            cost_usd=total_cost,
            quantity=batch_size,
            metadata={"duration_ms": duration_ms},
        )
        
        self._record_local(entry)
        if self._redis:
            asyncio.create_task(self._record_redis(entry))
    
    def record_prediction_cost(self, prediction_id: str, total_cost_usd: float):
        """Record total cost for a single prediction."""
        entry = CostEntry(
            timestamp=datetime.utcnow(),
            category=CostCategory.MODEL_INFERENCE,
            service="prediction",
            operation="full_pipeline",
            cost_usd=total_cost_usd,
            quantity=1,
            metadata={"prediction_id": prediction_id},
        )
        
        self._record_local(entry)
        if self._redis:
            asyncio.create_task(self._record_redis(entry))
        
        # Track session cost
        self._session_costs["total"] += total_cost_usd
        self._session_costs["predictions"] += 1
    
    def _flush_local(self):
        """Flush local buffer to file."""
        if not self._local_buffer:
            return
        
        try:
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
            filename = f"costs_{date_str}.jsonl"
            
            with open(filename, "a") as f:
                for entry in self._local_buffer:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            
            self._local_buffer.clear()
        except Exception:
            pass
    
    async def get_cost_summary(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> CostSummary:
        """Get cost summary for a period."""
        # Collect from local buffer
        all_entries = list(self._local_buffer)
        
        # Collect from files
        current = start_date
        while current <= end_date:
            filename = f"costs_{current.strftime('%Y-%m-%d')}.jsonl"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            entry = CostEntry(
                                timestamp=datetime.fromisoformat(data["timestamp"]),
                                category=CostCategory(data["category"]),
                                service=data["service"],
                                operation=data["operation"],
                                cost_usd=data["cost_usd"],
                                quantity=data["quantity"],
                                metadata=data.get("metadata", {}),
                            )
                            all_entries.append(entry)
                        except Exception:
                            pass
            current += timedelta(days=1)
        
        # Collect from Redis
        if self._redis:
            current = start_date
            while current <= end_date:
                key = f"costs:{current.strftime('%Y-%m-%d')}"
                try:
                    entries = await self._redis.lrange(key, 0, -1)
                    for entry_json in entries:
                        data = json.loads(entry_json)
                        entry = CostEntry(
                            timestamp=datetime.fromisoformat(data["timestamp"]),
                            category=CostCategory(data["category"]),
                            service=data["service"],
                            operation=data["operation"],
                            cost_usd=data["cost_usd"],
                            quantity=data["quantity"],
                            metadata=data.get("metadata", {}),
                        )
                        all_entries.append(entry)
                except Exception:
                    pass
                current += timedelta(days=1)
        
        # Filter by date range
        filtered = [
            e for e in all_entries
            if start_date <= e.timestamp <= end_date
        ]
        
        # Aggregate
        total = sum(e.cost_usd for e in filtered)
        
        by_category = defaultdict(float)
        by_service = defaultdict(float)
        by_operation = defaultdict(float)
        
        for e in filtered:
            by_category[e.category.value] += e.cost_usd
            by_service[e.service] += e.cost_usd
            by_operation[e.operation] += e.cost_usd
        
        predictions = self._session_costs.get("predictions", 0)
        cost_per_pred = total / predictions if predictions > 0 else 0
        
        return CostSummary(
            period_start=start_date,
            period_end=end_date,
            total_cost_usd=total,
            by_category=dict(by_category),
            by_service=dict(by_service),
            by_operation=dict(by_operation),
            cost_per_prediction=cost_per_pred,
            predictions_count=predictions,
        )
    
    def get_session_cost(self) -> Dict[str, Any]:
        """Get current session costs."""
        return {
            "total_usd": self._session_costs.get("total", 0),
            "predictions": self._session_costs.get("predictions", 0),
            "cost_per_prediction": (
                self._session_costs.get("total", 0) / self._session_costs.get("predictions", 1)
                if self._session_costs.get("predictions", 0) > 0 else 0
            ),
        }
    
    async def export_costs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> str:
        """Export costs for a period."""
        summary = await self.get_cost_summary(start_date, end_date)
        
        if format == "json":
            return json.dumps(asdict(summary), indent=2, default=str)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Category", "Service", "Operation", "Cost USD"])
            for cat, cost in summary.by_category.items():
                writer.writerow([cat, "", "", cost])
            return output.getvalue()
        else:
            return str(summary)


# Decorator for automatic cost tracking
def track_cost(category: CostCategory, service: str, operation: str):
    """Decorator to automatically track function costs."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracker = get_cost_tracker()
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                tracker.record_inference(operation, duration_ms)
        return wrapper
    return decorator


# Global cost tracker
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create cost tracker singleton."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


async def demo_cost_tracking():
    """Demo cost tracking."""
    async with CostTracker() as tracker:
        # Simulate some API calls
        tracker.record_api_call("coingecko", "price", 10)
        tracker.record_api_call("binance", "funding_rate", 5)
        
        # Simulate training
        tracker.record_training("lstm", 300, gpu_used=True)  # 5 min on GPU
        tracker.record_training("prophet", 60, gpu_used=False)  # 1 min on CPU
        
        # Simulate predictions
        for i in range(10):
            tracker.record_inference("ensemble", 150, batch_size=1)  # 150ms per prediction
            tracker.record_prediction_cost(f"pred_{i}", 0.0001)
        
        # Get summary
        summary = await tracker.get_cost_summary(
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow(),
        )
        
        print("Cost Summary:")
        print(f"  Total: ${summary.total_cost_usd:.6f}")
        print(f"  Per prediction: ${summary.cost_per_prediction:.6f}")
        print(f"  Predictions: {summary.predictions_count}")
        print(f"  By category: {summary.by_category}")
        print(f"  By service: {summary.by_service}")


if __name__ == "__main__":
    asyncio.run(demo_cost_tracking())
