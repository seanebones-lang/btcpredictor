"""
A/B testing framework for BTC Predictor model versions.
"""

import asyncio
import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

from .mlflow_tracking import BTCModelTracker, get_tracker


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ExperimentConfig:
    """Configuration for A/B experiment."""
    name: str
    description: str
    model_a_run_id: str  # Champion/control
    model_b_run_id: str  # Challenger/treatment
    traffic_split: float = 0.5  # Fraction to model B
    min_samples: int = 100
    max_duration_days: int = 7
    success_metric: str = "direction_accuracy"
    minimum_detectable_effect: float = 0.02  # 2%
    confidence_level: float = 0.95
    random_seed: int = 42


@dataclass
class ExperimentResult:
    """Results of an A/B experiment."""
    experiment_id: str
    name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    model_a: Dict[str, Any]
    model_b: Dict[str, Any]
    winner: Optional[str] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    lift: Optional[float] = None
    statistical_significance: bool = False


class ABTestManager:
    """Manages A/B testing experiments for model versions."""
    
    def __init__(self):
        self.tracker = get_tracker()
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.assignments: Dict[str, str] = {}  # user_id -> variant
        self.metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B experiment."""
        experiment_id = hashlib.md5(
            f"{config.name}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        self.experiments[experiment_id] = config
        self.results[experiment_id] = ExperimentResult(
            experiment_id=experiment_id,
            name=config.name,
            status=ExperimentStatus.DRAFT,
            start_time=datetime.utcnow(),
            end_time=None,
            model_a={"run_id": config.model_a_run_id, "samples": 0, "metrics": {}},
            model_b={"run_id": config.model_b_run_id, "samples": 0, "metrics": {}},
        )
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        self.results[experiment_id].status = ExperimentStatus.RUNNING
        self.results[experiment_id].start_time = datetime.utcnow()
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id not in self.experiments:
            return False
        
        self.results[experiment_id].status = ExperimentStatus.COMPLETED
        self.results[experiment_id].end_time = datetime.utcnow()
        self._analyze_experiment(experiment_id)
        return True
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to variant A or B using consistent hashing."""
        if experiment_id not in self.experiments:
            return "A"
        
        config = self.experiments[experiment_id]
        
        # Consistent assignment based on user_id + experiment_id
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        assignment = "B" if (hash_val / 2**128) < config.traffic_split else "A"
        
        self.assignments[f"{experiment_id}:{user_id}"] = assignment
        return assignment
    
    def record_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        value: float,
    ):
        """Record a metric for a user in an experiment."""
        key = f"{experiment_id}:{user_id}"
        if key not in self.assignments:
            return
        
        variant = self.assignments[key]
        self.metrics[experiment_id][f"{variant}_{metric_name}"].append(value)
        
        # Update running results
        result = self.results[experiment_id]
        if variant == "A":
            result.model_a["samples"] += 1
            if metric_name not in result.model_a["metrics"]:
                result.model_a["metrics"][metric_name] = []
            result.model_a["metrics"][metric_name].append(value)
        else:
            result.model_b["samples"] += 1
            if metric_name not in result.model_b["metrics"]:
                result.model_b["metrics"][metric_name] = []
            result.model_b["metrics"][metric_name].append(value)
    
    def _analyze_experiment(self, experiment_id: str):
        """Analyze experiment results using statistical tests."""
        import numpy as np
        from scipy import stats
        
        config = self.experiments[experiment_id]
        result = self.results[experiment_id]
        
        metric_a = result.model_a["metrics"].get(config.success_metric, [])
        metric_b = result.model_b["metrics"].get(config.success_metric, [])
        
        if len(metric_a) < config.min_samples or len(metric_b) < config.min_samples:
            result.winner = "inconclusive"
            return
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(metric_b, metric_a, equal_var=False)
        
        mean_a = np.mean(metric_a)
        mean_b = np.mean(metric_b)
        
        # Confidence interval for difference
        se = np.sqrt(np.var(metric_a) / len(metric_a) + np.var(metric_b) / len(metric_b))
        alpha = 1 - config.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)
        margin = z * se
        diff = mean_b - mean_a
        
        result.p_value = float(p_value)
        result.confidence_interval = [float(diff - margin), float(diff + margin)]
        result.lift = float((mean_b - mean_a) / mean_a) if mean_a != 0 else 0
        result.statistical_significance = p_value < alpha
        
        # Determine winner
        if result.statistical_significance and diff > 0:
            result.winner = "B"
        elif result.statistical_significance and diff < 0:
            result.winner = "A"
        else:
            result.winner = "tie"
    
    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get current experiment status."""
        return self.results.get(experiment_id)
    
    def list_experiments(self) -> List[ExperimentResult]:
        """List all experiments."""
        return list(self.results.values())
    
    def promote_winner(self, experiment_id: str) -> bool:
        """Promote winning model to champion in MLflow."""
        result = self.results.get(experiment_id)
        if not result or result.winner != "B":
            return False
        
        config = self.experiments[experiment_id]
        promoted = self.tracker.promote_challenger(
            run_id=config.model_b_run_id,
            min_improvement=config.minimum_detectable_effect,
        )
        
        if promoted:
            result.status = ExperimentStatus.COMPLETED
        return promoted


# Global A/B test manager
_ab_manager: Optional[ABTestManager] = None


def get_ab_manager() -> ABTestManager:
    """Get or create A/B test manager singleton."""
    global _ab_manager
    if _ab_manager is None:
        _ab_manager = ABTestManager()
    return _ab_manager


async def run_ab_test(
    champion_run_id: str,
    challenger_run_id: str,
    experiment_name: str = "model_comparison",
    duration_hours: int = 24,
) -> ExperimentResult:
    """Run a quick A/B test between two model versions."""
    manager = get_ab_manager()
    
    config = ExperimentConfig(
        name=experiment_name,
        description=f"Compare {champion_run_id[:8]} vs {challenger_run_id[:8]}",
        model_a_run_id=champion_run_id,
        model_b_run_id=challenger_run_id,
        traffic_split=0.5,
        min_samples=50,
        max_duration_days=duration_hours / 24,
    )
    
    experiment_id = manager.create_experiment(config)
    manager.start_experiment(experiment_id)
    
    # Simulate experiment (in production, this would run over real traffic)
    # For demo, we'll load models and test on recent data
    from .data_sources import AsyncDataFetcher
    from .predictor_async import EnrichedConfig, get_prediction_with_intervals
    
    fetcher = AsyncDataFetcher()
    await fetcher.__aenter__()
    
    try:
        config_model = EnrichedConfig()
        
        # Generate test predictions from both models
        # In real implementation, load specific model versions
        n_tests = 100
        
        for i in range(n_tests):
            user_id = f"test_user_{i}"
            variant = manager.assign_variant(experiment_id, user_id)
            
            # Get prediction (same for both in this demo)
            pred_result = await get_prediction_with_intervals(fetcher, config_model)
            
            # Convert numpy types
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
            
            pred_result = {k: to_native(v) for k, v in pred_result.items()}
            
            # Simulate actual outcome (random for demo)
            actual_correct = random.random() < pred_result['confidence'] / 100
            
            manager.record_metric(
                experiment_id, user_id, "direction_accuracy",
                1.0 if actual_correct else 0.0
            )
            manager.record_metric(
                experiment_id, user_id, "confidence",
                pred_result['confidence'] / 100
            )
        
        manager.stop_experiment(experiment_id)
        
        return manager.get_experiment_status(experiment_id)
        
    finally:
        await fetcher.__aexit__(None, None, None)


if __name__ == "__main__":
    manager = get_ab_manager()
    
    # Demo
    config = ExperimentConfig(
        name="demo_experiment",
        description="Demo A/B test",
        model_a_run_id="run_a123",
        model_b_run_id="run_b456",
    )
    
    exp_id = manager.create_experiment(config)
    print(f"Created experiment: {exp_id}")
    
    manager.start_experiment(exp_id)
    
    # Simulate some traffic
    for i in range(100):
        user_id = f"user_{i}"
        variant = manager.assign_variant(exp_id, user_id)
        manager.record_metric(exp_id, user_id, "direction_accuracy", random.random() > 0.5)
    
    manager.stop_experiment(exp_id)
    
    result = manager.get_experiment_status(exp_id)
    print(f"Winner: {result.winner}")
    print(f"P-value: {result.p_value}")
    print(f"Lift: {result.lift:.2%}")
