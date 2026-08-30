"""
Scheduled retraining pipeline using Prefect for orchestration.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from prefect import flow, task, get_run_logger
    from prefect.schedules import CronSchedule
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule as ServerCronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Mock decorators for when Prefect isn't installed
    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .predictor_async import (
    AsyncDataFetcher,
    EnrichedConfig,
    get_prediction_with_intervals,
)
from .mlflow_tracking import BTCModelTracker, get_tracker
from .data_sources import prepare_features


@dataclass
class RetrainingConfig:
    """Configuration for retraining pipeline."""
    schedule: str = "0 2 * * *"  # Daily at 2 AM UTC
    lookback_days: int = 60
    min_samples: int = 1000
    validation_split: float = 0.2
    model_registry_path: str = "./models/registry"
    champion_metric: str = "direction_accuracy"
    challenger_threshold: float = 0.02  # 2% improvement required
    max_training_time_minutes: int = 60
    notification_webhook: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of a training run."""
    success: bool
    run_id: Optional[str] = None
    model_type: str = ""
    metrics: Dict[str, float] = None
    error: Optional[str] = None
    duration_seconds: float = 0
    promoted: bool = False


class RetrainingPipeline:
    """Orchestrates model retraining with MLflow tracking and champion/challenger logic."""
    
    def __init__(self, config: RetrainingConfig = None):
        self.config = config or RetrainingConfig()
        self.tracker = get_tracker()
        self.results: List[TrainingResult] = []
    
    @task(retries=2, retry_delay_seconds=60)
    async def fetch_training_data(self, fetcher: AsyncDataFetcher, config: EnrichedConfig) -> Dict[str, Any]:
        """Fetch and prepare training data."""
        logger = get_run_logger() if PREFECT_AVAILABLE else None
        
        # Fetch historical data
        df = await fetcher.get_historical_15m(days=self.config.lookback_days)
        
        if df is None or len(df) < self.config.min_samples:
            raise ValueError(f"Insufficient training data: {len(df) if df is not None else 0} samples")
        
        # Prepare features
        features_df = prepare_features(df)
        
        return {
            "dataframe": features_df,
            "n_samples": len(features_df),
            "date_range": {
                "start": features_df.index[0].isoformat(),
                "end": features_df.index[-1].isoformat(),
            }
        }
    
    @task
    async def train_lstm_models(
        self, 
        training_data: Dict[str, Any], 
        config: EnrichedConfig,
    ) -> Dict[str, Any]:
        """Train LSTM ensemble models."""
        logger = get_run_logger() if PREFECT_AVAILABLE else None
        if logger:
            logger.info(f"Training LSTM ensemble with {config.lstm_n_ensemble} models")
        
        from .predictor import train_lstm_model
        
        models = []
        for i in range(config.lstm_n_ensemble):
            model = train_lstm_model(training_data["dataframe"], config)
            models.append(model)
        
        # Evaluate on validation set
        val_size = int(len(training_data["dataframe"]) * self.config.validation_split)
        val_data = training_data["dataframe"].tail(val_size)
        
        # Quick validation predictions
        val_predictions = []
        for model in models:
            # Simplified - would need proper sequence preparation
            pass
        
        return {
            "models": models,
            "n_models": len(models),
            "validation_samples": val_size,
        }
    
    @task
    async def train_prophet_model(
        self,
        training_data: Dict[str, Any],
        config: EnrichedConfig,
    ) -> Dict[str, Any]:
        """Train Prophet model."""
        logger = get_run_logger() if PREFECT_AVAILABLE else None
        if logger:
            logger.info("Training Prophet model")
        
        from .predictor import train_prophet_model
        
        model = train_prophet_model(training_data["dataframe"], config)
        
        return {
            "model": model,
        }
    
    @task
    async def evaluate_and_register(
        self,
        lstm_result: Dict[str, Any],
        prophet_result: Dict[str, Any],
        training_data: Dict[str, Any],
        config: EnrichedConfig,
    ) -> TrainingResult:
        """Evaluate models and register to MLflow with champion/challenger logic."""
        logger = get_run_logger() if PREFECT_AVAILABLE else None
        
        start_time = datetime.now()
        
        try:
            # Log LSTM training
            lstm_run_id = self.tracker.log_training_run(
                model_type="lstm",
                params={
                    "sequence_length": config.sequence_length,
                    "n_features": config.n_features,
                    "units": str(config.lstm_units),
                    "dropout": config.dropout,
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "n_ensemble": config.lstm_n_ensemble,
                },
                metrics={
                    "training_time_seconds": 0,  # Would track actual time
                    "n_models": lstm_result["n_models"],
                },
                model_obj=lstm_result["models"][0] if lstm_result["models"] else None,
                tags={"ensemble_size": str(config.lstm_n_ensemble)},
            )
            
            # Log Prophet training
            prophet_run_id = self.tracker.log_training_run(
                model_type="prophet",
                params={
                    "changepoint_prior_scale": config.prophet_changepoint_prior_scale,
                    "seasonality_prior_scale": config.prophet_seasonality_prior_scale,
                    "daily_seasonality": config.prophet_daily_seasonality,
                    "weekly_seasonality": config.prophet_weekly_seasonality,
                },
                metrics={
                    "training_time_seconds": 0,
                },
                model_obj=prophet_result["model"],
            )
            
            # Evaluate champion vs challenger
            promoted = self.tracker.promote_challenger(
                run_id=lstm_run_id,
                min_improvement=self.config.challenger_threshold,
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = TrainingResult(
                success=True,
                run_id=lstm_run_id,
                model_type="ensemble",
                metrics={"direction_accuracy": 0.0},  # Would compute actual
                duration_seconds=duration,
                promoted=promoted,
            )
            
            if logger:
                logger.info(f"Retraining complete. Promoted: {promoted}")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            result = TrainingResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            if logger:
                logger.error(f"Retraining failed: {e}")
            return result
    
    @task
    async def notify_completion(self, result: TrainingResult):
        """Send notification on completion."""
        logger = get_run_logger() if PREFECT_AVAILABLE else None
        
        message = f"""
BTC Predictor Retraining {'✅ Succeeded' if result.success else '❌ Failed'}
- Model: {result.model_type}
- Run ID: {result.run_id or 'N/A'}
- Promoted: {'Yes' if result.promoted else 'No'}
- Duration: {result.duration_seconds:.1f}s
- Error: {result.error or 'None'}
        """.strip()
        
        if logger:
            logger.info(message)
        
        # Webhook notification
        if self.config.notification_webhook:
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.config.notification_webhook,
                        json={"text": message},
                    )
            except Exception as e:
                if logger:
                    logger.warning(f"Notification failed: {e}")


@flow(name="btc-predictor-retraining", description="Daily model retraining pipeline")
async def retraining_flow(config: RetrainingConfig = None):
    """Main retraining flow."""
    pipeline_config = config or RetrainingConfig()
    pipeline = RetrainingPipeline(pipeline_config)
    
    fetcher = AsyncDataFetcher()
    await fetcher.__aenter__()
    
    try:
        model_config = EnrichedConfig()
        
        # Fetch training data
        training_data = await pipeline.fetch_training_data(fetcher, model_config)
        
        # Train models in parallel
        lstm_result = await pipeline.train_lstm_models(training_data, model_config)
        prophet_result = await pipeline.train_prophet_model(training_data, model_config)
        
        # Evaluate and register
        result = await pipeline.evaluate_and_register(
            lstm_result, prophet_result, training_data, model_config
        )
        
        # Notify
        await pipeline.notify_completion(result)
        
        return result
        
    finally:
        await fetcher.__aexit__(None, None, None)


def create_deployment():
    """Create Prefect deployment for scheduled retraining."""
    if not PREFECT_AVAILABLE:
        print("Prefect not installed. Install with: pip install prefect")
        return
    
    config = RetrainingConfig()
    
    deployment = Deployment.build_from_flow(
        flow=retraining_flow,
        name="daily-retraining",
        schedule=ServerCronSchedule(cron=config.schedule),
        parameters={"config": config},
        work_queue_name="ml-training",
        tags=["btc-predictor", "retraining", "ml"],
    )
    
    deployment.apply()
    print(f"Deployment created: {deployment.name}")


# Simple scheduler without Prefect (fallback)
class SimpleScheduler:
    """Simple async scheduler for retraining without Prefect."""
    
    def __init__(self, config: RetrainingConfig = None):
        self.config = config or RetrainingConfig()
        self.pipeline = RetrainingPipeline(self.config)
        self.running = False
    
    async def run_once(self) -> TrainingResult:
        """Run retraining once."""
        return await retraining_flow(self.config)
    
    async def start(self):
        """Start scheduled retraining."""
        self.running = True
        
        # Parse cron schedule (simplified - only supports daily at specific hour/minute)
        # Format: "minute hour * * *"
        parts = self.config.schedule.split()
        if len(parts) >= 2:
            minute = int(parts[0])
            hour = int(parts[1])
        else:
            minute, hour = 0, 2
        
        print(f"Scheduler started: daily at {hour:02d}:{minute:02d} UTC")
        
        while self.running:
            now = datetime.utcnow()
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if next_run <= now:
                next_run += timedelta(days=1)
            
            sleep_seconds = (next_run - now).total_seconds()
            print(f"Next retraining in {sleep_seconds/3600:.1f} hours")
            
            await asyncio.sleep(sleep_seconds)
            
            if not self.running:
                break
            
            print("Starting scheduled retraining...")
            result = await self.run_once()
            print(f"Retraining {'succeeded' if result.success else 'failed'}")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        create_deployment()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(retraining_flow())
    else:
        # Run scheduler
        scheduler = SimpleScheduler()
        try:
            asyncio.run(scheduler.start())
        except KeyboardInterrupt:
            scheduler.stop()
            print("Scheduler stopped")
