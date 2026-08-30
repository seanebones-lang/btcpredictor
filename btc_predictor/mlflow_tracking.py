"""
MLflow tracking for BTC Predictor model versioning and experiment management.
"""

import os
import json
import mlflow
import mlflow.keras
import mlflow.pyfunc
import mlflow.sklearn
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from .predictor_async import EnrichedConfig, get_prediction_with_intervals
from .data_sources import AsyncDataFetcher


@dataclass
class ModelMetadata:
    """Metadata for a trained model version."""
    run_id: str
    model_name: str
    version: int
    timestamp: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: Dict[str, str]
    artifacts: List[str]


class BTCModelTracker:
    """MLflow-based model tracking for BTC Predictor."""
    
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "btc-predictor",
        model_registry_name: str = "btc-predictor-ensemble",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.model_registry_name = model_registry_name
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
    
    def log_training_run(
        self,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_obj: Any = None,
        artifacts: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
    ) -> str:
        """Log a training run to MLflow."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Set default tags
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("framework", "tensorflow" if model_type == "lstm" else "prophet")
            mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
            
            # Log model artifact
            if model_obj is not None:
                if model_type == "lstm":
                    mlflow.keras.log_model(model_obj, "model")
                elif model_type == "prophet":
                    # Prophet models need custom serialization
                    import tempfile
                    import pickle
                    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                        pickle.dump(model_obj, f)
                        mlflow.log_artifact(f.name, "model")
                else:
                    mlflow.sklearn.log_model(model_obj, "model")
            
            # Log additional artifacts
            if artifacts:
                import tempfile
                for name, content in artifacts.items():
                    with tempfile.NamedTemporaryFile(mode='w', suffix=f"_{name}.json", delete=False) as f:
                        json.dump(content, f, default=str)
                        mlflow.log_artifact(f.name, "artifacts")
            
            run_id = run.info.run_id
            print(f"Logged training run: {run_id}")
            return run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str = None,
        alias: str = "champion",
    ) -> ModelMetadata:
        """Register a model from a run to the model registry."""
        model_name = model_name or self.model_registry_name
        
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Set alias
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(model_name, alias, model_version.version)
        
        # Get run details
        run = mlflow.get_run(run_id)
        
        metadata = ModelMetadata(
            run_id=run_id,
            model_name=model_name,
            version=model_version.version,
            timestamp=datetime.utcnow().isoformat(),
            metrics=run.data.metrics,
            params=run.data.params,
            tags=run.data.tags,
            artifacts=[f.path for f in client.list_artifacts(run_id)],
        )
        
        print(f"Registered model {model_name} v{model_version.version} as '{alias}'")
        return metadata
    
    def get_champion_model(self, model_name: str = None) -> Optional[ModelMetadata]:
        """Get the current champion model."""
        model_name = model_name or self.model_registry_name
        client = mlflow.MlflowClient()
        
        try:
            model_version = client.get_model_version_by_alias(model_name, "champion")
            run = mlflow.get_run(model_version.run_id)
            
            return ModelMetadata(
                run_id=model_version.run_id,
                model_name=model_name,
                version=model_version.version,
                timestamp=datetime.utcnow().isoformat(),
                metrics=run.data.metrics,
                params=run.data.params,
                tags=run.data.tags,
                artifacts=[f.path for f in client.list_artifacts(model_version.run_id)],
            )
        except Exception as e:
            print(f"No champion model found: {e}")
            return None
    
    def load_champion_model(self, model_name: str = None) -> Any:
        """Load the champion model for inference."""
        model_name = model_name or self.model_registry_name
        model_uri = f"models:/{model_name}@champion"
        
        try:
            # Try loading as Keras model first
            model = mlflow.keras.load_model(model_uri)
            return model
        except Exception:
            try:
                # Try as pyfunc (generic)
                model = mlflow.pyfunc.load_model(model_uri)
                return model
            except Exception as e:
                print(f"Failed to load champion model: {e}")
                return None
    
    def promote_challenger(
        self,
        run_id: str,
        model_name: str = None,
        min_improvement: float = 0.02,
    ) -> bool:
        """Promote a challenger model to champion if it beats the current champion."""
        model_name = model_name or self.model_registry_name
        client = mlflow.MlflowClient()
        
        # Get challenger metrics
        challenger_run = mlflow.get_run(run_id)
        challenger_metric = challenger_run.data.metrics.get("direction_accuracy", 0)
        
        # Get champion metrics
        champion = self.get_champion_model(model_name)
        if not champion:
            # No champion yet, promote automatically
            self.register_model(run_id, model_name, "champion")
            return True
        
        champion_metric = champion.metrics.get("direction_accuracy", 0)
        improvement = challenger_metric - champion_metric
        
        if improvement >= min_improvement:
            print(f"Promoting challenger: {challenger_metric:.4f} > {champion_metric:.4f} (improvement: {improvement:.4f})")
            self.register_model(run_id, model_name, "champion")
            # Demote old champion to 'archived'
            client.set_registered_model_alias(model_name, "archived", champion.version)
            return True
        else:
            print(f"Challenger not better: {challenger_metric:.4f} vs {champion_metric:.4f} (need {min_improvement:.4f})")
            return False
    
    def get_model_history(self, model_name: str = None, limit: int = 10) -> List[ModelMetadata]:
        """Get version history for a model."""
        model_name = model_name or self.model_registry_name
        client = mlflow.MlflowClient()
        
        versions = client.search_model_versions(f"name='{model_name}'")
        versions.sort(key=lambda v: int(v.version), reverse=True)
        
        history = []
        for v in versions[:limit]:
            run = mlflow.get_run(v.run_id)
            history.append(ModelMetadata(
                run_id=v.run_id,
                model_name=model_name,
                version=int(v.version),
                timestamp=v.creation_timestamp,
                metrics=run.data.metrics,
                params=run.data.params,
                tags=run.data.tags,
                artifacts=[f.path for f in client.list_artifacts(v.run_id)],
            ))
        
        return history
    
    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple model runs."""
        rows = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            row = {
                "run_id": run_id,
                "model_type": run.data.tags.get("model_type", "unknown"),
                **run.data.metrics,
                **{f"param_{k}": v for k, v in run.data.params.items()},
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def log_prediction_batch(
        self,
        predictions: List[Dict[str, Any]],
        model_version: str = None,
    ):
        """Log a batch of predictions for monitoring."""
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        ) as run:
            mlflow.set_tag("type", "prediction_batch")
            mlflow.set_tag("model_version", model_version or "unknown")
            mlflow.set_tag("count", len(predictions))
            
            # Log aggregate metrics
            if predictions:
                confidences = [p.get("confidence", 0) for p in predictions]
                mlflow.log_metric("avg_confidence", np.mean(confidences))
                mlflow.log_metric("min_confidence", np.min(confidences))
                mlflow.log_metric("max_confidence", np.max(confidences))
                
                changes = [p.get("change_pct", 0) for p in predictions]
                mlflow.log_metric("avg_predicted_change", np.mean(changes))
                
                # Save predictions as artifact
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='_predictions.json', delete=False) as f:
                    json.dump(predictions, f, default=str)
                    mlflow.log_artifact(f.name, "predictions")


async def run_training_with_tracking(
    tracker: BTCModelTracker,
    fetcher: AsyncDataFetcher,
    config: EnrichedConfig,
    model_type: str = "ensemble",
) -> str:
    """Run a training cycle with full MLflow tracking."""
    import time
    from .predictor import train_lstm_model, train_prophet_model
    
    start_time = time.time()
    
    if model_type in ["lstm", "ensemble"]:
        # Train LSTM
        print("Training LSTM models...")
        lstm_models = []
        for i in range(config.lstm_n_ensemble):
            model = train_lstm_model(fetcher, config)
            lstm_models.append(model)
        
        lstm_time = time.time() - start_time
        
        # Log LSTM training
        lstm_params = {
            "sequence_length": config.sequence_length,
            "n_features": config.n_features,
            "units": str(config.lstm_units),
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "n_ensemble": config.lstm_n_ensemble,
        }
        
        lstm_metrics = {
            "training_time_seconds": lstm_time,
            "n_models": len(lstm_models),
        }
        
        tracker.log_training_run(
            model_type="lstm",
            params=lstm_params,
            metrics=lstm_metrics,
            model_obj=lstm_models[0] if lstm_models else None,  # Log first model
            tags={"ensemble_size": str(config.lstm_n_ensemble)},
        )
    
    if model_type in ["prophet", "ensemble"]:
        # Train Prophet
        print("Training Prophet model...")
        prophet_model = train_prophet_model(fetcher, config)
        
        prophet_time = time.time() - start_time - (lstm_time if model_type == "ensemble" else 0)
        
        prophet_params = {
            "changepoint_prior_scale": config.prophet_changepoint_prior_scale,
            "seasonality_prior_scale": config.prophet_seasonality_prior_scale,
            "daily_seasonality": config.prophet_daily_seasonality,
            "weekly_seasonality": config.prophet_weekly_seasonality,
        }
        
        prophet_metrics = {
            "training_time_seconds": prophet_time,
        }
        
        tracker.log_training_run(
            model_type="prophet",
            params=prophet_params,
            metrics=prophet_metrics,
            model_obj=prophet_model,
        )
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    
    return "completed"


# Singleton tracker instance
_tracker: Optional[BTCModelTracker] = None


def get_tracker() -> BTCModelTracker:
    """Get or create MLflow tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = BTCModelTracker()
    return _tracker


if __name__ == "__main__":
    # Demo usage
    tracker = get_tracker()
    print(f"Experiment ID: {tracker.experiment_id}")
    print(f"Tracking URI: {tracker.tracking_uri}")
    
    # Show model history
    history = tracker.get_model_history()
    print(f"\nModel history ({len(history)} versions):")
    for h in history:
        print(f"  v{h.version}: run={h.run_id[:8]} dir_acc={h.metrics.get('direction_accuracy', 0):.4f}")
