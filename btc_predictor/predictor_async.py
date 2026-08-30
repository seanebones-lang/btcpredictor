"""
Async version of the BTC predictor with enriched features from multiple sources.
"""

import asyncio
import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from prophet import Prophet
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping

from .data_sources import AsyncDataFetcher, get_fetcher, close_fetcher
from .predictor import Config


@dataclass
class EnrichedConfig(Config):
    """Extended config with enriched features."""
    USE_BINANCE_FEATURES: bool = True
    USE_MACRO_FEATURES: bool = False
    ENSEMBLE_MODELS: int = 3  # Number of LSTM models for ensemble
    PREDICTION_INTERVAL_ALPHA: float = 0.1  # For conformal prediction


async def add_indicators_async(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe (async wrapper)."""
    df = df.copy()
    df["rsi"] = RSIIndicator(df["y"], window=14).rsi()
    df["macd"] = MACD(df["y"]).macd()
    df["ema_12"] = EMAIndicator(df["y"], window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(df["y"], window=26).ema_indicator()
    bb = BollingerBands(df["y"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["momentum_30m"] = df["y"].pct_change(periods=2)
    df["momentum_1h"] = df["y"].pct_change(periods=4)
    df["momentum_4h"] = df["y"].pct_change(periods=16)
    df["volatility_1h"] = df["y"].pct_change().rolling(4).std()
    df["volume_ma"] = df["volume"].rolling(12).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    return df.dropna()


async def build_deep_lstm_async(df: pd.DataFrame, lookback: int = 50) -> float:
    """Build and train a deep bidirectional LSTM model (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_lstm_sync, df, lookback)


def _build_lstm_sync(df: pd.DataFrame, lookback: int = 50) -> float:
    """Synchronous LSTM training for executor."""
    data = df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = Sequential([
        Input(shape=(lookback, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(48, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    last_seq = scaled[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_seq, verbose=0)
    return float(scaler.inverse_transform(pred_scaled)[0][0])


async def build_ensemble_lstm(df: pd.DataFrame, lookback: int = 50, n_models: int = 3) -> Tuple[float, float]:
    """Build ensemble of LSTM models for uncertainty quantification."""
    loop = asyncio.get_event_loop()
    
    # Train multiple models with different seeds
    tasks = [
        loop.run_in_executor(None, _build_lstm_sync, df, lookback)
        for _ in range(n_models)
    ]
    predictions = await asyncio.gather(*tasks)
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    return float(mean_pred), float(std_pred)


async def build_prophet_model_async(df: pd.DataFrame) -> Tuple[Prophet, float]:
    """Build Prophet model with regressors (async)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_prophet_sync, df)


def _build_prophet_sync(df: pd.DataFrame) -> Tuple[Prophet, float]:
    """Synchronous Prophet training."""
    regressors = ["rsi", "macd", "ema_12", "ema_26", "bb_upper", "bb_lower", 
                  "volume", "momentum_30m", "momentum_1h", "momentum_4h",
                  "volatility_1h", "volume_ratio"]
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        interval_width=0.9  # For prediction intervals
    )
    for col in regressors:
        if col in df.columns:
            model.add_regressor(col)
    
    train_df = df[["ds", "y"] + [c for c in regressors if c in df.columns]].copy()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=1, freq="15min")
    for col in regressors:
        if col in df.columns:
            future[col] = df[col].iloc[-1]
    
    forecast = model.predict(future)
    return model, float(forecast.iloc[-1]["yhat"])


async def get_prediction_with_intervals(
    fetcher: AsyncDataFetcher,
    config: EnrichedConfig
) -> Dict:
    """Generate prediction with uncertainty intervals."""
    
    # Get enriched data
    enriched = await fetcher.get_enriched_features()
    current_price = enriched.get("price") or 62669.0
    
    # Get historical data
    df = await fetcher.get_best_historical(config.HISTORY_DAYS)
    if df is None or len(df) < 100:
        return {"error": "Insufficient data"}
    
    df = await add_indicators_async(df)
    
    # Get predictions from both models
    prophet_model, prophet_pred = await build_prophet_model_async(df)
    lstm_mean, lstm_std = await build_ensemble_lstm(df, config.LOOKBACK_PERIODS, config.ENSEMBLE_MODELS)
    
    # Prophet prediction interval
    future = prophet_model.make_future_dataframe(periods=1, freq="15min")
    regressors = ["rsi", "macd", "ema_12", "ema_26", "bb_upper", "bb_lower", 
                  "volume", "momentum_30m", "momentum_1h", "momentum_4h",
                  "volatility_1h", "volume_ratio"]
    for col in regressors:
        if col in df.columns:
            future[col] = df[col].iloc[-1]
    prophet_forecast = prophet_model.predict(future)
    prophet_lower = float(prophet_forecast.iloc[-1]["yhat_lower"])
    prophet_upper = float(prophet_forecast.iloc[-1]["yhat_upper"])
    
    # Ensemble (LSTM-heavy for short-term)
    next_price = round((prophet_pred * config.PROPHET_WEIGHT + lstm_mean * config.LSTM_WEIGHT), 2)
    
    # Combined uncertainty
    prophet_uncertainty = (prophet_upper - prophet_lower) / 2
    combined_std = np.sqrt((prophet_uncertainty * config.PROPHET_WEIGHT) ** 2 + 
                           (lstm_std * config.LSTM_WEIGHT) ** 2)
    
    # Conformal prediction adjustment (simplified)
    alpha = config.PREDICTION_INTERVAL_ALPHA
    z_score = 1.645  # 90% interval
    margin = z_score * combined_std
    
    change = round(((next_price - current_price) / current_price) * 100, 2)
    direction = "📈 UP" if change > 0 else "📉 DOWN"
    
    # Confidence based on volatility and model agreement
    recent_vol = df["y"].tail(50).std() / df["y"].tail(50).mean()
    model_agreement = 1 - abs(prophet_pred - lstm_mean) / current_price
    confidence = max(70, min(95, int(85 * model_agreement - recent_vol * 50)))
    
    window_start, window_end = get_prediction_window()
    
    # Convert numpy types to native Python
    def to_native(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj
    
    result = {
        "window_start": window_start,
        "window_end": window_end,
        "current_price": current_price,
        "next_price": next_price,
        "change_pct": change,
        "direction": direction,
        "confidence": confidence,
        "prophet_pred": prophet_pred,
        "lstm_mean": lstm_mean,
        "lstm_std": lstm_std,
        "prophet_interval": [prophet_lower, prophet_upper],
        "ensemble_interval": [next_price - margin, next_price + margin],
        "enriched_features": {
            "funding_rate": enriched.get("funding_rate"),
            "open_interest": enriched.get("open_interest"),
            "liquidations_count": len(enriched.get("liquidations", [])) if enriched.get("liquidations") else 0,
        },
        "model": "Async Ensemble: Deep LSTM (x3) + Prophet w/ intervals"
    }
    
    # Convert numpy types to native Python for formatting
    def to_native(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj
    
    return to_native(result)


def get_prediction_window() -> Tuple[str, str]:
    """Determine the next 15-minute prediction window."""
    now = datetime.datetime.now()
    minute = now.minute
    hour = now.hour

    if minute < 15:
        return f"{hour:02d}:15", f"{hour:02d}:30"
    elif minute < 30:
        return f"{hour:02d}:30", f"{hour:02d}:45"
    elif minute < 45:
        return f"{hour:02d}:45", f"{(hour+1)%24:02d}:00"
    else:
        return f"{(hour+1)%24:02d}:00", f"{(hour+1)%24:02d}:15"


async def generate_async_prediction() -> str:
    """Generate async prediction with full output."""
    fetcher = await get_fetcher()
    config = EnrichedConfig()
    
    try:
        result = await get_prediction_with_intervals(fetcher, config)
        
        if result is None:
            return "Error: get_prediction_with_intervals returned None"
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Convert all numpy types to native Python
        def to_native(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj
        
        result = {k: to_native(v) for k, v in result.items()}
        
        window_start = result['window_start']
        window_end = result['window_end']
        current_price = result['current_price']
        change_pct = result['change_pct']
        direction = result['direction']
        next_price = result['next_price']
        ensemble_interval = result['ensemble_interval']
        confidence = result['confidence']
        enriched = result['enriched_features']
        model = result['model']
        
        return f"""**BTC 15-min Prediction** ({window_start} → {window_end})

Current: ${current_price:,.2f}
Expected move: {direction} {abs(change_pct)}%

Next target: ${next_price:,.2f}
90% Interval: ${ensemble_interval[0]:,.2f} - ${ensemble_interval[1]:,.2f}
Confidence: {confidence}%

Enriched: Funding={(enriched.get('funding_rate') or 0):.6f} OI={(enriched.get('open_interest') or 0):,.0f}
Model: {model}"""
    finally:
        await close_fetcher()


if __name__ == "__main__":
    print(asyncio.run(generate_async_prediction()))
