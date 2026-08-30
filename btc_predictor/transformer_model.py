"""
Transformer/Attention model for BTC price prediction.
Adds attention mechanism as ensemble member alongside LSTM and Prophet.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    sequence_length: int = 96
    n_features: int = 12
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dff: int = 512
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10


class PositionalEncoding(layers.Layer):
    """Positional encoding for Transformer."""
    
    def __init__(self, sequence_length: int, d_model: int):
        super().__init__()
        self.pos_encoding = self._positional_encoding(sequence_length, d_model)
    
    def _positional_encoding(self, length: int, depth: int) -> np.ndarray:
        """Generate positional encodings."""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000 ** depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )
        
        return pos_encoding[np.newaxis, ...].astype(np.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split into multiple heads."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        return self.dense(output), attention_weights


class TransformerBlock(layers.Layer):
    """Transformer encoder block."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = models.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerPredictor:
    """Transformer-based price predictor."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model: Optional[models.Model] = None
        self.history: Optional[Dict] = None
    
    def build_model(self) -> models.Model:
        """Build the Transformer model."""
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))
        
        # Input projection
        x = layers.Dense(self.config.d_model)(inputs)
        
        # Positional encoding
        x = PositionalEncoding(self.config.sequence_length, self.config.d_model)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Transformer encoder blocks
        for _ in range(self.config.num_layers):
            x = TransformerBlock(
                self.config.d_model,
                self.config.num_heads,
                self.config.dff,
                self.config.dropout_rate
            )(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="TransformerPredictor")
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        
        self.model = model
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 1,
    ) -> Dict:
        """Train the Transformer model."""
        if self.model is None:
            self.build_model()
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose,
            ),
        ]
        
        self.history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=verbose,
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call build_model() and train() first.")
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath: str):
        """Save model."""
        if self.model:
            self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, config: TransformerConfig) -> 'TransformerPredictor':
        """Load model."""
        predictor = cls(config)
        predictor.model = models.load_model(filepath)
        return predictor


def build_transformer_model(config: TransformerConfig = None) -> TransformerPredictor:
    """Factory function to build Transformer predictor."""
    if config is None:
        config = TransformerConfig()
    return TransformerPredictor(config)


class TransformerEnsemble:
    """Ensemble of Transformer models for uncertainty quantification."""
    
    def __init__(self, config: TransformerConfig, n_models: int = 3):
        self.config = config
        self.n_models = n_models
        self.models: List[TransformerPredictor] = []
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: int = 1):
        """Train multiple Transformer models with different seeds."""
        for i in range(self.n_models):
            print(f"Training Transformer model {i+1}/{self.n_models}...")
            
            # Set different seed for each model
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)
            
            model = TransformerPredictor(self.config)
            model.train(X, y, verbose=verbose)
            self.models.append(model)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions with mean and std."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)  # (n_models, n_samples)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def predict_with_intervals(
        self, 
        X: np.ndarray, 
        alpha: float = 0.10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions with conformal prediction intervals."""
        mean_pred, std_pred = self.predict(X)
        
        # Conformal prediction interval (simplified)
        # In practice, use calibration set for proper conformal prediction
        z_score = 1.645  # 90% interval
        margin = z_score * std_pred
        
        lower = mean_pred - margin
        upper = mean_pred + margin
        
        return mean_pred, lower, upper


# Integration with existing ensemble
async def get_transformer_prediction(
    fetcher: 'AsyncDataFetcher',
    config: 'EnrichedConfig',
) -> Dict[str, Any]:
    """Generate Transformer prediction compatible with existing pipeline."""
    from .data_sources import prepare_features
    from .predictor import prepare_sequences
    
    # This is a placeholder - full implementation would need:
    # 1. Fetch and prepare data
    # 2. Train/load Transformer model
    # 3. Generate prediction
    
    return {
        "model": "Transformer",
        "prediction": None,
        "interval": None,
        "confidence": 0,
    }


if __name__ == "__main__":
    # Demo
    config = TransformerConfig(
        sequence_length=96,
        n_features=12,
        d_model=128,
        num_heads=8,
        num_layers=4,
    )
    
    predictor = TransformerPredictor(config)
    model = predictor.build_model()
    model.summary()
    
    print("\nTransformer model built successfully!")
    print(f"Parameters: {model.count_params():,}")
