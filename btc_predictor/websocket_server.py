"""
WebSocket server for real-time BTC predictions streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .predictor_async import generate_async_prediction, get_fetcher, close_fetcher
from .data_sources import AsyncDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionMessage:
    """WebSocket message for predictions."""
    type: str = "prediction"
    window_start: str = ""
    window_end: str = ""
    current_price: float = 0.0
    next_price: float = 0.0
    change_pct: float = 0.0
    direction: str = ""
    confidence: int = 0
    ensemble_interval: list = None
    enriched_features: dict = None
    model: str = ""
    timestamp: str = ""
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class PriceUpdateMessage:
    """WebSocket message for live price updates."""
    type: str = "price_update"
    price: float = 0.0
    timestamp: str = ""
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class StatusMessage:
    """WebSocket message for status updates."""
    type: str = "status"
    status: str = ""
    message: str = ""
    timestamp: str = ""
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.client_info: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.client_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow().isoformat(),
            "subscriptions": ["predictions", "prices"],
        }
        logger.info(f"Client connected: {self.client_info[websocket]['client_id']} "
                   f"(total: {len(self.active_connections)})")
        
        # Send welcome message
        await self.send_personal_message(
            websocket,
            StatusMessage(
                status="connected",
                message=f"Connected to BTC Predictor WebSocket",
                timestamp=datetime.utcnow().isoformat(),
            ).to_json()
        )
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        if websocket in self.active_connections:
            client_id = self.client_info.get(websocket, {}).get("client_id", "unknown")
            self.active_connections.remove(websocket)
            self.client_info.pop(websocket, None)
            logger.info(f"Client disconnected: {client_id} "
                       f"(remaining: {len(self.active_connections)})")
    
    async def send_personal_message(self, websocket: WebSocket, message: str):
        """Send message to specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str, subscription: str = None):
        """Broadcast message to all connected clients."""
        disconnected = set()
        
        for websocket in self.active_connections:
            # Check subscription
            client_subs = self.client_info.get(websocket, {}).get("subscriptions", [])
            if subscription and subscription not in client_subs:
                continue
            
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to client: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)


# Global connection manager
manager = ConnectionManager()


class PredictionStreamer:
    """Streams predictions at regular intervals."""
    
    def __init__(self, interval_seconds: int = 900):  # 15 minutes
        self.interval_seconds = interval_seconds
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start streaming predictions."""
        self.running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info(f"Prediction streamer started (interval: {self.interval_seconds}s)")
    
    async def stop(self):
        """Stop streaming."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Prediction streamer stopped")
    
    async def _stream_loop(self):
        """Main streaming loop."""
        while self.running:
            try:
                # Wait for next 15-minute boundary
                now = datetime.now()
                minutes_until_next = (15 - now.minute % 15) % 15
                if minutes_until_next == 0:
                    minutes_until_next = 15
                seconds_until = minutes_until_next * 60 - now.second
                
                logger.info(f"Next prediction in {seconds_until} seconds")
                await asyncio.sleep(seconds_until)
                
                if not self.running:
                    break
                
                # Generate prediction
                logger.info("Generating scheduled prediction for WebSocket clients...")
                prediction_text = await generate_async_prediction()
                
                # Parse prediction for structured message
                # For now, send as raw text + status
                await manager.broadcast(
                    StatusMessage(
                        status="prediction_generated",
                        message=prediction_text,
                        timestamp=datetime.utcnow().isoformat(),
                    ).to_json(),
                    subscription="predictions"
                )
                
                # Also try to get structured data
                try:
                    fetcher = await get_fetcher()
                    from .predictor_async import get_prediction_with_intervals, EnrichedConfig
                    result = await get_prediction_with_intervals(fetcher, EnrichedConfig())
                    
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
                    
                    result = {k: to_native(v) for k, v in result.items()}
                    
                    # Send structured prediction
                    pred_msg = PredictionMessage(
                        window_start=result['window_start'],
                        window_end=result['window_end'],
                        current_price=result['current_price'],
                        next_price=result['next_price'],
                        change_pct=result['change_pct'],
                        direction=result['direction'],
                        confidence=result['confidence'],
                        ensemble_interval=result['ensemble_interval'],
                        enriched_features=result['enriched_features'],
                        model=result['model'],
                        timestamp=datetime.utcnow().isoformat(),
                    )
                    
                    await manager.broadcast(
                        pred_msg.to_json(),
                        subscription="predictions"
                    )
                    
                    await close_fetcher()
                    
                except Exception as e:
                    logger.error(f"Failed to get structured prediction: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry


class PriceStreamer:
    """Streams live price updates."""
    
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.fetcher: Optional[AsyncDataFetcher] = None
    
    async def start(self):
        """Start streaming prices."""
        self.fetcher = AsyncDataFetcher()
        await self.fetcher.__aenter__()
        self.running = True
        self._task = asyncio.create_task(self._stream_loop())
        logger.info(f"Price streamer started (interval: {self.interval_seconds}s)")
    
    async def stop(self):
        """Stop streaming."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.fetcher:
            await self.fetcher.__aexit__(None, None, None)
        logger.info("Price streamer stopped")
    
    async def _stream_loop(self):
        """Main price streaming loop."""
        while self.running:
            try:
                price = await self.fetcher.get_live_price()
                
                if price:
                    await manager.broadcast(
                        PriceUpdateMessage(
                            price=price,
                            timestamp=datetime.utcnow().isoformat(),
                        ).to_json(),
                        subscription="prices"
                    )
                
                await asyncio.sleep(self.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price stream error: {e}")
                await asyncio.sleep(self.interval_seconds)


# Global streamers
prediction_streamer = PredictionStreamer()
price_streamer = PriceStreamer()


# FastAPI app with WebSocket
ws_app = FastAPI(
    title="BTC Predictor WebSocket",
    description="Real-time BTC predictions and price streaming",
    version="2.0.0",
)

ws_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@ws_app.on_event("startup")
async def startup():
    """Start streamers on startup."""
    await price_streamer.start()
    await prediction_streamer.start()


@ws_app.on_event("shutdown")
async def shutdown():
    """Stop streamers on shutdown."""
    await price_streamer.stop()
    await prediction_streamer.stop()


@ws_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive messages from client (for subscriptions, ping/pong, etc.)
            data = await websocket.receive_text()
            
            try:
                msg = json.loads(data)
                msg_type = msg.get("type")
                
                if msg_type == "subscribe":
                    # Update client subscriptions
                    subscriptions = msg.get("subscriptions", [])
                    if websocket in manager.client_info:
                        manager.client_info[websocket]["subscriptions"] = subscriptions
                    await manager.send_personal_message(
                        websocket,
                        StatusMessage(
                            status="subscribed",
                            message=f"Subscribed to: {', '.join(subscriptions)}",
                            timestamp=datetime.utcnow().isoformat(),
                        ).to_json()
                    )
                
                elif msg_type == "ping":
                    await manager.send_personal_message(
                        websocket,
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                    )
                
                elif msg_type == "get_prediction":
                    # Force generate prediction
                    await manager.send_personal_message(
                        websocket,
                        StatusMessage(
                            status="generating",
                            message="Generating prediction...",
                            timestamp=datetime.utcnow().isoformat(),
                        ).to_json()
                    )
                    prediction = await generate_async_prediction()
                    await manager.send_personal_message(
                        websocket,
                        StatusMessage(
                            status="prediction",
                            message=prediction,
                            timestamp=datetime.utcnow().isoformat(),
                        ).to_json()
                    )
                
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    websocket,
                    StatusMessage(
                        status="error",
                        message="Invalid JSON",
                        timestamp=datetime.utcnow().isoformat(),
                    ).to_json()
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@ws_app.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "active_connections": manager.get_connection_count(),
        "clients": [
            {
                "client_id": info["client_id"],
                "connected_at": info["connected_at"],
                "subscriptions": info["subscriptions"],
            }
            for info in manager.client_info.values()
        ],
    }


# Combined app for running both REST and WebSocket
def create_combined_app():
    """Create app with both REST API and WebSocket."""
    from .api import app as rest_app
    
    # Mount WebSocket routes
    rest_app.mount("/ws", ws_app)
    
    return rest_app


if __name__ == "__main__":
    uvicorn.run(
        "websocket_server:ws_app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )