"""
Real-time dashboard for BTC Predictor with prediction vs actual visualization.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from .predictor_async import generate_async_prediction, get_fetcher, close_fetcher
from .data_sources import AsyncDataFetcher


@dataclass
class DashboardDataPoint:
    """Single data point for dashboard."""
    timestamp: str
    actual_price: float
    predicted_price: Optional[float] = None
    predicted_direction: Optional[str] = None
    confidence: Optional[int] = None
    ensemble_lower: Optional[float] = None
    ensemble_upper: Optional[float] = None


class DashboardManager:
    """Manages dashboard data and WebSocket connections."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.price_history: List[DashboardDataPoint] = []
        self.prediction_history: List[Dict] = []
        self.active_connections: List[WebSocket] = []
        self.fetcher: Optional[AsyncDataFetcher] = None
        self.running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start dashboard data collection."""
        self.fetcher = AsyncDataFetcher()
        await self.fetcher.__aenter__()
        self.running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._collect_prices()),
            asyncio.create_task(self._collect_predictions()),
        ]
    
    async def stop(self):
        """Stop dashboard."""
        self.running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.fetcher:
            await self.fetcher.__aexit__(None, None, None)
    
    async def _collect_prices(self):
        """Collect live prices."""
        while self.running:
            try:
                price = await self.fetcher.get_live_price()
                if price:
                    point = DashboardDataPoint(
                        timestamp=datetime.utcnow().isoformat(),
                        actual_price=price,
                    )
                    self.price_history.append(point)
                    if len(self.price_history) > self.max_points:
                        self.price_history = self.price_history[-self.max_points:]
                    
                    # Broadcast to WebSocket clients
                    await self._broadcast_price(point)
                
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                print(f"Price collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_predictions(self):
        """Collect predictions at 15-minute intervals."""
        while self.running:
            try:
                # Wait for next 15-minute boundary
                now = datetime.now()
                minutes_until = (15 - now.minute % 15) % 15
                if minutes_until == 0:
                    minutes_until = 15
                seconds_until = minutes_until * 60 - now.second
                
                await asyncio.sleep(seconds_until)
                
                if not self.running:
                    break
                
                # Generate prediction
                fetcher = await get_fetcher()
                from .predictor_async import get_prediction_with_intervals, EnrichedConfig
                result = await get_prediction_with_intervals(fetcher, EnrichedConfig())
                await close_fetcher()
                
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
                
                # Store prediction
                pred_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "window_start": result['window_start'],
                    "window_end": result['window_end'],
                    "predicted_price": result['next_price'],
                    "current_price": result['current_price'],
                    "change_pct": result['change_pct'],
                    "direction": result['direction'],
                    "confidence": result['confidence'],
                    "ensemble_interval": result['ensemble_interval'],
                    "model": result['model'],
                }
                self.prediction_history.append(pred_record)
                
                # Update latest price point with prediction
                if self.price_history:
                    self.price_history[-1].predicted_price = result['next_price']
                    self.price_history[-1].predicted_direction = result['direction']
                    self.price_history[-1].confidence = result['confidence']
                    self.price_history[-1].ensemble_lower = result['ensemble_interval'][0]
                    self.price_history[-1].ensemble_upper = result['ensemble_interval'][1]
                
                # Broadcast to WebSocket clients
                await self._broadcast_prediction(pred_record)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Prediction collection error: {e}")
                await asyncio.sleep(60)
    
    async def _broadcast_price(self, point: DashboardDataPoint):
        """Broadcast price update to WebSocket clients."""
        message = json.dumps({
            "type": "price",
            "data": asdict(point),
        })
        await self._broadcast(message)
    
    async def _broadcast_prediction(self, pred: Dict):
        """Broadcast prediction to WebSocket clients."""
        message = json.dumps({
            "type": "prediction",
            "data": pred,
        })
        await self._broadcast(message)
    
    async def _broadcast(self, message: str):
        """Broadcast to all connected clients."""
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        
        for ws in disconnected:
            self.active_connections.remove(ws)
    
    def add_connection(self, ws: WebSocket):
        """Add WebSocket connection."""
        self.active_connections.append(ws)
    
    def remove_connection(self, ws: WebSocket):
        """Remove WebSocket connection."""
        if ws in self.active_connections:
            self.active_connections.remove(ws)
    
    def get_chart_data(self) -> Dict[str, Any]:
        """Get data for chart rendering."""
        return {
            "prices": [
                {
                    "timestamp": p.timestamp,
                    "actual": p.actual_price,
                    "predicted": p.predicted_price,
                    "ensemble_lower": p.ensemble_lower,
                    "ensemble_upper": p.ensemble_upper,
                }
                for p in self.price_history[-200:]  # Last 200 points
            ],
            "predictions": self.prediction_history[-20:],  # Last 20 predictions
            "current_price": self.price_history[-1].actual_price if self.price_history else None,
            "latest_prediction": self.prediction_history[-1] if self.prediction_history else None,
        }


# Global dashboard manager
dashboard = DashboardManager()


# Dashboard FastAPI app
dash_app = FastAPI(
    title="BTC Predictor Dashboard",
    description="Real-time BTC price prediction dashboard",
    version="2.0.0",
)

# Templates
templates = Jinja2Templates(directory="templates")


@dash_app.on_event("startup")
async def startup():
    await dashboard.start()


@dash_app.on_event("shutdown")
async def shutdown():
    await dashboard.stop()


@dash_app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Serve dashboard HTML."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@dash_app.get("/api/data")
async def get_dashboard_data():
    """Get dashboard data as JSON."""
    return dashboard.get_chart_data()


@dash_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    dashboard.add_connection(websocket)
    
    try:
        # Send initial data
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": dashboard.get_chart_data(),
        }))
        
        while True:
            data = await websocket.receive_text()
            # Handle client messages (ping, subscriptions, etc.)
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    
    except WebSocketDisconnect:
        dashboard.remove_connection(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        dashboard.remove_connection(websocket)


# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Predictor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #e6edf3;
            min-height: 100vh;
        }
        .header {
            background: #161b22;
            border-bottom: 1px solid #30363d;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .status { display: flex; align-items: center; gap: 1rem; }
        .status-indicator {
            width: 10px; height: 10px; border-radius: 50%;
            background: #3fb950;
        }
        .status-indicator.disconnected { background: #f85149; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 2rem;
        }
        @media (max-width: 1000px) {
            .grid { grid-template-columns: 1fr; }
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1.5rem;
        }
        .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #8b949e; }
        .chart-container { position: relative; height: 500px; }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        .metric {
            background: #21262d;
            border-radius: 6px;
            padding: 1rem;
        }
        .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }
        .metric-value { font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }
        .metric-value.up { color: #3fb950; }
        .metric-value.down { color: #f85149; }
        .prediction-list { max-height: 400px; overflow-y: auto; }
        .prediction-item {
            background: #21262d;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid #30363d;
        }
        .prediction-item.up { border-left-color: #3fb950; }
        .prediction-item.down { border-left-color: #f85149; }
        .pred-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
        .pred-time { font-size: 0.75rem; color: #8b949e; }
        .pred-window { font-size: 0.75rem; font-weight: 500; }
        .pred-details { display: flex; justify-content: space-between; font-size: 0.875rem; }
        .pred-price { font-weight: 600; }
        .pred-confidence { color: #8b949e; }
        .interval-bar {
            height: 4px;
            background: #30363d;
            border-radius: 2px;
            margin-top: 0.5rem;
            overflow: hidden;
        }
        .interval-fill {
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #58a6ff);
            border-radius: 2px;
        }
        .legend { display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.75rem; }
        .legend-item { display: flex; align-items: center; gap: 0.5rem; }
        .legend-color { width: 16px; height: 3px; border-radius: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 BTC Predictor Dashboard</h1>
        <div class="status">
            <span id="connection-status">Connecting...</span>
            <div class="status-indicator" id="status-indicator"></div>
        </div>
    </div>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>Price Chart (Prediction vs Actual)</h2>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #58a6ff;"></div>
                        <span>Actual Price</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f7816f;"></div>
                        <span>Predicted</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgba(88, 166, 255, 0.2);"></div>
                        <span>Confidence Interval</span>
                    </div>
                </div>
            </div>
            <div>
                <div class="card">
                    <h2>Current Metrics</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value" id="current-price">--</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Next Prediction</div>
                            <div class="metric-value" id="next-prediction">--</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Expected Change</div>
                            <div class="metric-value" id="expected-change">--</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value" id="confidence">--</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h2>Recent Predictions</h2>
                    <div class="prediction-list" id="prediction-list">
                        <div style="color: #8b949e; text-align: center; padding: 2rem;">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const statusEl = document.getElementById('connection-status');
        const indicatorEl = document.getElementById('status-indicator');
        const chartEl = document.getElementById('priceChart');
        let chart = null;
        
        ws.onopen = () => {
            statusEl.textContent = 'Connected';
            indicatorEl.classList.remove('disconnected');
        };
        ws.onclose = () => {
            statusEl.textContent = 'Disconnected';
            indicatorEl.classList.add('disconnected');
            setTimeout(() => window.location.reload(), 5000);
        };
        ws.onerror = () => {
            statusEl.textContent = 'Error';
            indicatorEl.classList.add('disconnected');
        };
        
        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'init' || msg.type === 'price' || msg.type === 'prediction') {
                updateDashboard(msg.data);
            }
        };
        
        function initChart(data) {
            const prices = data.prices || [];
            const labels = prices.map(p => new Date(p.timestamp).toLocaleTimeString());
            const actual = prices.map(p => p.actual);
            const predicted = prices.map(p => p.predicted);
            const lower = prices.map(p => p.ensemble_lower);
            const upper = prices.map(p => p.ensemble_upper);
            
            const ctx = chartEl.getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: actual,
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.2,
                            pointRadius: 0,
                        },
                        {
                            label: 'Predicted',
                            data: predicted,
                            borderColor: '#f7816f',
                            backgroundColor: 'rgba(247, 129, 111, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.2,
                            pointRadius: 0,
                        },
                        {
                            label: 'Upper Bound',
                            data: upper,
                            borderColor: 'rgba(88, 166, 255, 0.3)',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            borderWidth: 1,
                            fill: '-1',
                            pointRadius: 0,
                        },
                        {
                            label: 'Lower Bound',
                            data: lower,
                            borderColor: 'rgba(88, 166, 255, 0.3)',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            borderWidth: 1,
                            fill: false,
                            pointRadius: 0,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: { backgroundColor: '#21262d', titleColor: '#e6edf3', bodyColor: '#e6edf3' }
                    },
                    scales: {
                        x: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } },
                        y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', callback: v => '$' + v.toLocaleString() } }
                    }
                }
            });
        }
        
        function updateChart(data) {
            if (!chart) return;
            const prices = data.prices || [];
            const labels = prices.map(p => new Date(p.timestamp).toLocaleTimeString());
            const actual = prices.map(p => p.actual);
            const predicted = prices.map(p => p.predicted);
            const lower = prices.map(p => p.ensemble_lower);
            const upper = prices.map(p => p.ensemble_upper);
            
            chart.data.labels = labels;
            chart.data.datasets[0].data = actual;
            chart.data.datasets[1].data = predicted;
            chart.data.datasets[2].data = upper;
            chart.data.datasets[3].data = lower;
            chart.update('none');
        }
        
        function updateDashboard(data) {
            if (!chart) {
                initChart(data);
            } else {
                updateChart(data);
            }
            
            // Update metrics
            if (data.current_price) {
                document.getElementById('current-price').textContent = '$' + data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }
            if (data.latest_prediction) {
                const p = data.latest_prediction;
                document.getElementById('next-prediction').textContent = '$' + (p.predicted_price || p.next_price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                const change = p.change_pct || 0;
                const changeEl = document.getElementById('expected-change');
                changeEl.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
                changeEl.className = 'metric-value ' + (change >= 0 ? 'up' : 'down');
                document.getElementById('confidence').textContent = (p.confidence || 0) + '%';
            }
            
            // Update prediction list
            if (data.predictions) {
                const list = document.getElementById('prediction-list');
                list.innerHTML = data.predictions.slice().reverse().map(p => {
                    const isUp = (p.direction || '').includes('UP');
                    const change = p.change_pct || 0;
                    const interval = p.ensemble_interval || [0, 0];
                    const width = interval[1] > interval[0] ? 100 : 0;
                    return `
                        <div class="prediction-item ${isUp ? 'up' : 'down'}">
                            <div class="pred-header">
                                <span class="pred-time">${new Date(p.timestamp).toLocaleTimeString()}</span>
                                <span class="pred-window">${p.window_start} → ${p.window_end}</span>
                            </div>
                            <div class="pred-details">
                                <span class="pred-price">$${(p.predicted_price || p.next_price || 0).toLocaleString()}</span>
                                <span class="pred-confidence">${p.confidence || 0}% conf</span>
                            </div>
                            <div class="pred-details">
                                <span style="color: ${isUp ? '#3fb950' : '#f85149'}">${p.direction || ''} ${Math.abs(change).toFixed(2)}%</span>
                                <span class="pred-confidence">Interval: $${interval[0]?.toLocaleString() || 0} - $${interval[1]?.toLocaleString() || 0}</span>
                            </div>
                        </div>
                    `.join('');
                }
            }
        }
    </script>
</body>
</html>
"""


def create_dashboard_templates():
    """Create dashboard HTML template."""
    import os
    os.makedirs("templates", exist_ok=True)
    with open("templates/dashboard.html", "w") as f:
        f.write(DASHBOARD_HTML)


if __name__ == "__main__":
    create_dashboard_templates()
    uvicorn.run("dashboard:dash_app", host="0.0.0.0", port=8002, reload=True)