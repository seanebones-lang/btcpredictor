"""
Backtest UI - Visual walk-forward results for BTC Predictor.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from .backtest import WalkForwardBacktest
from .data_sources import AsyncDataFetcher
from .predictor_async import EnrichedConfig


@dataclass
class BacktestChartData:
    """Data for backtest visualization."""
    timestamps: List[str]
    actual_prices: List[float]
    predicted_prices: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    direction_correct: List[bool]
    confidence: List[int]


# Backtest UI app
bt_app = FastAPI(
    title="BTC Predictor Backtest UI",
    description="Visual walk-forward backtest results",
    version="2.0.0",
)

templates = Jinja2Templates(directory="templates")


# HTML template for backtest UI
BACKTEST_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Predictor - Backtest Results</title>
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
        .header-actions { display: flex; gap: 1rem; }
        .btn {
            background: #238636;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
        }
        .btn:hover { background: #2ea043; }
        .btn.secondary { background: #30363d; }
        .btn.secondary:hover { background: #484f58; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1.5rem;
        }
        .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }
        .metric-value { font-size: 2rem; font-weight: 600; margin-top: 0.5rem; }
        .metric-value.positive { color: #3fb950; }
        .metric-value.negative { color: #f85149; }
        .metric-value.neutral { color: #58a6ff; }
        .chart-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .chart-container { position: relative; height: 400px; }
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid #30363d;
        }
        .tab {
            padding: 0.75rem 1.5rem;
            background: none;
            border: none;
            color: #8b949e;
            cursor: pointer;
            font-size: 0.875rem;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }
        .tab.active {
            color: #58a6ff;
            border-bottom-color: #58a6ff;
        }
        .tab:hover { color: #e6edf3; }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }
        .trade-list { max-height: 400px; overflow-y: auto; }
        .trade-item {
            background: #21262d;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid #30363d;
        }
        .trade-item.correct { border-left-color: #3fb950; }
        .trade-item.incorrect { border-left-color: #f85149; }
        .trade-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
        .trade-time { font-size: 0.75rem; color: #8b949e; }
        .trade-window { font-size: 0.75rem; font-weight: 500; }
        .trade-details { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.875rem; }
        .trade-detail { display: flex; flex-direction: column; }
        .trade-detail-label { font-size: 0.625rem; color: #8b949e; text-transform: uppercase; }
        .trade-detail-value { font-weight: 500; }
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th, .summary-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }
        .summary-table th { color: #8b949e; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 BTC Predictor - Backtest Results</h1>
        <div class="header-actions">
            <button class="btn secondary" onclick="loadBacktest()">🔄 Refresh</button>
            <button class="btn" onclick="runNewBacktest()">▶️ Run New Backtest</button>
        </div>
    </div>
    <div class="container">
        <div id="metrics" class="metrics-grid">Loading...</div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('chart')">📊 Chart</button>
            <button class="tab" onclick="showTab('trades')">📋 Trades</button>
            <button class="tab" onclick="showTab('summary')">📈 Summary</button>
        </div>
        
        <div id="chart-panel" class="tab-panel active">
            <div class="chart-card">
                <h2>Walk-Forward Predictions vs Actual</h2>
                <div class="chart-container">
                    <canvas id="backtestChart"></canvas>
                </div>
            </div>
        </div>
        
        <div id="trades-panel" class="tab-panel">
            <div class="chart-card">
                <h2>Individual Trade Results</h2>
                <div class="trade-list" id="trade-list">Loading...</div>
            </div>
        </div>
        
        <div id="summary-panel" class="tab-panel">
            <div class="chart-card">
                <h2>Backtest Summary</h2>
                <table class="summary-table" id="summary-table">
                    <thead>
                        <tr><th>Metric</th><th>Value</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>
    </div>
    <script>
        let chart = null;
        let backtestData = null;
        
        async function loadBacktest() {
            try {
                const response = await fetch('/api/backtest');
                const data = await response.json();
                backtestData = data;
                renderAll(data);
            } catch (e) {
                console.error('Failed to load backtest:', e);
                document.getElementById('metrics').innerHTML = '<div style="color: #f85149;">Failed to load backtest data</div>';
            }
        }
        
        async function runNewBacktest() {
            const btn = event.target;
            btn.disabled = true;
            btn.textContent = '⏳ Running...';
            try {
                const response = await fetch('/api/backtest/run', { method: 'POST' });
                const data = await response.json();
                backtestData = data;
                renderAll(data);
            } catch (e) {
                console.error('Failed to run backtest:', e);
            } finally {
                btn.disabled = false;
                btn.textContent = '▶️ Run New Backtest';
            }
        }
        
        function renderAll(data) {
            renderMetrics(data.summary);
            renderChart(data.chart_data);
            renderTrades(data.results);
            renderSummary(data.summary);
        }
        
        function renderMetrics(summary) {
            const metrics = [
                { label: 'Direction Accuracy', value: (summary.direction_accuracy * 100).toFixed(1) + '%', class: 'positive' },
                { label: 'Mean MAE', value: '$' + summary.mean_mae.toLocaleString(undefined, {maximumFractionDigits: 0}), class: 'neutral' },
                { label: 'Mean MAPE', value: summary.mean_mape.toFixed(2) + '%', class: 'neutral' },
                { label: 'Interval Coverage', value: (summary.interval_coverage * 100).toFixed(1) + '%', class: 'positive' },
                { label: 'Sharpe Ratio', value: summary.sharpe_ratio.toFixed(2), class: summary.sharpe_ratio > 1 ? 'positive' : 'neutral' },
                { label: 'Max Drawdown', value: summary.max_drawdown.toFixed(2) + '%', class: 'negative' },
                { label: 'Total Return', value: summary.total_return.toFixed(2) + '%', class: summary.total_return > 0 ? 'positive' : 'negative' },
                { label: 'Win Rate', value: (summary.win_rate * 100).toFixed(1) + '%', class: summary.win_rate > 0.5 ? 'positive' : 'negative' },
            ];
            
            document.getElementById('metrics').innerHTML = metrics.map(m => `
                <div class="metric-card">
                    <div class="metric-label">${m.label}</div>
                    <div class="metric-value ${m.class}">${m.value}</div>
                </div>
            `).join('');
        }
        
        function renderChart(chartData) {
            const ctx = document.getElementById('backtestChart').getContext('2d');
            
            if (chart) {
                chart.data.labels = chartData.timestamps;
                chart.data.datasets[0].data = chartData.actual_prices;
                chart.data.datasets[1].data = chartData.predicted_prices;
                chart.data.datasets[2].data = chartData.upper_bounds;
                chart.data.datasets[3].data = chartData.lower_bounds;
                chart.update('none');
            } else {
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: chartData.timestamps,
                        datasets: [
                            {
                                label: 'Actual Price',
                                data: chartData.actual_prices,
                                borderColor: '#58a6ff',
                                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.2,
                                pointRadius: 0,
                            },
                            {
                                label: 'Predicted',
                                data: chartData.predicted_prices,
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
                                data: chartData.upper_bounds,
                                borderColor: 'rgba(88, 166, 255, 0.3)',
                                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                                borderWidth: 1,
                                fill: '-1',
                                pointRadius: 0,
                            },
                            {
                                label: 'Lower Bound',
                                data: chartData.lower_bounds,
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
                            x: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', maxTicksLimit: 20 } },
                            y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e', callback: v => '$' + v.toLocaleString() } }
                        }
                    }
                });
            }
        }
        
        function renderTrades(results) {
            const list = document.getElementById('trade-list');
            list.innerHTML = results.slice().reverse().map(r => {
                const correct = r.direction_correct;
                const change = r.change_pct_actual;
                return `
                    <div class="trade-item ${correct ? 'correct' : 'incorrect'}">
                        <div class="trade-header">
                            <span class="trade-time">${new Date(r.window_start).toLocaleString()}</span>
                            <span class="trade-window">${r.window_start} → ${r.window_end}</span>
                        </div>
                        <div class="trade-details">
                            <div class="trade-detail">
                                <span class="trade-detail-label">Actual</span>
                                <span class="trade-detail-value">$${r.actual_price.toLocaleString()}</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">Predicted</span>
                                <span class="trade-detail-value">$${r.predicted_price.toLocaleString()}</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">Actual Change</span>
                                <span class="trade-detail-value ${change >= 0 ? 'positive' : 'negative'}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">Pred Change</span>
                                <span class="trade-detail-value">${r.change_pct_pred >= 0 ? '+' : ''}${r.change_pct_pred.toFixed(2)}%</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">MAE</span>
                                <span class="trade-detail-value">$${r.mae.toLocaleString()}</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">MAPE</span>
                                <span class="trade-detail-value">${r.mape.toFixed(2)}%</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">In Interval</span>
                                <span class="trade-detail-value ${r.within_interval ? 'positive' : 'negative'}">${r.within_interval ? '✓' : '✗'}</span>
                            </div>
                            <div class="trade-detail">
                                <span class="trade-detail-label">Confidence</span>
                                <span class="trade-detail-value">${r.confidence}%</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function renderSummary(summary) {
            const rows = [
                ['Total Windows', summary.total_windows],
                ['Direction Accuracy', (summary.direction_accuracy * 100).toFixed(1) + '%'],
                ['Mean MAE', '$' + summary.mean_mae.toLocaleString(undefined, {maximumFractionDigits: 0})],
                ['Mean MAPE', summary.mean_mape.toFixed(2) + '%'],
                ['Interval Coverage', (summary.interval_coverage * 100).toFixed(1) + '%'],
                ['Sharpe Ratio', summary.sharpe_ratio.toFixed(2)],
                ['Max Drawdown', summary.max_drawdown.toFixed(2) + '%'],
                ['Total Return', summary.total_return.toFixed(2) + '%'],
                ['Win Rate', (summary.win_rate * 100).toFixed(1) + '%'],
                ['Profit Factor', summary.profit_factor === Infinity ? '∞' : summary.profit_factor.toFixed(2)],
            ];
            
            document.querySelector('#summary-table tbody').innerHTML = rows.map(([k, v]) => `
                <tr><td>${k}</td><td>${v}</td></tr>
            `).join('');
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-panel').classList.add('active');
        }
        
        // Load on startup
        loadBacktest();
    </script>
</body>
</html>
"""


def create_backtest_template():
    import os
    os.makedirs("templates", exist_ok=True)
    with open("templates/backtest.html", "w") as f:
        f.write(BACKTEST_HTML)


# Global backtest instance
_backtest_instance = None


@bt_app.on_event("startup")
async def startup():
    create_backtest_template()


@bt_app.get("/", response_class=HTMLResponse)
async def backtest_home(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})


@bt_app.get("/api/backtest")
async def get_backtest_results():
    """Get existing backtest results or run new one."""
    global _backtest_instance
    
    if _backtest_instance and _backtest_instance.results:
        return format_backtest_response(_backtest_instance)
    
    # Try to load from file
    try:
        with open("backtest_results.json", "r") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        # Run quick backtest
        return await run_backtest_async()


@bt_app.post("/api/backtest/run")
async def run_backtest():
    """Run a new backtest."""
    return await run_backtest_async()


async def run_backtest_async():
    """Run backtest asynchronously."""
    global _backtest_instance
    
    fetcher = AsyncDataFetcher()
    await fetcher.__aenter__()
    
    try:
        config = EnrichedConfig()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Shorter for demo
        
        backtest = WalkForwardBacktest(
            initial_train_days=14,
            test_window_days=3,
            step_days=1,
            purge_days=1,
        )
        
        results = await backtest.run_backtest(fetcher, config, start_date, end_date)
        summary = backtest.compute_summary()
        
        _backtest_instance = backtest
        
        return format_backtest_response(backtest)
    finally:
        await fetcher.__aexit__(None, None, None)


def format_backtest_response(backtest: WalkForwardBacktest) -> Dict:
    """Format backtest results for frontend."""
    results = backtest.results
    summary = backtest.compute_summary()
    
    # Prepare chart data
    timestamps = []
    actual_prices = []
    predicted_prices = []
    lower_bounds = []
    upper_bounds = []
    direction_correct = []
    confidence = []
    
    for r in results:
        timestamps.append(r.window_end)
        actual_prices.append(r.actual_price)
        predicted_prices.append(r.predicted_price)
        lower_bounds.append(r.ensemble_interval[0])
        upper_bounds.append(r.ensemble_interval[1])
        direction_correct.append(r.direction_correct)
        confidence.append(r.confidence)
    
    chart_data = BacktestChartData(
        timestamps=timestamps,
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        direction_correct=direction_correct,
        confidence=confidence,
    )
    
    return {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
        "chart_data": asdict(chart_data),
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    create_backtest_template()
    uvicorn.run("backtest_ui:bt_app", host="0.0.0.0", port=8003, reload=True)