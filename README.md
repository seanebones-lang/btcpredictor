# BTC Predictor

**Production-oriented Bitcoin 15-minute direction and price forecasts.**

Ensemble of bidirectional LSTM + Prophet with technical indicators, short-term momentum, and optional derivatives features. Ships as a FastAPI service with Redis caching, scheduled workers, Prometheus/Grafana, and Docker Compose.

Part of the [NextEleven](https://github.com/seanebones-lang) AI systems portfolio.

> **Not financial advice.** This is research and engineering software. Markets are noisy. Past fit does not imply future edge. Do not trade real capital on these outputs without your own risk process.

---

## What it does

On each request (or every 15 minutes from the worker), the system:

1. Pulls live BTC/USD from CoinGecko (spot) and optional Binance futures fields (funding, open interest).
2. Resamples to 15-minute bars and adds indicators (RSI, MACD, EMAs, Bollinger, ATR, OBV, VWAP, multi-horizon momentum).
3. Fits an LSTM ensemble and a Prophet model with indicator regressors.
4. Combines them (default ~60/40 LSTM/Prophet) and attaches conformal-style intervals plus a volatility-aware confidence score.
5. Returns the next 15-minute window: current price, target, % change, direction, and intervals.

Example CLI-style output:

```text
**BTC 15-min Prediction** (14:15 → 14:30)

Current: $67,420.10
Expected move: UP 0.18%

Next target: $67,541.33
Confidence: 81%

Model: Production Deep LSTM + Prophet + short-term momentum
```

JSON from `GET /predict` includes `window_start`, `window_end`, `current_price`, `next_price`, `change_pct`, `direction`, `confidence`, `ensemble_interval`, `prophet_interval`, `lstm_std`, and `enriched_features`.

---

## Architecture

```text
CoinGecko / Binance  →  Async fetcher + Redis cache
                              ↓
                     Feature pipeline (ta + macro)
                              ↓
              LSTM ensemble  +  Prophet  +  intervals
                              ↓
              FastAPI  /predict  /predict/text  /health  /metrics
                              ↓
         Worker (15m signals)  |  Retrain job (champion/challenger)
                              ↓
                   Prometheus  →  Grafana
```

| Layer | Location |
| --- | --- |
| Core models + features | `btc_predictor/` |
| REST API | `btc_predictor/api.py` |
| Async prediction path | `btc_predictor/predictor_async.py` |
| Transformer experiment | `btc_predictor/transformer_model.py` |
| Walk-forward backtest | `btc_predictor/backtest.py` |
| Retrain / registry | `btc_predictor/retraining_pipeline.py` |
| Scheduled worker + Telegram | `btc_predictor/worker.py` |
| WebSocket stream | `btc_predictor/websocket_server.py` |
| Config | `conf/config.yaml` |
| Observability | `prometheus.yml`, `grafana/` |
| Legacy Berkeley notebooks | `predictor/` |
| Historical collector | `heroku-script/` |

Version: **2.0.0** (`btc_predictor.__version__`).

---

## Quick start

### Local (Python 3.11+)

```bash
git clone https://github.com/seanebones-lang/btcpredictor.git
cd btcpredictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

One-shot prediction:

```bash
python -m btc_predictor.predictor
# or
python btc_alert.py
```

API (development — auth is skipped when `ENVIRONMENT=development`):

```bash
export ENVIRONMENT=development
uvicorn btc_predictor.api:app --reload --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/predict
curl http://localhost:8000/predict/text
curl http://localhost:8000/metrics
```

Interactive docs: http://localhost:8000/docs

### Docker Compose (API + Redis + worker + Prometheus + Grafana)

```bash
cp .env.example .env   # if you add one; otherwise export vars
export API_KEY=change-me-in-production
export GRAFANA_PASSWORD=admin
docker compose up --build
```

| Service | Port |
| --- | --- |
| API | 8000 |
| Grafana | 3000 (admin / `$GRAFANA_PASSWORD`) |
| Prometheus | 9090 |
| Redis | 6379 |

Worker command inside Compose: `python -m btc_predictor.worker`.

Production API expects header `X-API-Key` when `ENVIRONMENT` is not `development`.

---

## Configuration

Primary file: [`conf/config.yaml`](conf/config.yaml).

Highlights:

- **Data:** CoinGecko + Binance rate limits, Redis TTLs
- **LSTM:** 96-step lookback (24h of 15m bars), stacked units `[128, 64, 32]`, ensemble of 3, early stopping
- **Prophet:** daily/weekly seasonality, 90% intervals
- **Ensemble:** LSTM 0.6 / Prophet 0.4, conformal alpha 0.10
- **Features:** SMA/EMA/RSI/MACD/BB/ATR/OBV/VWAP + funding, OI, liquidations
- **API:** host/port, `60/minute` rate limit, API key from `API_KEY`
- **Worker:** 15-minute prediction cadence, 24h retrain, Telegram env vars
- **Retrain:** daily 02:00 UTC, champion metric `direction_accuracy`, +2% challenger gate
- **Backtest:** 30d train / 7d test, 1d purge, 5 splits

Environment variables used in Docker/API:

| Variable | Purpose |
| --- | --- |
| `ENVIRONMENT` | `development` / `production` / `test` |
| `API_KEY` | Required in production |
| `REDIS_URL` | Default `redis://localhost:6379/0` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Worker alerts |
| `GRAFANA_PASSWORD` | Grafana admin |
| `RATE_LIMIT` | e.g. `60/minute` |

---

## API

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/health` | no | Liveness + service map |
| GET | `/predict` | yes (prod) | Structured 15-minute forecast |
| GET | `/predict/text` | yes (prod) | Human-readable forecast |
| GET | `/metrics` | no | Prometheus scrape |

Rate limit default: 60 requests / minute per client IP.

---

## Models and features

**LSTM path** — bidirectional + stacked LSTM on scaled close (and, in the async/enriched path, multi-feature sequences). Ensemble of independently initialized networks; interval from cross-model std.

**Prophet path** — close as `y`, technicals and momentum as extra regressors, one-step 15-minute horizon.

**Blend** — weighted average; direction is sign of predicted % change vs live spot.

**Confidence** — compressed from recent realized volatility (clamped band). Treat it as a heuristic, not a calibrated probability.

**Transformer** — experimental module in `transformer_model.py` for sequence modeling research; not the default serving path.

This repo also keeps the original course models under `predictor/models/`:

- `baseline.py` — price-only LSTM
- `indicator.py` — OHLCV technicals
- `oracle.py` — technicals + Twitter sentiment (legacy; Twitter APIs have changed)

Notebooks in `predictor/` (`BaselineDemo`, `IndicatorDemo`, `OracleDemo`, `Dashboard`, `RealTimeTradingSignalsDemo`) are the historical walkthroughs.

---

## Backtesting and retraining

Walk-forward config lives under `backtest:` in `conf/config.yaml`. Results schema is in `backtest_results.json` (`direction_accuracy`, MAE/MAPE, interval coverage, Sharpe, max drawdown, win rate, profit factor).

The committed `backtest_results.json` is an empty template until you run `btc_predictor.backtest` against live or archived data. Publish numbers only after a real walk-forward with purge/embargo — do not invent a track record.

Retraining (`btc_predictor.retraining_pipeline`) compares a challenger to the registered champion on `direction_accuracy` and promotes only if the lift clears the configured threshold (default 2%).

---

## Tests and CI

```bash
pytest tests/ -v --cov=btc_predictor
```

GitHub Actions (`.github/workflows/ci.yml`) on `main` / `develop`:

- Ruff + Black + MyPy
- Unit tests with Redis service + coverage upload
- Integration job against a live uvicorn instance
- Trivy + Bandit + detect-secrets
- Docker build/push on `main` (requires Docker Hub secrets)

Python target: **3.11**.

---

## Project layout

```text
btcpredictor/
├── btc_predictor/          # v2 production package
├── conf/config.yaml        # Hydra-style / YAML config
├── tests/
├── grafana/  prometheus.yml
├── Dockerfile  docker-compose.yml
├── predictor/              # original Data-X notebooks + models
├── heroku-script/          # 24/7 collector (legacy)
├── btc_alert.py            # standalone one-shot script
├── requirements.txt
└── LICENSE                 # MIT
```

---

## Lineage

v2 is a modernization of the UC Berkeley Data-X / Anchain.ai **BTC Predictor** course project (LSTM dashboard, Firebase collector, sentiment oracle). That work used older Twitter tooling and Heroku-era collection. This fork keeps the notebooks for history and replaces the serving path with current TensorFlow, Prophet, FastAPI, Redis, and containerized ops.

Original public tree for reference: [Bitcoin-Price-Prediction/btcpredictor](https://github.com/Bitcoin-Price-Prediction/btcpredictor).

---

## Disclaimer and limits

- Forecasts are **short-horizon research signals**, not execution instructions.
- CoinGecko hourly series is interpolated to 15m in the simple path — that is not true exchange 15m OHLCV. Prefer Binance klines when you need bar-accurate features.
- Confidence scores are **not** calibrated probabilities.
- Empty or zeroed backtest summaries mean "not yet measured," not "zero risk."
- You are responsible for API keys, rate limits, and any live trading wrapper you attach.

---

## License

MIT © 2026 Sean McDonnell (NextEleven).

---

## Related

Built and maintained by **Sean McDonnell**, CTO, NextEleven LLC — AI systems for markets, inventory, and operations.
