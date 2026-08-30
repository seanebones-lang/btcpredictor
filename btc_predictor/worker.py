"""
Background worker for scheduled predictions, retraining, and alerting.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
import logging

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .predictor_async import generate_async_prediction, get_fetcher, close_fetcher
from .data_sources import AsyncDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PREDICTION_INTERVAL_MINUTES = int(os.getenv("PREDICTION_INTERVAL_MINUTES", "15"))
RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))

class PredictionWorker:
    """Worker for generating scheduled predictions and sending alerts."""
    
    def __init__(self):
        self.running = False
        self.fetcher: Optional[AsyncDataFetcher] = None
    
    async def start(self):
        """Start the worker."""
        self.running = True
        self.fetcher = AsyncDataFetcher()
        await self.fetcher.__aenter__()
        logger.info("Worker started")
        
        # Start background tasks
        await asyncio.gather(
            self._prediction_loop(),
            self._retrain_loop(),
            self._health_check_loop(),
        )
    
    async def stop(self):
        """Stop the worker."""
        self.running = False
        if self.fetcher:
            await self.fetcher.__aexit__(None, None, None)
        logger.info("Worker stopped")
    
    async def _prediction_loop(self):
        """Generate predictions at regular intervals."""
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
                logger.info("Generating scheduled prediction...")
                prediction = await generate_async_prediction()
                logger.info(f"Prediction: {prediction[:100]}...")
                
                # Send to Telegram if configured
                await self._send_telegram(prediction)
                
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _retrain_loop(self):
        """Retrain models at regular intervals."""
        while self.running:
            try:
                await asyncio.sleep(RETRAIN_INTERVAL_HOURS * 3600)
                
                if not self.running:
                    break
                
                logger.info("Starting scheduled retraining...")
                # TODO: Implement actual retraining
                # For now, just log
                logger.info("Retraining complete (placeholder)")
                
            except Exception as e:
                logger.error(f"Retrain loop error: {e}")
                await asyncio.sleep(3600)  # Wait before retry
    
    async def _health_check_loop(self):
        """Periodic health checks."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                if not self.running:
                    break
                
                # Check API connectivity
                price = await self.fetcher.get_live_price()
                if price:
                    logger.debug(f"Health check OK - BTC: ${price:,.2f}")
                else:
                    logger.warning("Health check failed - could not fetch price")
                    
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _send_telegram(self, message: str):
        """Send message via Telegram Bot API."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(f"Telegram send failed: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Telegram error: {e}")


async def main():
    """Main entry point."""
    worker = PredictionWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
