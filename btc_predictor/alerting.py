"""
Alerting system for BTC Predictor - Telegram, Discord, Email notifications.
"""

import asyncio
import os
import smtplib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiohttp

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"


@dataclass
class AlertConfig:
    """Configuration for alerting channels."""
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = None
    price_change_threshold: float = 2.0
    confidence_threshold: int = 90
    regime_change_alert: bool = True
    model_drift_threshold: float = 0.1
    
    def __post_init__(self):
        if self.email_to is None:
            self.email_to = []


@dataclass
class Alert:
    """Alert message."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = None
    channel: AlertChannel = AlertChannel.TELEGRAM
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class AlertManager:
    """Manages alerts across multiple channels."""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def send_alert(self, alert: Alert):
        """Send alert to configured channels."""
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        tasks = []
        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            tasks.append(self._send_telegram(alert))
        if self.config.discord_webhook_url:
            tasks.append(self._send_discord(alert))
        if self.config.smtp_host and self.config.email_to:
            tasks.append(self._send_email(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_telegram(self, alert: Alert):
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return
        
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }.get(alert.level, "📢")
        
        text = f"{emoji} *{alert.title}*\n\n{alert.message}"
        
        if alert.data:
            text += "\n\n*Details:*\n"
            for k, v in alert.data.items():
                text += f"  • {k}: `{v}`\n"
        
        text += f"\n_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}_"
        
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.warning(f"Telegram alert failed: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    async def _send_discord(self, alert: Alert):
        if not self.config.discord_webhook_url:
            return
        
        color = {
            AlertLevel.INFO: 0x58a6ff,
            AlertLevel.WARNING: 0xffa500,
            AlertLevel.CRITICAL: 0xff0000,
        }.get(alert.level, 0x58a6ff)
        
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color,
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {"name": k, "value": str(v), "inline": True}
                for k, v in (alert.data or {}).items()
            ],
            "footer": {"text": "BTC Predictor Alerting"},
        }
        
        payload = {"embeds": [embed]}
        
        try:
            async with self._session.post(self.config.discord_webhook_url, json=payload) as resp:
                if resp.status not in [200, 204]:
                    logger.warning(f"Discord alert failed: {resp.status}")
        except Exception as e:
            logger.error(f"Discord error: {e}")
    
    async def _send_email(self, alert: Alert):
        if not self.config.smtp_host or not self.config.email_to:
            return
        
        subject = f"[{alert.level.value.upper()}] BTC Predictor: {alert.title}"
        
        body = f"""
{alert.message}

Details:
"""
        for k, v in (alert.data or {}).items():
            body += f"  {k}: {v}\n"
        
        body += f"\nTime: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        msg = MIMEMultipart()
        msg["From"] = self.config.email_from
        msg["To"] = ", ".join(self.config.email_to)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_smtp, msg)
        except Exception as e:
            logger.error(f"Email error: {e}")
    
    def _send_smtp(self, msg: MIMEMultipart):
        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            server.starttls()
            if self.config.smtp_username and self.config.smtp_password:
                server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
    
    def check_price_alert(
        self, 
        current_price: float, 
        previous_price: float,
        prediction: Dict[str, Any],
    ) -> Optional[Alert]:
        change_pct = abs((current_price - previous_price) / previous_price * 100)
        
        if change_pct >= self.config.price_change_threshold:
            direction = "📈 UP" if current_price > previous_price else "📉 DOWN"
            return Alert(
                level=AlertLevel.WARNING,
                title=f"Significant Price Movement: {direction} {change_pct:.2f}%",
                message=f"BTC moved {direction} {change_pct:.2f}% from ${previous_price:,.2f} to ${current_price:,.2f}",
                timestamp=datetime.utcnow(),
                data={
                    "current_price": f"${current_price:,.2f}",
                    "previous_price": f"${previous_price:,.2f}",
                    "change_pct": f"{change_pct:.2f}%",
                    "predicted_direction": prediction.get("direction", "N/A"),
                    "predicted_change": f"{prediction.get('change_pct', 0):.2f}%",
                },
            )
        return None
    
    def check_confidence_alert(self, prediction: Dict[str, Any]) -> Optional[Alert]:
        confidence = prediction.get("confidence", 0)
        
        if confidence >= self.config.confidence_threshold:
            return Alert(
                level=AlertLevel.INFO,
                title=f"High Confidence Prediction: {confidence}%",
                message=f"Model confidence is {confidence}% for {prediction.get('window_start', '')} → {prediction.get('window_end', '')}",
                timestamp=datetime.utcnow(),
                data={
                    "confidence": f"{confidence}%",
                    "direction": prediction.get("direction", "N/A"),
                    "predicted_price": f"${prediction.get('next_price', 0):,.2f}",
                },
            )
        elif confidence < 50:
            return Alert(
                level=AlertLevel.WARNING,
                title=f"Low Confidence Prediction: {confidence}%",
                message=f"Model confidence is only {confidence}% - prediction unreliable",
                timestamp=datetime.utcnow(),
                data={
                    "confidence": f"{confidence}%",
                    "direction": prediction.get("direction", "N/A"),
                },
            )
        return None
    
    def check_regime_alert(
        self, 
        current_regime: str, 
        previous_regime: str,
        confidence: float,
    ) -> Optional[Alert]:
        if not self.config.regime_change_alert or current_regime == previous_regime:
            return None
        
        return Alert(
            level=AlertLevel.WARNING,
            title=f"Regime Change Detected: {previous_regime} → {current_regime}",
            message=f"Market correlation regime shifted from {previous_regime} to {current_regime} (confidence: {confidence:.0%})",
            timestamp=datetime.utcnow(),
            data={
                "previous_regime": previous_regime,
                "current_regime": current_regime,
                "confidence": f"{confidence:.0%}",
            },
        )
    
    def check_model_drift_alert(
        self,
        recent_accuracy: float,
        baseline_accuracy: float,
    ) -> Optional[Alert]:
        drift = baseline_accuracy - recent_accuracy
        
        if drift >= self.config.model_drift_threshold:
            return Alert(
                level=AlertLevel.CRITICAL,
                title=f"Model Drift Detected: {drift:.1%} accuracy drop",
                message=f"Recent direction accuracy ({recent_accuracy:.1%}) dropped {drift:.1%} below baseline ({baseline_accuracy:.1%})",
                timestamp=datetime.utcnow(),
                data={
                    "recent_accuracy": f"{recent_accuracy:.1%}",
                    "baseline_accuracy": f"{baseline_accuracy:.1%}",
                    "drift": f"{drift:.1%}",
                },
            )
        return None
    
    def check_interval_breach(
        self,
        actual_price: float,
        prediction: Dict[str, Any],
    ) -> Optional[Alert]:
        interval = prediction.get("ensemble_interval", [])
        if len(interval) != 2:
            return None
        
        lower, upper = interval
        if actual_price < lower or actual_price > upper:
            breach_type = "below lower bound" if actual_price < lower else "above upper bound"
            return Alert(
                level=AlertLevel.WARNING,
                title=f"Prediction Interval Breached: {breach_type}",
                message=f"Actual price ${actual_price:,.2f} fell outside 90% interval [${lower:,.2f}, ${upper:,.2f}]",
                timestamp=datetime.utcnow(),
                data={
                    "actual_price": f"${actual_price:,.2f}",
                    "interval_lower": f"${lower:,.2f}",
                    "interval_upper": f"${upper:,.2f}",
                    "predicted_price": f"${prediction.get('next_price', 0):,.2f}",
                },
            )
        return None


class AlertScheduler:
    """Schedules and manages periodic alert checks."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.last_price: Optional[float] = None
        self.last_regime: Optional[str] = None
        self.baseline_accuracy: float = 0.75
    
    async def start(self, interval_seconds: int = 60):
        self.running = True
        self._task = asyncio.create_task(self._run_loop(interval_seconds))
    
    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _run_loop(self, interval_seconds: int):
        while self.running:
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert scheduler error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def check_prediction(self, prediction: Dict[str, Any], actual_price: float = None):
        alert = self.alert_manager.check_confidence_alert(prediction)
        if alert:
            await self.alert_manager.send_alert(alert)
        
        if actual_price is not None and self.last_price is not None:
            alert = self.alert_manager.check_price_alert(actual_price, self.last_price, prediction)
            if alert:
                await self.alert_manager.send_alert(alert)
        
        if actual_price is not None:
            alert = self.alert_manager.check_interval_breach(actual_price, prediction)
            if alert:
                await self.alert_manager.send_alert(alert)
        
        self.last_price = actual_price
    
    def update_regime(self, regime: str, confidence: float):
        if self.last_regime and regime != self.last_regime:
            alert = self.alert_manager.check_regime_alert(regime, self.last_regime, confidence)
            if alert:
                asyncio.create_task(self.alert_manager.send_alert(alert))
        self.last_regime = regime
    
    def update_accuracy(self, recent_accuracy: float):
        alert = self.alert_manager.check_model_drift_alert(recent_accuracy, self.baseline_accuracy)
        if alert:
            asyncio.create_task(self.alert_manager.send_alert(alert))


async def send_test_alert():
    config = AlertConfig(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
    )
    
    async with AlertManager(config) as manager:
        alert = Alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="BTC Predictor alerting system is configured and working!",
            timestamp=datetime.utcnow(),
            data={"test": "true"},
        )
        await manager.send_alert(alert)
        print("Test alert sent!")


if __name__ == "__main__":
    asyncio.run(send_test_alert())