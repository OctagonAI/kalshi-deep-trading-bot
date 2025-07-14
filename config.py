"""
Configuration management for the Kalshi Trading Bot
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv

load_dotenv()

class KalshiConfig(BaseModel):
    """Configuration for Kalshi API"""
    api_key: str = Field(..., description="Kalshi API key")
    api_secret: str = Field(..., description="Kalshi API secret")
    base_url: str = Field(default="https://api.kalshi.com", description="Base API URL")
    websocket_url: str = Field(default="wss://api.kalshi.com/ws/v1", description="WebSocket URL")
    rate_limit_requests_per_second: int = Field(default=5, description="Rate limit for API requests")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("Kalshi API key is required")
        return v
    
    @validator('api_secret')
    def validate_api_secret(cls, v):
        if not v:
            raise ValueError("Kalshi API secret is required")
        return v

class OctagonConfig(BaseModel):
    """Configuration for Octagon Deep Research API"""
    api_key: str = Field(..., description="Octagon Deep Research API key")
    base_url: str = Field(default="https://api.octagon.ai", description="Base API URL")
    rate_limit_requests_per_day: int = Field(default=5000, description="Daily rate limit")
    concurrent_streams: int = Field(default=10, description="Max concurrent streams")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("Octagon API key is required")
        return v

class RiskManagementConfig(BaseModel):
    """Risk management configuration"""
    max_position_size: float = Field(default=0.1, description="Maximum position size as % of portfolio")
    max_daily_loss: float = Field(default=0.05, description="Maximum daily loss as % of portfolio")
    max_portfolio_risk: float = Field(default=0.2, description="Maximum portfolio risk exposure")
    stop_loss_percent: float = Field(default=0.15, description="Stop loss percentage")
    take_profit_percent: float = Field(default=0.25, description="Take profit percentage")
    min_probability_threshold: float = Field(default=0.55, description="Minimum probability to enter trade")
    max_concurrent_positions: int = Field(default=5, description="Maximum concurrent positions")

class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    enabled_strategies: List[str] = Field(default=[
        "sentiment_momentum",
        "event_arbitrage", 
        "research_based_trades",
        "political_polling"
    ], description="Enabled trading strategies")
    
    market_categories: List[str] = Field(default=[
        "politics",
        "economics", 
        "technology",
        "sports",
        "entertainment"
    ], description="Market categories to trade")
    
    min_volume_threshold: int = Field(default=1000, description="Minimum volume for market entry")
    max_spread: float = Field(default=0.05, description="Maximum bid-ask spread to trade")
    research_refresh_minutes: int = Field(default=30, description="Minutes between research updates")

class LoggingConfig(BaseModel):
    """Logging configuration"""
    log_level: str = Field(default="INFO", description="Log level")
    log_file: str = Field(default="trading_bot.log", description="Log file path")
    max_file_size: str = Field(default="10 MB", description="Maximum log file size")
    retention_days: int = Field(default=30, description="Log retention in days")

class DatabaseConfig(BaseModel):
    """Database configuration for storing trades and research"""
    db_path: str = Field(default="trading_bot.db", description="SQLite database path")
    backup_interval_hours: int = Field(default=24, description="Backup interval in hours")

class BotConfig(BaseModel):
    """Main bot configuration"""
    kalshi: KalshiConfig
    octagon: OctagonConfig
    risk_management: RiskManagementConfig
    trading: TradingConfig
    logging: LoggingConfig
    database: DatabaseConfig
    
    # General bot settings
    dry_run: bool = Field(default=True, description="Run in dry-run mode (no real trades)")
    loop_interval_seconds: int = Field(default=60, description="Main loop interval in seconds")
    startup_delay_seconds: int = Field(default=30, description="Startup delay in seconds")

def load_config() -> BotConfig:
    """Load configuration from environment variables"""
    return BotConfig(
        kalshi=KalshiConfig(
            api_key=os.getenv("KALSHI_API_KEY", ""),
            api_secret=os.getenv("KALSHI_API_SECRET", ""),
            base_url=os.getenv("KALSHI_BASE_URL", "https://api.kalshi.com"),
            websocket_url=os.getenv("KALSHI_WS_URL", "wss://api.kalshi.com/ws/v1"),
            rate_limit_requests_per_second=int(os.getenv("KALSHI_RATE_LIMIT", "5"))
        ),
        octagon=OctagonConfig(
            api_key=os.getenv("OCTAGON_API_KEY", ""),
            base_url=os.getenv("OCTAGON_BASE_URL", "https://api.octagon.ai"),
            rate_limit_requests_per_day=int(os.getenv("OCTAGON_RATE_LIMIT", "5000")),
            concurrent_streams=int(os.getenv("OCTAGON_CONCURRENT_STREAMS", "10"))
        ),
        risk_management=RiskManagementConfig(
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", "0.2")),
            stop_loss_percent=float(os.getenv("STOP_LOSS_PERCENT", "0.15")),
            take_profit_percent=float(os.getenv("TAKE_PROFIT_PERCENT", "0.25")),
            min_probability_threshold=float(os.getenv("MIN_PROBABILITY_THRESHOLD", "0.55")),
            max_concurrent_positions=int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))
        ),
        trading=TradingConfig(
            enabled_strategies=os.getenv("ENABLED_STRATEGIES", "sentiment_momentum,event_arbitrage,research_based_trades,political_polling").split(","),
            market_categories=os.getenv("MARKET_CATEGORIES", "politics,economics,technology,sports,entertainment").split(","),
            min_volume_threshold=int(os.getenv("MIN_VOLUME_THRESHOLD", "1000")),
            max_spread=float(os.getenv("MAX_SPREAD", "0.05")),
            research_refresh_minutes=int(os.getenv("RESEARCH_REFRESH_MINUTES", "30"))
        ),
        logging=LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "trading_bot.log"),
            max_file_size=os.getenv("MAX_FILE_SIZE", "10 MB"),
            retention_days=int(os.getenv("RETENTION_DAYS", "30"))
        ),
        database=DatabaseConfig(
            db_path=os.getenv("DB_PATH", "trading_bot.db"),
            backup_interval_hours=int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
        ),
        dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
        loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
        startup_delay_seconds=int(os.getenv("STARTUP_DELAY_SECONDS", "30"))
    ) 