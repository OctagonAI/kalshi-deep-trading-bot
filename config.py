"""
Configuration management for the simple trading bot.
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KalshiConfig(BaseModel):
    """Kalshi API configuration."""
    api_key: str = Field(..., description="Kalshi API key")
    private_key: str = Field(..., description="Kalshi private key (PEM format)")
    use_demo: bool = Field(default=True, description="Use demo environment")
    
    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on environment."""
        if self.use_demo:
            return "https://demo-api.kalshi.co"
        return "https://api.kalshi.co"

class OctagonConfig(BaseModel):
    """Octagon Deep Research API configuration."""
    api_key: str = Field(..., description="Octagon API key")
    base_url: str = Field(default="https://api.octagon.ai", description="Octagon API base URL")

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="OpenAI model to use")

class BotConfig(BaseSettings):
    """Main bot configuration."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API configurations
    kalshi: KalshiConfig = Field(..., description="Kalshi configuration")
    octagon: OctagonConfig = Field(..., description="Octagon configuration")
    openai: OpenAIConfig = Field(..., description="OpenAI configuration")
    
    # Bot settings
    dry_run: bool = Field(default=True, description="Run in dry-run mode")
    max_markets: int = Field(default=50, description="Maximum number of events to process")
    max_bet_amount: float = Field(default=100.0, description="Maximum bet amount per market")
    
    def __init__(self, **data):
        # Build nested configs from environment variables
        kalshi_config = KalshiConfig(
            api_key=os.getenv("KALSHI_API_KEY", ""),
            private_key=os.getenv("KALSHI_PRIVATE_KEY", ""),
            use_demo=os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"
        )
        
        octagon_config = OctagonConfig(
            api_key=os.getenv("OCTAGON_API_KEY", ""),
            base_url=os.getenv("OCTAGON_BASE_URL", "https://api.octagon.ai")
        )
        
        openai_config = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o")
        )
        
        data.update({
            "kalshi": kalshi_config,
            "octagon": octagon_config,
            "openai": openai_config,
            "dry_run": os.getenv("DRY_RUN", "true").lower() == "true",
            "max_markets": int(os.getenv("MAX_MARKETS", "500")),
            "max_bet_amount": float(os.getenv("MAX_BET_AMOUNT", "100.0"))
        })
        
        super().__init__(**data)

def load_config() -> BotConfig:
    """Load and validate configuration."""
    return BotConfig() 