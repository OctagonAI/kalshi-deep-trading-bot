"""
Configuration management for the simple trading bot.
"""
import os
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator
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
    
    @validator('private_key')
    def validate_private_key(cls, v):
        """Validate and format private key."""
        if not v or v == "your_kalshi_private_key_here":
            raise ValueError("KALSHI_PRIVATE_KEY is required. Please set it in your .env file.")
        
        # If it looks like a file path, try to read it
        if not v.startswith('-----BEGIN') and (Path(v).exists() or v.endswith('.pem')):
            try:
                with open(v, 'r') as f:
                    v = f.read()
            except Exception as e:
                raise ValueError(f"Could not read private key file '{v}': {e}")
        
        # Basic validation that it looks like a PEM key
        if not v.strip().startswith('-----BEGIN') or not v.strip().endswith('-----'):
            raise ValueError(
                "Private key must be in PEM format starting with '-----BEGIN' and ending with '-----'. "
                "Make sure to include \\n for line breaks in your .env file."
            )
        
        return v

class OctagonConfig(BaseModel):
    """Octagon Deep Research API configuration."""
    api_key: str = Field(..., description="Octagon API key")
    base_url: str = Field(default="https://api.octagon.ai", description="Octagon API base URL")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v == "your_octagon_api_key_here":
            raise ValueError("OCTAGON_API_KEY is required. Please set it in your .env file.")
        return v

class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="OpenAI model to use")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return v

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
    max_events_to_analyze: int = Field(default=50, description="Number of top events to analyze by volume_24h")
    research_batch_size: int = Field(default=10, description="Number of parallel deep research requests to batch")
    
    def __init__(self, **data):
        # Build nested configs from environment variables
        
        # Handle private key from file if specified
        private_key = os.getenv("KALSHI_PRIVATE_KEY", "")
        private_key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
        
        if private_key_file and not private_key:
            private_key = private_key_file  # Will be processed by validator
        
        kalshi_config = KalshiConfig(
            api_key=os.getenv("KALSHI_API_KEY", ""),
            private_key=private_key,
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
            "max_markets": int(os.getenv("MAX_MARKETS", "50")),
            "max_bet_amount": float(os.getenv("MAX_BET_AMOUNT", "100.0")),
            "max_events_to_analyze": int(os.getenv("MAX_EVENTS_TO_ANALYZE", "50")),
            "research_batch_size": int(os.getenv("RESEARCH_BATCH_SIZE", "10"))
        })
        
        super().__init__(**data)

def load_config() -> BotConfig:
    """Load and validate configuration."""
    return BotConfig() 