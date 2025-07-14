"""
Simple Octagon Deep Research API client.
"""
import httpx
from typing import Dict, Any, Optional
from loguru import logger
from config import OctagonConfig


class OctagonClient:
    """Simple client for Octagon Deep Research API."""
    
    def __init__(self, config: OctagonConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            },
            timeout=120.0
        )
    
    async def research_market(self, market_title: str, market_ticker: str) -> str:
        """
        Research a single market using Octagon Deep Research.
        
        Args:
            market_title: The title/question of the market
            market_ticker: The ticker symbol of the market
            
        Returns:
            Research response as a string
        """
        try:
            prompt = f"""
            Research this prediction market for trading insights:
            
            Market: {market_title}
            Ticker: {market_ticker}
            
            Please provide:
            1. Current sentiment and news analysis
            2. Key factors that could influence the outcome
            3. Probability assessment and reasoning
            4. Trading recommendation (buy YES, buy NO, or skip)
            5. Risk factors to consider
            
            Focus on actionable insights for trading decisions.
            """
            
            response = await self.client.post(
                "/v1/responses",
                json={
                    "model": "octagon-deep-research-agent",
                    "prompt": prompt
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except Exception as e:
            logger.error(f"Error researching market {market_ticker}: {e}")
            return f"Error researching market: {str(e)}"
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 