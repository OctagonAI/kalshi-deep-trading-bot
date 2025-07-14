"""
Simple Octagon Deep Research API client using OpenAI SDK.
"""
import openai
from typing import Dict, Any, Optional
from loguru import logger
from config import OctagonConfig


class OctagonClient:
    """Simple client for Octagon Deep Research API using OpenAI SDK."""
    
    def __init__(self, config: OctagonConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
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
            
            response = await self.client.chat.completions.create(
                model="octagon-deep-research-agent",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error researching market {market_ticker}: {e}")
            return f"Error researching market: {str(e)}"
    
    async def close(self):
        """Close the client (OpenAI client doesn't need explicit closing)."""
        pass 