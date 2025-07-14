"""
Simple Octagon Deep Research API client using OpenAI SDK.
"""
import openai
import json
from typing import Dict, Any, Optional, List
from loguru import logger
from config import OctagonConfig


class OctagonClient:
    """Simple client for Octagon Deep Research API using OpenAI SDK."""
    
    def __init__(self, config: OctagonConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=1800.0  # 30 minutes timeout for deep research
        )
    
    async def research_event(self, event: Dict[str, Any], markets: List[Dict[str, Any]]) -> str:
        """
        Research an event and its markets using Octagon Deep Research.
        
        Args:
            event: Event information (title, subtitle, category, etc.)
            markets: List of markets within the event (without odds)
            
        Returns:
            Research response as a string
        """
        try:
            # Format the event and markets for analysis
            event_info = f"""
            Event: {event.get('title', '')}
            Subtitle: {event.get('subtitle', '')}
            Category: {event.get('category', '')}
            Volume: {event.get('volume', 0)}
            Mutually Exclusive: {event.get('mutually_exclusive', False)}
            """
            
            markets_info = "Markets:\n"
            for i, market in enumerate(markets, 1):
                markets_info += f"{i}. {market.get('ticker', '')}: {market.get('title', '')}\n"
                if market.get('subtitle'):
                    markets_info += f"   Subtitle: {market.get('subtitle', '')}\n"
                markets_info += f"   Volume: {market.get('volume', 0)}\n"
                markets_info += f"   Open: {market.get('open_time', '')}\n"
                markets_info += f"   Close: {market.get('close_time', '')}\n\n"
            
            prompt = f"""
            You are a prediction market expert. Research this event and predict the probability for each market independently.
            
            {event_info}
            
            {markets_info}
            
            Please provide:
            1. Overall event analysis and key factors
            2. Current sentiment and news analysis
            3. For each market, predict the probability of YES outcome (0-100%)
            4. Confidence level for each prediction (1-10)
            5. Key risks and catalysts that could affect outcomes
            6. Trading recommendations for each market
            
            Focus on:
            - Independent analysis of each market's probability
            - Consider correlations between markets if mutually exclusive
            - Provide actionable insights for trading decisions
            - Be specific about probability estimates
            
            Format your response clearly with market tickers and probability predictions.
            """
            
            event_ticker = event.get('event_ticker', 'UNKNOWN')
            logger.info(f"Starting deep research for event {event_ticker} (this may take several minutes)...")
            
            response = await self.client.chat.completions.create(
                model="octagon-deep-research-agent",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            logger.info(f"Completed deep research for event {event_ticker}")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error researching event {event.get('event_ticker', '')}: {e}")
            return f"Error researching event: {str(e)}"
    
    async def close(self):
        """Close the client (OpenAI client doesn't need explicit closing)."""
        pass 