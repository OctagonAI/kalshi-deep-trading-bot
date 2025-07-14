"""
Octagon Deep Research API Client for investment research and trading insights
Uses the actual Octagon Deep Research Agent API
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import re

class ResearchType(Enum):
    SENTIMENT = "sentiment"
    MARKET_FORECAST = "market_forecast"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    EVENT_ANALYSIS = "event_analysis"
    REGULATORY_ANALYSIS = "regulatory_analysis"

@dataclass
class SentimentAnalysis:
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    sources: List[str]
    key_themes: List[str]
    market_impact: str
    timestamp: datetime
    
@dataclass
class MarketForecast:
    probability: float
    confidence: float
    timeframe: str
    key_factors: List[str]
    risk_factors: List[str]
    supporting_data: Dict[str, Any]
    
@dataclass
class TradingSignal:
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float  # 0 to 1
    probability: float  # 0 to 1
    timeframe: str
    reasoning: str
    risk_level: str
    expected_return: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class ResearchInsight:
    market_id: str
    title: str
    insight_type: ResearchType
    content: str
    confidence: float
    relevance: float
    timestamp: datetime
    sources: List[str]
    key_points: List[str]
    trading_implications: List[str]
    risk_assessment: str
    
@dataclass
class MarketResearch:
    market_id: str
    market_title: str
    sentiment: Optional[SentimentAnalysis] = None
    forecast: Optional[MarketForecast] = None
    signals: List[TradingSignal] = field(default_factory=list)
    insights: List[ResearchInsight] = field(default_factory=list)
    overall_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class OctagonResearchClient:
    """Octagon Deep Research API client for investment research"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.octagon.ai", 
                 rate_limit_per_day: int = 5000, concurrent_streams: int = 10):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.rate_limit_per_day = rate_limit_per_day
        self.concurrent_streams = concurrent_streams
        self.session: Optional[httpx.AsyncClient] = None
        
        # Rate limiting tracking
        self.daily_requests = 0
        self.last_reset = datetime.now().date()
        self.request_times = []
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),  # Longer timeout for research queries
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    def _reset_daily_counter(self):
        """Reset daily request counter if needed"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_requests = 0
            self.last_reset = today
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        self._reset_daily_counter()
        
        if self.daily_requests >= self.rate_limit_per_day:
            logger.warning("Daily rate limit reached")
            # Wait until next day
            tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            wait_seconds = (tomorrow - datetime.now()).total_seconds()
            await asyncio.sleep(wait_seconds)
            self._reset_daily_counter()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_research_request(self, prompt: str) -> str:
        """Make research request to Octagon Deep Research Agent API"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        await self._check_rate_limit()
        
        # Use the actual Octagon Deep Research Agent API endpoint
        url = f"{self.base_url}/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "KalshiTradingBot/1.0"
        }
        
        data = {
            "model": "octagon-deep-research-agent",
            "prompt": prompt,
            "max_tokens": 4000,
            "temperature": 0.1
        }
        
        logger.debug(f"Making research request to {url}")
        
        try:
            response = await self.session.post(
                url=url,
                headers=headers,
                json=data
            )
            
            response.raise_for_status()
            self.daily_requests += 1
            
            result = response.json()
            
            # Extract the research content from the response
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            elif "response" in result:
                return result["response"]
            else:
                return str(result)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            if e.response.status_code == 429:
                await asyncio.sleep(300)  # Wait 5 minutes for rate limit
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def analyze_market_sentiment(self, market_title: str, context: str = "") -> SentimentAnalysis:
        """Analyze sentiment for a specific market"""
        prompt = f"""
        Analyze the sentiment and market implications for the prediction market: {market_title}
        
        {f"Additional context: {context}" if context else ""}
        
        Please provide a comprehensive sentiment analysis including:
        1. Overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
        2. Confidence level (0 to 1)
        3. Key themes and sentiment drivers
        4. Market impact assessment
        5. Relevant news sources and social media sentiment
        
        Format your response with clear sections and quantitative scores where possible.
        """
        
        response_text = await self._make_research_request(prompt)
        
        # Parse the response to extract sentiment information
        sentiment_data = self._parse_sentiment_response(response_text)
        
        return SentimentAnalysis(
            score=sentiment_data.get('score', 0.0),
            confidence=sentiment_data.get('confidence', 0.0),
            sources=sentiment_data.get('sources', []),
            key_themes=sentiment_data.get('key_themes', []),
            market_impact=sentiment_data.get('market_impact', 'neutral'),
            timestamp=datetime.now()
        )
    
    async def get_market_forecast(self, market_title: str, timeframe: str = "1d") -> MarketForecast:
        """Get market forecast using AI models"""
        prompt = f"""
        Provide a detailed forecast for the prediction market: {market_title}
        
        Time horizon: {timeframe}
        
        Please provide a comprehensive market forecast including:
        1. Probability of the predicted outcome (0 to 1)
        2. Confidence level in the forecast (0 to 1)
        3. Key factors supporting this probability
        4. Risk factors that could change the outcome
        5. Supporting data and evidence
        6. Recent developments or catalysts
        
        Consider all relevant factors including:
        - Historical data and trends
        - Current market conditions
        - Political, economic, or social factors
        - Expert opinions and analysis
        - Recent news and events
        
        Format your response with clear sections and quantitative assessments.
        """
        
        response_text = await self._make_research_request(prompt)
        
        # Parse the response to extract forecast information
        forecast_data = self._parse_forecast_response(response_text)
        
        return MarketForecast(
            probability=forecast_data.get('probability', 0.5),
            confidence=forecast_data.get('confidence', 0.0),
            timeframe=timeframe,
            key_factors=forecast_data.get('key_factors', []),
            risk_factors=forecast_data.get('risk_factors', []),
            supporting_data=forecast_data.get('supporting_data', {})
        )
    
    async def analyze_trading_opportunity(self, market_title: str, current_price: float, 
                                        market_context: str = "") -> TradingSignal:
        """Analyze a trading opportunity and generate trading signal"""
        prompt = f"""
        Analyze this prediction market for trading opportunities:
        
        Market: {market_title}
        Current Price: ${current_price}
        Context: {market_context}
        
        Please provide a comprehensive trading analysis including:
        1. Trading signal (BUY/SELL/HOLD)
        2. Signal strength (0 to 1)
        3. Probability of success (0 to 1)
        4. Risk assessment (low/medium/high)
        5. Expected return potential
        6. Recommended timeframe
        7. Stop loss level
        8. Take profit level
        9. Detailed reasoning for the recommendation
        
        Consider factors such as:
        - Current market pricing vs fair value
        - Market sentiment and momentum
        - Upcoming catalysts or events
        - Historical performance patterns
        - Risk-reward ratio
        - Liquidity and volume
        
        Format your response with clear sections and quantitative assessments.
        """
        
        response_text = await self._make_research_request(prompt)
        
        # Parse the response to extract trading signal information
        signal_data = self._parse_trading_signal_response(response_text)
        
        return TradingSignal(
            signal_type=signal_data.get('signal', 'HOLD'),
            strength=signal_data.get('strength', 0.5),
            probability=signal_data.get('probability', 0.5),
            timeframe=signal_data.get('timeframe', '1d'),
            reasoning=signal_data.get('reasoning', ''),
            risk_level=signal_data.get('risk_level', 'medium'),
            expected_return=signal_data.get('expected_return'),
            stop_loss=signal_data.get('stop_loss'),
            take_profit=signal_data.get('take_profit')
        )
    
    async def research_event_impact(self, event_description: str, related_markets: List[str]) -> List[ResearchInsight]:
        """Research the impact of an event on related markets"""
        query = f"""
        Research the impact of this event on prediction markets:
        
        Event: {event_description}
        Related Markets: {', '.join(related_markets)}
        
        Analyze:
        1. Direct impact on each market
        2. Probability changes
        3. Timeline of effects
        4. Trading implications
        5. Risk factors
        """
        
        response = await self._make_request(
            "/v1/research/event_impact",
            data={
                "query": query,
                "markets": related_markets,
                "depth": "comprehensive"
            },
            method="POST"
        )
        
        insights = []
        for insight_data in response.get('insights', []):
            insights.append(ResearchInsight(
                market_id=insight_data.get('market_id', ''),
                title=insight_data.get('title', ''),
                insight_type=ResearchType.EVENT_ANALYSIS,
                content=insight_data.get('content', ''),
                confidence=insight_data.get('confidence', 0.0),
                relevance=insight_data.get('relevance', 0.0),
                timestamp=datetime.now(),
                sources=insight_data.get('sources', []),
                key_points=insight_data.get('key_points', []),
                trading_implications=insight_data.get('trading_implications', []),
                risk_assessment=insight_data.get('risk_assessment', '')
            ))
        
        return insights
    
    async def analyze_portfolio_optimization(self, current_positions: List[Dict], 
                                           available_markets: List[Dict]) -> Dict[str, Any]:
        """Optimize portfolio allocation across prediction markets"""
        query = f"""
        Optimize portfolio allocation for prediction markets:
        
        Current Positions: {json.dumps(current_positions)}
        Available Markets: {json.dumps(available_markets)}
        
        Provide:
        1. Optimal allocation percentages
        2. Risk-adjusted returns
        3. Diversification recommendations
        4. Risk management suggestions
        """
        
        response = await self._make_request(
            "/v1/portfolio/optimize",
            data={
                "query": query,
                "current_positions": current_positions,
                "available_markets": available_markets,
                "optimization_objective": "max_return_risk_ratio"
            },
            method="POST"
        )
        
        return response
    
    async def get_regulatory_analysis(self, market_category: str) -> ResearchInsight:
        """Get regulatory analysis for a market category"""
        query = f"""
        Analyze regulatory factors and compliance considerations for {market_category} prediction markets:
        
        1. Current regulatory environment
        2. Upcoming regulatory changes
        3. Compliance requirements
        4. Risk factors
        5. Impact on market viability
        """
        
        response = await self._make_request(
            "/v1/regulatory/analysis",
            data={
                "query": query,
                "category": market_category
            },
            method="POST"
        )
        
        return ResearchInsight(
            market_id=f"{market_category}_regulatory",
            title=f"Regulatory Analysis: {market_category}",
            insight_type=ResearchType.REGULATORY_ANALYSIS,
            content=response.get('content', ''),
            confidence=response.get('confidence', 0.0),
            relevance=response.get('relevance', 0.0),
            timestamp=datetime.now(),
            sources=response.get('sources', []),
            key_points=response.get('key_points', []),
            trading_implications=response.get('trading_implications', []),
            risk_assessment=response.get('risk_assessment', '')
        )
    
    async def comprehensive_market_research(self, market_id: str, market_title: str, 
                                          current_price: float, context: str = "") -> MarketResearch:
        """Perform comprehensive research on a market"""
        logger.info(f"Starting comprehensive research for market: {market_title}")
        
        # Gather all research components in parallel
        research_tasks = [
            self.analyze_market_sentiment(market_title, context),
            self.get_market_forecast(market_title),
            self.analyze_trading_opportunity(market_title, current_price, context)
        ]
        
        try:
            sentiment, forecast, signal = await asyncio.gather(*research_tasks)
            
            # Calculate overall score based on all factors
            overall_score = self._calculate_overall_score(sentiment, forecast, signal)
            
            return MarketResearch(
                market_id=market_id,
                market_title=market_title,
                sentiment=sentiment,
                forecast=forecast,
                signals=[signal],
                overall_score=overall_score,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive research: {e}")
            # Return basic research object with error info
            return MarketResearch(
                market_id=market_id,
                market_title=market_title,
                overall_score=0.0,
                last_updated=datetime.now()
            )
    
    def _calculate_overall_score(self, sentiment: SentimentAnalysis, 
                               forecast: MarketForecast, signal: TradingSignal) -> float:
        """Calculate overall market score from research components"""
        # Weighted combination of different factors
        sentiment_weight = 0.3
        forecast_weight = 0.4
        signal_weight = 0.3
        
        # Normalize sentiment score from [-1, 1] to [0, 1]
        sentiment_normalized = (sentiment.score + 1) / 2
        
        # Weight by confidence
        sentiment_score = sentiment_normalized * sentiment.confidence * sentiment_weight
        forecast_score = forecast.probability * forecast.confidence * forecast_weight
        signal_score = signal.strength * signal.probability * signal_weight
        
        overall_score = sentiment_score + forecast_score + signal_score
        
        return min(max(overall_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def get_market_correlations(self, market_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation analysis between markets"""
        query = f"""
        Analyze correlations between these prediction markets:
        Markets: {', '.join(market_ids)}
        
        Provide correlation coefficients and explanations for relationships.
        """
        
        response = await self._make_request(
            "/v1/analysis/correlations",
            data={
                "query": query,
                "markets": market_ids
            },
            method="POST"
        )
        
        return response.get('correlations', {})
    
    async def stream_market_updates(self, market_ids: List[str], callback):
        """Stream real-time research updates for markets"""
        # This would implement WebSocket streaming for real-time updates
        # For now, we'll simulate with periodic updates
        logger.info(f"Starting research stream for {len(market_ids)} markets")
        
        while True:
            try:
                for market_id in market_ids:
                    # Simulate research update
                    update = {
                        "market_id": market_id,
                        "timestamp": datetime.now().isoformat(),
                        "update_type": "research_refresh",
                        "data": {"status": "monitoring"}
                    }
                    await callback(update)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in research stream: {e}")
                await asyncio.sleep(5)
    
    def is_research_actionable(self, research: MarketResearch, min_confidence: float = 0.6) -> bool:
        """Check if research meets actionable criteria"""
        if research.overall_score < min_confidence:
            return False
        
        if research.sentiment and research.sentiment.confidence < min_confidence:
            return False
        
        if research.forecast and research.forecast.confidence < min_confidence:
            return False
        
        return True
    
    def get_research_summary(self, research: MarketResearch) -> str:
        """Get a human-readable summary of research"""
        summary = f"Market Research Summary for {research.market_title}\n"
        summary += f"Overall Score: {research.overall_score:.2f}\n"
        
        if research.sentiment:
            summary += f"Sentiment: {research.sentiment.score:.2f} (confidence: {research.sentiment.confidence:.2f})\n"
        
        if research.forecast:
            summary += f"Forecast Probability: {research.forecast.probability:.2f}\n"
        
        if research.signals:
            signal = research.signals[0]
            summary += f"Trading Signal: {signal.signal_type} (strength: {signal.strength:.2f})\n"
        
        return summary
    
    def _parse_sentiment_response(self, response_text: str) -> Dict[str, Any]:
        """Parse sentiment analysis response from text"""
        try:
            # Extract sentiment score
            score_match = re.search(r'sentiment.*score.*?(-?\d+\.?\d*)', response_text, re.IGNORECASE)
            score = float(score_match.group(1)) if score_match else 0.0
            
            # Extract confidence
            confidence_match = re.search(r'confidence.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            # Extract key themes
            themes = []
            themes_section = re.search(r'themes?\s*:?\s*(.*?)(?:\n\n|\n[A-Z]|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if themes_section:
                theme_text = themes_section.group(1)
                themes = [theme.strip('- ') for theme in theme_text.split('\n') if theme.strip()]
            
            # Extract market impact
            impact_match = re.search(r'market.*impact.*?:?\s*(\w+)', response_text, re.IGNORECASE)
            market_impact = impact_match.group(1).lower() if impact_match else 'neutral'
            
            # Extract sources
            sources = []
            source_section = re.search(r'sources?\s*:?\s*(.*?)(?:\n\n|\n[A-Z]|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if source_section:
                source_text = source_section.group(1)
                sources = [source.strip('- ') for source in source_text.split('\n') if source.strip()]
            
            return {
                'score': max(-1, min(1, score)),  # Clamp to [-1, 1]
                'confidence': max(0, min(1, confidence)),  # Clamp to [0, 1]
                'key_themes': themes[:5],  # Limit to top 5
                'market_impact': market_impact,
                'sources': sources[:10]  # Limit to top 10
            }
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            return {
                'score': 0.0,
                'confidence': 0.5,
                'key_themes': [],
                'market_impact': 'neutral',
                'sources': []
            }
    
    def _parse_forecast_response(self, response_text: str) -> Dict[str, Any]:
        """Parse forecast response from text"""
        try:
            # Extract probability
            prob_match = re.search(r'probability.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            probability = float(prob_match.group(1)) if prob_match else 0.5
            
            # Extract confidence
            conf_match = re.search(r'confidence.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.7
            
            # Extract key factors
            key_factors = []
            factors_section = re.search(r'key.*factors?\s*:?\s*(.*?)(?:\n\n|\nrisk|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if factors_section:
                factor_text = factors_section.group(1)
                key_factors = [factor.strip('- ') for factor in factor_text.split('\n') if factor.strip()]
            
            # Extract risk factors
            risk_factors = []
            risk_section = re.search(r'risk.*factors?\s*:?\s*(.*?)(?:\n\n|\nsupporting|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if risk_section:
                risk_text = risk_section.group(1)
                risk_factors = [risk.strip('- ') for risk in risk_text.split('\n') if risk.strip()]
            
            return {
                'probability': max(0, min(1, probability)),  # Clamp to [0, 1]
                'confidence': max(0, min(1, confidence)),  # Clamp to [0, 1]
                'key_factors': key_factors[:5],  # Limit to top 5
                'risk_factors': risk_factors[:5],  # Limit to top 5
                'supporting_data': {'raw_response': response_text}
            }
            
        except Exception as e:
            logger.error(f"Error parsing forecast response: {e}")
            return {
                'probability': 0.5,
                'confidence': 0.5,
                'key_factors': [],
                'risk_factors': [],
                'supporting_data': {}
            }
    
    def _parse_trading_signal_response(self, response_text: str) -> Dict[str, Any]:
        """Parse trading signal response from text"""
        try:
            # Extract signal type
            signal_match = re.search(r'signal.*?:?\s*(BUY|SELL|HOLD)', response_text, re.IGNORECASE)
            signal = signal_match.group(1).upper() if signal_match else 'HOLD'
            
            # Extract strength
            strength_match = re.search(r'strength.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            strength = float(strength_match.group(1)) if strength_match else 0.5
            
            # Extract probability
            prob_match = re.search(r'probability.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            probability = float(prob_match.group(1)) if prob_match else 0.5
            
            # Extract risk level
            risk_match = re.search(r'risk.*?:?\s*(low|medium|high)', response_text, re.IGNORECASE)
            risk_level = risk_match.group(1).lower() if risk_match else 'medium'
            
            # Extract expected return
            return_match = re.search(r'expected.*return.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            expected_return = float(return_match.group(1)) if return_match else None
            
            # Extract stop loss
            stop_match = re.search(r'stop.*loss.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            stop_loss = float(stop_match.group(1)) if stop_match else None
            
            # Extract take profit
            profit_match = re.search(r'take.*profit.*?(\d+\.?\d*)', response_text, re.IGNORECASE)
            take_profit = float(profit_match.group(1)) if profit_match else None
            
            # Extract timeframe
            time_match = re.search(r'timeframe.*?:?\s*(\w+)', response_text, re.IGNORECASE)
            timeframe = time_match.group(1) if time_match else '1d'
            
            # Extract reasoning
            reasoning = response_text[:500] + "..." if len(response_text) > 500 else response_text
            
            return {
                'signal': signal,
                'strength': max(0, min(1, strength)),  # Clamp to [0, 1]
                'probability': max(0, min(1, probability)),  # Clamp to [0, 1]
                'risk_level': risk_level,
                'expected_return': expected_return,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timeframe': timeframe,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Error parsing trading signal response: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0.5,
                'probability': 0.5,
                'risk_level': 'medium',
                'expected_return': None,
                'stop_loss': None,
                'take_profit': None,
                'timeframe': '1d',
                'reasoning': 'Error parsing response'
            } 