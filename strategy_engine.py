"""
Trading Strategy Engine that combines Octagon Deep Research with Kalshi market analysis
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from loguru import logger

from kalshi_client import KalshiClient, Market, Order, OrderSide, OrderType, Portfolio
from research_client import (
    OctagonResearchClient, MarketResearch, TradingSignal, 
    SentimentAnalysis, MarketForecast, ResearchInsight
)

class StrategyType(Enum):
    SENTIMENT_MOMENTUM = "sentiment_momentum"
    EVENT_ARBITRAGE = "event_arbitrage"
    RESEARCH_BASED_TRADES = "research_based_trades"
    POLITICAL_POLLING = "political_polling"
    MEAN_REVERSION = "mean_reversion"
    CORRELATION_PAIRS = "correlation_pairs"

class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingOpportunity:
    market_id: str
    market_title: str
    strategy_type: StrategyType
    direction: TradeDirection
    confidence: float
    expected_return: float
    risk_level: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    reasoning: str = ""
    research_score: float = 0.0
    technical_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class StrategyPerformance:
    strategy_type: StrategyType
    total_trades: int
    winning_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period: float
    last_updated: datetime = field(default_factory=datetime.now)

class StrategyEngine:
    """Trading strategy engine that combines research insights with market analysis"""
    
    def __init__(self, kalshi_client: KalshiClient, research_client: OctagonResearchClient,
                 enabled_strategies: List[str], config: Dict[str, Any]):
        self.kalshi_client = kalshi_client
        self.research_client = research_client
        self.enabled_strategies = [StrategyType(s) for s in enabled_strategies]
        self.config = config
        
        # Strategy state
        self.active_opportunities: List[TradingOpportunity] = []
        self.research_cache: Dict[str, MarketResearch] = {}
        self.market_cache: Dict[str, Market] = {}
        self.performance_tracking: Dict[StrategyType, StrategyPerformance] = {}
        
        # Initialize performance tracking
        for strategy in self.enabled_strategies:
            self.performance_tracking[strategy] = StrategyPerformance(
                strategy_type=strategy,
                total_trades=0,
                winning_trades=0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_holding_period=0.0
            )
    
    async def analyze_markets(self) -> List[TradingOpportunity]:
        """Analyze all available markets for trading opportunities"""
        logger.info("Starting market analysis for trading opportunities")
        
        # Get all open markets
        markets = await self.kalshi_client.get_markets(status="OPEN")
        
        # Filter markets based on tradability criteria
        tradable_markets = [
            market for market in markets 
            if self.kalshi_client.is_market_tradable(
                market, 
                self.config.get('min_volume_threshold', 1000),
                self.config.get('max_spread', 0.05)
            )
        ]
        
        logger.info(f"Found {len(tradable_markets)} tradable markets")
        
        # Analyze markets in parallel (batches to respect rate limits)
        opportunities = []
        batch_size = 5  # Process 5 markets at a time
        
        for i in range(0, len(tradable_markets), batch_size):
            batch = tradable_markets[i:i + batch_size]
            batch_opportunities = await self._analyze_market_batch(batch)
            opportunities.extend(batch_opportunities)
            
            # Small delay between batches to respect rate limits
            await asyncio.sleep(1)
        
        # Sort opportunities by confidence score
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        
        self.active_opportunities = opportunities
        logger.info(f"Identified {len(opportunities)} trading opportunities")
        
        return opportunities
    
    async def _analyze_market_batch(self, markets: List[Market]) -> List[TradingOpportunity]:
        """Analyze a batch of markets for trading opportunities"""
        opportunities = []
        
        # Get research for all markets in parallel
        research_tasks = []
        for market in markets:
            current_price = market.last_price or (market.bid + market.ask) / 2 if market.bid and market.ask else 50.0
            research_tasks.append(
                self.research_client.comprehensive_market_research(
                    market.id, market.title, current_price
                )
            )
        
        research_results = await asyncio.gather(*research_tasks, return_exceptions=True)
        
        # Analyze each market with its research
        for market, research in zip(markets, research_results):
            if isinstance(research, Exception):
                logger.error(f"Research failed for market {market.id}: {research}")
                continue
                
            # Cache research and market data
            self.research_cache[market.id] = research
            self.market_cache[market.id] = market
            
            # Apply enabled strategies
            for strategy_type in self.enabled_strategies:
                opportunity = await self._apply_strategy(market, research, strategy_type)
                if opportunity:
                    opportunities.append(opportunity)
        
        return opportunities
    
    async def _apply_strategy(self, market: Market, research: MarketResearch, 
                            strategy_type: StrategyType) -> Optional[TradingOpportunity]:
        """Apply a specific strategy to a market"""
        
        if strategy_type == StrategyType.SENTIMENT_MOMENTUM:
            return await self._sentiment_momentum_strategy(market, research)
        elif strategy_type == StrategyType.EVENT_ARBITRAGE:
            return await self._event_arbitrage_strategy(market, research)
        elif strategy_type == StrategyType.RESEARCH_BASED_TRADES:
            return await self._research_based_strategy(market, research)
        elif strategy_type == StrategyType.POLITICAL_POLLING:
            return await self._political_polling_strategy(market, research)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return await self._mean_reversion_strategy(market, research)
        elif strategy_type == StrategyType.CORRELATION_PAIRS:
            return await self._correlation_pairs_strategy(market, research)
        
        return None
    
    async def _sentiment_momentum_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy based on sentiment momentum"""
        if not research.sentiment or not research.signals:
            return None
        
        sentiment = research.sentiment
        signal = research.signals[0]
        
        # Strong positive sentiment with high confidence
        if sentiment.score > 0.3 and sentiment.confidence > 0.7:
            if signal.signal_type == "BUY" and signal.strength > 0.6:
                return TradingOpportunity(
                    market_id=market.id,
                    market_title=market.title,
                    strategy_type=StrategyType.SENTIMENT_MOMENTUM,
                    direction=TradeDirection.BUY,
                    confidence=sentiment.confidence * signal.strength,
                    expected_return=signal.expected_return or 0.15,
                    risk_level=signal.risk_level,
                    entry_price=market.ask or market.last_price or 50.0,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    reasoning=f"Strong positive sentiment ({sentiment.score:.2f}) with buy signal",
                    research_score=research.overall_score,
                    technical_score=self._calculate_technical_score(market)
                )
        
        # Strong negative sentiment for short opportunities
        elif sentiment.score < -0.3 and sentiment.confidence > 0.7:
            if signal.signal_type == "SELL" and signal.strength > 0.6:
                return TradingOpportunity(
                    market_id=market.id,
                    market_title=market.title,
                    strategy_type=StrategyType.SENTIMENT_MOMENTUM,
                    direction=TradeDirection.SELL,
                    confidence=sentiment.confidence * signal.strength,
                    expected_return=signal.expected_return or 0.15,
                    risk_level=signal.risk_level,
                    entry_price=market.bid or market.last_price or 50.0,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    reasoning=f"Strong negative sentiment ({sentiment.score:.2f}) with sell signal",
                    research_score=research.overall_score,
                    technical_score=self._calculate_technical_score(market)
                )
        
        return None
    
    async def _event_arbitrage_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy based on event arbitrage opportunities"""
        if not research.forecast or not research.signals:
            return None
        
        forecast = research.forecast
        signal = research.signals[0]
        current_price = market.last_price or (market.bid + market.ask) / 2 if market.bid and market.ask else 50.0
        
        # Look for significant price-probability mismatches
        implied_probability = self.kalshi_client.calculate_implied_probability(current_price)
        forecast_probability = forecast.probability
        
        probability_diff = abs(forecast_probability - implied_probability)
        
        # Significant arbitrage opportunity
        if probability_diff > 0.15 and forecast.confidence > 0.7:
            direction = TradeDirection.BUY if forecast_probability > implied_probability else TradeDirection.SELL
            
            return TradingOpportunity(
                market_id=market.id,
                market_title=market.title,
                strategy_type=StrategyType.EVENT_ARBITRAGE,
                direction=direction,
                confidence=forecast.confidence,
                expected_return=probability_diff * 0.8,  # Conservative estimate
                risk_level="medium",
                entry_price=market.ask if direction == TradeDirection.BUY else market.bid,
                reasoning=f"Probability mismatch: forecast {forecast_probability:.2f} vs implied {implied_probability:.2f}",
                research_score=research.overall_score,
                technical_score=self._calculate_technical_score(market)
            )
        
        return None
    
    async def _research_based_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy based on comprehensive research analysis"""
        if not research.signals or research.overall_score < 0.6:
            return None
        
        signal = research.signals[0]
        
        # Only take high-confidence research signals
        if signal.strength > 0.7 and signal.probability > 0.65:
            direction = TradeDirection.BUY if signal.signal_type == "BUY" else TradeDirection.SELL
            
            return TradingOpportunity(
                market_id=market.id,
                market_title=market.title,
                strategy_type=StrategyType.RESEARCH_BASED_TRADES,
                direction=direction,
                confidence=signal.strength * signal.probability,
                expected_return=signal.expected_return or 0.20,
                risk_level=signal.risk_level,
                entry_price=market.ask if direction == TradeDirection.BUY else market.bid,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reasoning=f"High-confidence research signal: {signal.reasoning}",
                research_score=research.overall_score,
                technical_score=self._calculate_technical_score(market)
            )
        
        return None
    
    async def _political_polling_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy focused on political markets using polling data"""
        if "politics" not in market.title.lower() and "election" not in market.title.lower():
            return None
        
        if not research.forecast or not research.sentiment:
            return None
        
        forecast = research.forecast
        sentiment = research.sentiment
        
        # Look for polling-based opportunities
        if forecast.confidence > 0.8 and "poll" in str(forecast.key_factors).lower():
            current_price = market.last_price or (market.bid + market.ask) / 2 if market.bid and market.ask else 50.0
            implied_probability = self.kalshi_client.calculate_implied_probability(current_price)
            
            # Significant divergence from polling data
            if abs(forecast.probability - implied_probability) > 0.1:
                direction = TradeDirection.BUY if forecast.probability > implied_probability else TradeDirection.SELL
                
                return TradingOpportunity(
                    market_id=market.id,
                    market_title=market.title,
                    strategy_type=StrategyType.POLITICAL_POLLING,
                    direction=direction,
                    confidence=forecast.confidence,
                    expected_return=abs(forecast.probability - implied_probability) * 0.7,
                    risk_level="medium",
                    entry_price=market.ask if direction == TradeDirection.BUY else market.bid,
                    reasoning=f"Polling data divergence: forecast {forecast.probability:.2f} vs market {implied_probability:.2f}",
                    research_score=research.overall_score,
                    technical_score=self._calculate_technical_score(market)
                )
        
        return None
    
    async def _mean_reversion_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy based on mean reversion patterns"""
        # Get market history for technical analysis
        try:
            history = await self.kalshi_client.get_market_history(market.id, limit=50)
            if len(history) < 20:
                return None
            
            prices = [h.get('price', 0) for h in history]
            current_price = market.last_price or prices[-1] if prices else 50.0
            
            # Calculate moving averages
            short_ma = np.mean(prices[-5:])
            long_ma = np.mean(prices[-20:])
            
            # Mean reversion signal
            if current_price < long_ma * 0.85 and research.overall_score > 0.5:
                return TradingOpportunity(
                    market_id=market.id,
                    market_title=market.title,
                    strategy_type=StrategyType.MEAN_REVERSION,
                    direction=TradeDirection.BUY,
                    confidence=0.6,
                    expected_return=0.10,
                    risk_level="medium",
                    entry_price=market.ask or current_price,
                    reasoning=f"Mean reversion buy: price {current_price:.2f} below long MA {long_ma:.2f}",
                    research_score=research.overall_score,
                    technical_score=self._calculate_technical_score(market)
                )
            
            elif current_price > long_ma * 1.15 and research.overall_score < 0.4:
                return TradingOpportunity(
                    market_id=market.id,
                    market_title=market.title,
                    strategy_type=StrategyType.MEAN_REVERSION,
                    direction=TradeDirection.SELL,
                    confidence=0.6,
                    expected_return=0.10,
                    risk_level="medium",
                    entry_price=market.bid or current_price,
                    reasoning=f"Mean reversion sell: price {current_price:.2f} above long MA {long_ma:.2f}",
                    research_score=research.overall_score,
                    technical_score=self._calculate_technical_score(market)
                )
                
        except Exception as e:
            logger.error(f"Error in mean reversion strategy for {market.id}: {e}")
        
        return None
    
    async def _correlation_pairs_strategy(self, market: Market, research: MarketResearch) -> Optional[TradingOpportunity]:
        """Strategy based on market correlations"""
        # This would require analysis of multiple markets
        # For now, return None as it needs more complex implementation
        return None
    
    def _calculate_technical_score(self, market: Market) -> float:
        """Calculate technical analysis score for a market"""
        score = 0.5  # Base score
        
        # Volume factor
        if market.volume:
            if market.volume > 5000:
                score += 0.2
            elif market.volume > 1000:
                score += 0.1
        
        # Spread factor
        spread = self.kalshi_client.calculate_spread(market)
        if spread:
            if spread < 0.02:
                score += 0.2
            elif spread < 0.05:
                score += 0.1
        
        # Price momentum (simplified)
        if market.last_price:
            if market.bid and market.ask:
                mid_price = (market.bid + market.ask) / 2
                if market.last_price > mid_price:
                    score += 0.1
                elif market.last_price < mid_price:
                    score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def filter_opportunities(self, opportunities: List[TradingOpportunity], 
                           min_confidence: float = 0.6, max_count: int = 10) -> List[TradingOpportunity]:
        """Filter opportunities based on confidence and other criteria"""
        filtered = [
            opp for opp in opportunities 
            if opp.confidence >= min_confidence and opp.expected_return > 0.05
        ]
        
        # Sort by combined score (confidence * expected_return)
        filtered.sort(key=lambda x: x.confidence * x.expected_return, reverse=True)
        
        return filtered[:max_count]
    
    def get_strategy_performance(self, strategy_type: StrategyType) -> StrategyPerformance:
        """Get performance metrics for a specific strategy"""
        return self.performance_tracking.get(strategy_type, StrategyPerformance(
            strategy_type=strategy_type,
            total_trades=0,
            winning_trades=0,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_holding_period=0.0
        ))
    
    def update_strategy_performance(self, strategy_type: StrategyType, 
                                  trade_return: float, holding_period: float):
        """Update performance tracking for a strategy"""
        if strategy_type not in self.performance_tracking:
            self.performance_tracking[strategy_type] = StrategyPerformance(
                strategy_type=strategy_type,
                total_trades=0,
                winning_trades=0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_holding_period=0.0
            )
        
        perf = self.performance_tracking[strategy_type]
        perf.total_trades += 1
        perf.total_return += trade_return
        perf.avg_holding_period = (perf.avg_holding_period * (perf.total_trades - 1) + holding_period) / perf.total_trades
        
        if trade_return > 0:
            perf.winning_trades += 1
        
        perf.last_updated = datetime.now()
    
    def get_market_research(self, market_id: str) -> Optional[MarketResearch]:
        """Get cached research for a market"""
        return self.research_cache.get(market_id)
    
    def get_market_data(self, market_id: str) -> Optional[Market]:
        """Get cached market data"""
        return self.market_cache.get(market_id)
    
    async def refresh_research(self, market_id: str) -> Optional[MarketResearch]:
        """Refresh research data for a specific market"""
        market = self.market_cache.get(market_id)
        if not market:
            return None
        
        current_price = market.last_price or (market.bid + market.ask) / 2 if market.bid and market.ask else 50.0
        
        research = await self.research_client.comprehensive_market_research(
            market.id, market.title, current_price
        )
        
        self.research_cache[market_id] = research
        return research
    
    def get_opportunity_summary(self, opportunities: List[TradingOpportunity]) -> Dict[str, Any]:
        """Get a summary of trading opportunities"""
        if not opportunities:
            return {
                "total_opportunities": 0,
                "avg_confidence": 0.0,
                "avg_expected_return": 0.0,
                "strategy_breakdown": {}
            }
        
        strategy_breakdown = {}
        for opp in opportunities:
            strategy_name = opp.strategy_type.value
            if strategy_name not in strategy_breakdown:
                strategy_breakdown[strategy_name] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_expected_return": 0.0
                }
            
            strategy_breakdown[strategy_name]["count"] += 1
        
        # Calculate averages
        for strategy_name in strategy_breakdown:
            strategy_opportunities = [o for o in opportunities if o.strategy_type.value == strategy_name]
            strategy_breakdown[strategy_name]["avg_confidence"] = np.mean([o.confidence for o in strategy_opportunities])
            strategy_breakdown[strategy_name]["avg_expected_return"] = np.mean([o.expected_return for o in strategy_opportunities])
        
        return {
            "total_opportunities": len(opportunities),
            "avg_confidence": np.mean([o.confidence for o in opportunities]),
            "avg_expected_return": np.mean([o.expected_return for o in opportunities]),
            "strategy_breakdown": strategy_breakdown
        } 