"""
Risk Management System for Kalshi Trading Bot
"""

import asyncio
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from loguru import logger

from kalshi_client import KalshiClient, Market, Order, OrderSide, Portfolio, Position
from strategy_engine import TradingOpportunity, TradeDirection, StrategyType

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    portfolio_value: float
    daily_pnl: float
    daily_pnl_percent: float
    max_drawdown: float
    value_at_risk: float
    sharpe_ratio: float
    portfolio_volatility: float
    concentration_risk: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PositionSizing:
    market_id: str
    max_position_size: float
    recommended_size: float
    risk_adjusted_size: float
    kelly_size: float
    volatility_adjusted_size: float
    confidence_adjusted_size: float
    final_position_size: float
    reasoning: str

@dataclass
class RiskAlert:
    alert_type: str
    severity: RiskLevel
    message: str
    action_required: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StopLoss:
    market_id: str
    stop_price: float
    stop_type: str  # "percentage", "absolute", "trailing"
    created_at: datetime
    triggered: bool = False
    order_id: Optional[str] = None

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, kalshi_client: KalshiClient, config: Dict[str, Any]):
        self.kalshi_client = kalshi_client
        self.config = config
        
        # Risk parameters from config
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.2)
        self.stop_loss_percent = config.get('stop_loss_percent', 0.15)
        self.take_profit_percent = config.get('take_profit_percent', 0.25)
        self.min_probability_threshold = config.get('min_probability_threshold', 0.55)
        self.max_concurrent_positions = config.get('max_concurrent_positions', 5)
        
        # Risk tracking
        self.active_stop_losses: Dict[str, StopLoss] = {}
        self.daily_trades: List[Dict] = []
        self.risk_alerts: List[RiskAlert] = []
        self.performance_history: List[RiskMetrics] = []
        
        # Portfolio tracking
        self.initial_portfolio_value = 0.0
        self.current_portfolio_value = 0.0
        self.daily_start_value = 0.0
        self.peak_portfolio_value = 0.0
        
    async def initialize(self):
        """Initialize risk manager with current portfolio state"""
        try:
            portfolio = await self.kalshi_client.get_portfolio()
            self.initial_portfolio_value = portfolio.balance
            self.current_portfolio_value = portfolio.balance
            self.daily_start_value = portfolio.balance
            self.peak_portfolio_value = portfolio.balance
            
            logger.info(f"Risk manager initialized with portfolio value: ${self.current_portfolio_value:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            raise
    
    async def calculate_position_size(self, opportunity: TradingOpportunity) -> PositionSizing:
        """Calculate optimal position size using multiple risk models"""
        portfolio = await self.kalshi_client.get_portfolio()
        
        # Base position size limits
        max_position_value = portfolio.available_balance * self.max_position_size
        max_position_size = max_position_value / opportunity.entry_price
        
        # Kelly Criterion sizing
        kelly_size = self._calculate_kelly_size(opportunity, portfolio.available_balance)
        
        # Volatility-adjusted sizing
        volatility_adjusted_size = await self._calculate_volatility_adjusted_size(
            opportunity, portfolio.available_balance
        )
        
        # Confidence-adjusted sizing
        confidence_adjusted_size = max_position_size * opportunity.confidence
        
        # Risk-adjusted sizing based on strategy
        risk_adjusted_size = self._calculate_risk_adjusted_size(
            opportunity, max_position_size
        )
        
        # Final position size (most conservative)
        final_size = min(
            max_position_size,
            kelly_size,
            volatility_adjusted_size,
            confidence_adjusted_size,
            risk_adjusted_size
        )
        
        # Ensure minimum viable position
        min_position = 1.0  # Minimum 1 contract
        final_size = max(final_size, min_position) if final_size > 0 else 0
        
        reasoning = f"Kelly: {kelly_size:.1f}, Vol-adj: {volatility_adjusted_size:.1f}, " \
                   f"Conf-adj: {confidence_adjusted_size:.1f}, Risk-adj: {risk_adjusted_size:.1f}"
        
        return PositionSizing(
            market_id=opportunity.market_id,
            max_position_size=max_position_size,
            recommended_size=final_size,
            risk_adjusted_size=risk_adjusted_size,
            kelly_size=kelly_size,
            volatility_adjusted_size=volatility_adjusted_size,
            confidence_adjusted_size=confidence_adjusted_size,
            final_position_size=final_size,
            reasoning=reasoning
        )
    
    def _calculate_kelly_size(self, opportunity: TradingOpportunity, available_balance: float) -> float:
        """Calculate Kelly criterion position size"""
        # Kelly formula: f = (bp - q) / b
        # where f = fraction of capital to wager
        # b = odds received (payout ratio)
        # p = probability of winning
        # q = probability of losing (1-p)
        
        win_probability = opportunity.confidence
        loss_probability = 1 - win_probability
        
        # Calculate odds from expected return
        if opportunity.expected_return > 0:
            odds = opportunity.expected_return / (1 - opportunity.expected_return)
        else:
            odds = 0.1  # Default conservative odds
        
        # Kelly fraction
        kelly_fraction = (odds * win_probability - loss_probability) / odds
        
        # Cap Kelly fraction at 25% for safety
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        # Convert to position size
        position_value = available_balance * kelly_fraction
        position_size = position_value / opportunity.entry_price
        
        return position_size
    
    async def _calculate_volatility_adjusted_size(self, opportunity: TradingOpportunity, 
                                                available_balance: float) -> float:
        """Calculate position size adjusted for market volatility"""
        try:
            # Get market history for volatility calculation
            history = await self.kalshi_client.get_market_history(opportunity.market_id, limit=30)
            
            if len(history) < 10:
                # Default to conservative sizing if insufficient history
                return available_balance * 0.05 / opportunity.entry_price
            
            # Calculate historical volatility
            prices = [h.get('price', 0) for h in history]
            returns = [
                (prices[i] - prices[i-1]) / prices[i-1] 
                for i in range(1, len(prices)) 
                if prices[i-1] != 0
            ]
            
            if not returns:
                return available_balance * 0.05 / opportunity.entry_price
            
            volatility = np.std(returns)
            
            # Adjust position size inversely with volatility
            # Higher volatility = smaller position
            base_size = available_balance * self.max_position_size / opportunity.entry_price
            volatility_adjustment = 1 / (1 + volatility * 10)  # Scale volatility impact
            
            volatility_adjusted_size = base_size * volatility_adjustment
            
            return volatility_adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return available_balance * 0.05 / opportunity.entry_price
    
    def _calculate_risk_adjusted_size(self, opportunity: TradingOpportunity, 
                                    max_position_size: float) -> float:
        """Calculate position size based on risk level and strategy"""
        
        # Risk multipliers by level
        risk_multipliers = {
            "low": 1.0,
            "medium": 0.7,
            "high": 0.4,
            "extreme": 0.2
        }
        
        risk_multiplier = risk_multipliers.get(opportunity.risk_level, 0.5)
        
        # Strategy-specific adjustments
        strategy_multipliers = {
            StrategyType.RESEARCH_BASED_TRADES: 1.0,
            StrategyType.SENTIMENT_MOMENTUM: 0.8,
            StrategyType.EVENT_ARBITRAGE: 0.9,
            StrategyType.POLITICAL_POLLING: 0.9,
            StrategyType.MEAN_REVERSION: 0.6,
            StrategyType.CORRELATION_PAIRS: 0.7
        }
        
        strategy_multiplier = strategy_multipliers.get(opportunity.strategy_type, 0.5)
        
        # Combined adjustment
        adjusted_size = max_position_size * risk_multiplier * strategy_multiplier
        
        return adjusted_size
    
    async def check_portfolio_risk(self) -> Tuple[bool, List[RiskAlert]]:
        """Check overall portfolio risk and generate alerts"""
        alerts = []
        
        try:
            portfolio = await self.kalshi_client.get_portfolio()
            self.current_portfolio_value = portfolio.balance
            
            # Update peak value
            if self.current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.current_portfolio_value
            
            # Calculate daily P&L
            daily_pnl = self.current_portfolio_value - self.daily_start_value
            daily_pnl_percent = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
            
            # Check daily loss limit
            if daily_pnl_percent < -self.max_daily_loss:
                alerts.append(RiskAlert(
                    alert_type="daily_loss_limit",
                    severity=RiskLevel.HIGH,
                    message=f"Daily loss limit exceeded: {daily_pnl_percent:.2%}",
                    action_required="STOP_TRADING"
                ))
            
            # Check drawdown
            drawdown = (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value
            if drawdown > self.max_portfolio_risk:
                alerts.append(RiskAlert(
                    alert_type="max_drawdown",
                    severity=RiskLevel.EXTREME,
                    message=f"Maximum drawdown exceeded: {drawdown:.2%}",
                    action_required="EMERGENCY_STOP"
                ))
            
            # Check position concentration
            concentration_risk = await self._calculate_concentration_risk(portfolio)
            if concentration_risk > 0.5:
                alerts.append(RiskAlert(
                    alert_type="concentration_risk",
                    severity=RiskLevel.MEDIUM,
                    message=f"High concentration risk: {concentration_risk:.2%}",
                    action_required="DIVERSIFY"
                ))
            
            # Check number of concurrent positions
            active_positions = len([p for p in portfolio.positions if p.quantity != 0])
            if active_positions > self.max_concurrent_positions:
                alerts.append(RiskAlert(
                    alert_type="position_limit",
                    severity=RiskLevel.MEDIUM,
                    message=f"Too many concurrent positions: {active_positions}",
                    action_required="REDUCE_POSITIONS"
                ))
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            # Calculate and store risk metrics
            risk_metrics = await self._calculate_risk_metrics(portfolio)
            self.performance_history.append(risk_metrics)
            
            # Return if trading should continue
            emergency_stop = any(alert.action_required == "EMERGENCY_STOP" for alert in alerts)
            stop_trading = any(alert.action_required in ["STOP_TRADING", "EMERGENCY_STOP"] for alert in alerts)
            
            return not stop_trading, alerts
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return False, [RiskAlert(
                alert_type="system_error",
                severity=RiskLevel.HIGH,
                message=f"Risk check failed: {e}",
                action_required="STOP_TRADING"
            )]
    
    async def _calculate_concentration_risk(self, portfolio: Portfolio) -> float:
        """Calculate portfolio concentration risk"""
        if not portfolio.positions:
            return 0.0
        
        # Calculate position values
        position_values = []
        for position in portfolio.positions:
            if position.quantity != 0:
                position_value = abs(position.quantity * position.avg_price)
                position_values.append(position_value)
        
        if not position_values:
            return 0.0
        
        total_value = sum(position_values)
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum((value / total_value) ** 2 for value in position_values)
        
        return hhi
    
    async def _calculate_risk_metrics(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        daily_pnl = self.current_portfolio_value - self.daily_start_value
        daily_pnl_percent = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = (self.peak_portfolio_value - self.current_portfolio_value) / self.peak_portfolio_value
        
        # Calculate Value at Risk (simplified)
        # This would normally use historical returns
        var_95 = self.current_portfolio_value * 0.05  # 5% VaR approximation
        
        # Calculate Sharpe ratio (simplified)
        # This would normally use historical returns and risk-free rate
        sharpe_ratio = daily_pnl_percent / (0.02 if daily_pnl_percent != 0 else 1)  # Rough approximation
        
        # Calculate portfolio volatility
        portfolio_volatility = abs(daily_pnl_percent)  # Simplified
        
        # Calculate concentration risk
        concentration_risk = await self._calculate_concentration_risk(portfolio)
        
        return RiskMetrics(
            portfolio_value=self.current_portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            max_drawdown=max_drawdown,
            value_at_risk=var_95,
            sharpe_ratio=sharpe_ratio,
            portfolio_volatility=portfolio_volatility,
            concentration_risk=concentration_risk
        )
    
    async def create_stop_loss(self, market_id: str, entry_price: float, 
                             direction: TradeDirection, stop_type: str = "percentage") -> StopLoss:
        """Create stop loss order"""
        
        if direction == TradeDirection.BUY:
            # Stop loss below entry price for long positions
            stop_price = entry_price * (1 - self.stop_loss_percent)
        else:
            # Stop loss above entry price for short positions  
            stop_price = entry_price * (1 + self.stop_loss_percent)
        
        stop_loss = StopLoss(
            market_id=market_id,
            stop_price=stop_price,
            stop_type=stop_type,
            created_at=datetime.now()
        )
        
        self.active_stop_losses[market_id] = stop_loss
        
        logger.info(f"Created stop loss for {market_id} at {stop_price:.2f}")
        
        return stop_loss
    
    async def check_stop_losses(self) -> List[str]:
        """Check all active stop losses and return markets that should be closed"""
        markets_to_close = []
        
        for market_id, stop_loss in self.active_stop_losses.items():
            if stop_loss.triggered:
                continue
                
            try:
                market = await self.kalshi_client.get_market(market_id)
                current_price = market.last_price or market.bid or market.ask
                
                if current_price is None:
                    continue
                
                # Check if stop loss should trigger
                should_trigger = False
                
                if stop_loss.stop_type == "percentage":
                    # For buy positions, trigger if price falls below stop
                    # For sell positions, trigger if price rises above stop
                    portfolio = await self.kalshi_client.get_portfolio()
                    position = next(
                        (p for p in portfolio.positions if p.market_id == market_id), 
                        None
                    )
                    
                    if position and position.quantity > 0:  # Long position
                        should_trigger = current_price <= stop_loss.stop_price
                    elif position and position.quantity < 0:  # Short position
                        should_trigger = current_price >= stop_loss.stop_price
                
                if should_trigger:
                    stop_loss.triggered = True
                    markets_to_close.append(market_id)
                    
                    logger.warning(f"Stop loss triggered for {market_id} at {current_price:.2f}")
                    
            except Exception as e:
                logger.error(f"Error checking stop loss for {market_id}: {e}")
        
        return markets_to_close
    
    def validate_trade(self, opportunity: TradingOpportunity, position_size: float) -> Tuple[bool, str]:
        """Validate if a trade meets risk criteria"""
        
        # Check minimum probability threshold
        if opportunity.confidence < self.min_probability_threshold:
            return False, f"Confidence {opportunity.confidence:.2f} below threshold {self.min_probability_threshold}"
        
        # Check position size limits
        if position_size <= 0:
            return False, "Position size must be positive"
        
        # Check if we have too many positions
        if len(self.active_stop_losses) >= self.max_concurrent_positions:
            return False, f"Maximum concurrent positions ({self.max_concurrent_positions}) reached"
        
        # Check expected return
        if opportunity.expected_return < 0.05:
            return False, f"Expected return {opportunity.expected_return:.2%} too low"
        
        return True, "Trade validated"
    
    async def reset_daily_tracking(self):
        """Reset daily tracking at start of new trading day"""
        portfolio = await self.kalshi_client.get_portfolio()
        self.daily_start_value = portfolio.balance
        self.daily_trades = []
        
        # Clear old alerts
        self.risk_alerts = [
            alert for alert in self.risk_alerts 
            if alert.timestamp > datetime.now() - timedelta(days=1)
        ]
        
        logger.info("Daily risk tracking reset")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        latest_metrics = self.performance_history[-1] if self.performance_history else None
        
        return {
            "portfolio_value": self.current_portfolio_value,
            "daily_pnl": latest_metrics.daily_pnl if latest_metrics else 0,
            "daily_pnl_percent": latest_metrics.daily_pnl_percent if latest_metrics else 0,
            "max_drawdown": latest_metrics.max_drawdown if latest_metrics else 0,
            "active_positions": len(self.active_stop_losses),
            "max_positions": self.max_concurrent_positions,
            "active_alerts": len([a for a in self.risk_alerts if a.timestamp > datetime.now() - timedelta(hours=1)]),
            "concentration_risk": latest_metrics.concentration_risk if latest_metrics else 0,
            "risk_limits": {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "max_portfolio_risk": self.max_portfolio_risk,
                "stop_loss_percent": self.stop_loss_percent
            }
        }
    
    def get_position_sizing_recommendation(self, opportunity: TradingOpportunity) -> str:
        """Get human-readable position sizing recommendation"""
        # This would be called after calculate_position_size
        return f"""
        Position Sizing Recommendation for {opportunity.market_title}:
        
        Strategy: {opportunity.strategy_type.value}
        Direction: {opportunity.direction.value}
        Confidence: {opportunity.confidence:.2%}
        Expected Return: {opportunity.expected_return:.2%}
        Risk Level: {opportunity.risk_level}
        
        Recommended Position Size: TBD (call calculate_position_size first)
        Entry Price: ${opportunity.entry_price:.2f}
        Stop Loss: ${opportunity.stop_loss or 'N/A'}
        Take Profit: ${opportunity.take_profit or 'N/A'}
        """ 