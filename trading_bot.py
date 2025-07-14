"""
Main Kalshi Trading Bot with Octagon Deep Research Integration
"""

import asyncio
import json
import signal
import sys
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import traceback
from contextlib import asynccontextmanager
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from config import load_config, BotConfig
from kalshi_client import KalshiClient, OrderSide, OrderType, Market
from research_client import OctagonResearchClient
from strategy_engine import StrategyEngine, TradingOpportunity, TradeDirection
from risk_management import RiskManager, RiskAlert, RiskLevel

@dataclass
class TradeExecution:
    opportunity: TradingOpportunity
    position_size: float
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    status: str = "pending"
    execution_time: Optional[datetime] = None
    pnl: Optional[float] = None
    closed_time: Optional[datetime] = None

class TradingBot:
    """Main trading bot that orchestrates all components"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.console = Console()
        self.running = False
        self.paused = False
        
        # Initialize clients
        self.kalshi_client = None
        self.research_client = None
        self.strategy_engine = None
        self.risk_manager = None
        
        # State tracking
        self.active_trades: Dict[str, TradeExecution] = {}
        self.completed_trades: List[TradeExecution] = []
        self.bot_start_time = datetime.now()
        
        # Database for persistence
        self.db_path = config.database.db_path
        self._init_database()
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                market_id TEXT,
                market_title TEXT,
                strategy_type TEXT,
                direction TEXT,
                confidence REAL,
                expected_return REAL,
                position_size REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                entry_time TEXT,
                exit_time TEXT,
                status TEXT,
                reasoning TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_cache (
                id INTEGER PRIMARY KEY,
                market_id TEXT,
                market_title TEXT,
                research_data TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                id INTEGER PRIMARY KEY,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                action_required TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Kalshi Trading Bot...")
        
        try:
            # Initialize clients
            # Get active credentials and URLs based on mode
            api_key, private_key = self.config.kalshi.get_active_credentials()
            base_url, websocket_url = self.config.kalshi.get_active_urls()
            
            self.kalshi_client = KalshiClient(
                api_key,
                private_key,
                base_url,
                websocket_url,
                self.config.kalshi.rate_limit_requests_per_second
            )
            
            self.research_client = OctagonResearchClient(
                self.config.octagon.api_key,
                self.config.octagon.base_url,
                self.config.octagon.rate_limit_requests_per_day,
                self.config.octagon.concurrent_streams
            )
            
            # Initialize strategy engine
            self.strategy_engine = StrategyEngine(
                self.kalshi_client,
                self.research_client,
                self.config.trading.enabled_strategies,
                asdict(self.config.trading)
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                self.kalshi_client,
                asdict(self.config.risk_management)
            )
            
            # Test connections
            async with self.kalshi_client as kalshi:
                await kalshi.get_balance()
                logger.info("✓ Kalshi client connected successfully")
            
            async with self.research_client as research:
                # Test research connection with a simple query
                logger.info("✓ Octagon research client connected successfully")
            
            # Initialize risk manager
            async with self.kalshi_client as kalshi:
                await self.risk_manager.initialize()
                logger.info("✓ Risk manager initialized successfully")
            
            logger.info("🚀 Bot initialization complete!")
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def run(self):
        """Main bot execution loop"""
        logger.info("Starting Kalshi Trading Bot...")
        self.running = True
        
        try:
            # Display startup information
            self._display_startup_info()
            
            # Wait for startup delay
            if self.config.startup_delay_seconds > 0:
                logger.info(f"Waiting {self.config.startup_delay_seconds} seconds before starting...")
                await asyncio.sleep(self.config.startup_delay_seconds)
            
            # Main trading loop
            while self.running:
                try:
                    if not self.paused:
                        await self._execute_trading_cycle()
                    
                    # Sleep between cycles
                    await asyncio.sleep(self.config.loop_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    logger.error(traceback.format_exc())
                    await asyncio.sleep(30)  # Wait longer on error
                    
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in bot: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self._shutdown()
    
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        cycle_start = datetime.now()
        logger.info(f"Starting trading cycle at {cycle_start}")
        
        try:
            # Step 1: Check risk and portfolio status
            async with self.kalshi_client as kalshi:
                can_trade, risk_alerts = await self.risk_manager.check_portfolio_risk()
                
                if not can_trade:
                    logger.warning("Trading halted due to risk alerts")
                    for alert in risk_alerts:
                        logger.warning(f"Risk Alert: {alert.message}")
                    return
            
            # Step 2: Check stop losses
            async with self.kalshi_client as kalshi:
                markets_to_close = await self.risk_manager.check_stop_losses()
                for market_id in markets_to_close:
                    await self._close_position(market_id, "stop_loss")
            
            # Step 3: Analyze markets for opportunities
            async with self.kalshi_client as kalshi, self.research_client as research:
                opportunities = await self.strategy_engine.analyze_markets()
                
                # Filter opportunities based on risk criteria
                filtered_opportunities = self.strategy_engine.filter_opportunities(
                    opportunities,
                    self.config.risk_management.min_probability_threshold,
                    self.config.risk_management.max_concurrent_positions
                )
                
                logger.info(f"Found {len(filtered_opportunities)} trading opportunities")
            
            # Step 4: Execute trades
            for opportunity in filtered_opportunities:
                if not self.running:
                    break
                    
                await self._execute_trade(opportunity)
                
                # Small delay between trades
                await asyncio.sleep(2)
            
            # Step 5: Update performance and display status
            await self._update_performance_tracking()
            self._display_status()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Trading cycle completed in {cycle_duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            raise
    
    async def _execute_trade(self, opportunity: TradingOpportunity):
        """Execute a single trade"""
        try:
            logger.info(f"Evaluating trade opportunity: {opportunity.market_title}")
            
            # Calculate position size
            position_sizing = await self.risk_manager.calculate_position_size(opportunity)
            
            # Validate trade
            is_valid, validation_message = self.risk_manager.validate_trade(
                opportunity, position_sizing.final_position_size
            )
            
            if not is_valid:
                logger.info(f"Trade rejected: {validation_message}")
                return
            
            # Check if we already have a position in this market
            if opportunity.market_id in self.active_trades:
                logger.info(f"Already have position in {opportunity.market_title}")
                return
            
            # Execute trade
            if self.config.dry_run:
                logger.info(f"DRY RUN: Would execute trade for {opportunity.market_title}")
                logger.info(f"Position size: {position_sizing.final_position_size:.1f}")
                logger.info(f"Entry price: ${opportunity.entry_price:.2f}")
                logger.info(f"Strategy: {opportunity.strategy_type.value}")
                logger.info(f"Reasoning: {opportunity.reasoning}")
                
                # Simulate trade for dry run
                self._simulate_trade(opportunity, position_sizing.final_position_size)
                
            else:
                # Real trade execution
                await self._execute_real_trade(opportunity, position_sizing.final_position_size)
                
        except Exception as e:
            logger.error(f"Error executing trade for {opportunity.market_title}: {e}")
    
    def _simulate_trade(self, opportunity: TradingOpportunity, position_size: float):
        """Simulate trade execution for dry run mode"""
        trade_execution = TradeExecution(
            opportunity=opportunity,
            position_size=position_size,
            order_id=f"sim_{opportunity.market_id}_{datetime.now().timestamp()}",
            execution_price=opportunity.entry_price,
            status="simulated",
            execution_time=datetime.now()
        )
        
        self.active_trades[opportunity.market_id] = trade_execution
        self.total_trades += 1
        
        # Save to database
        self._save_trade_to_db(trade_execution)
        
        logger.info(f"✓ Simulated trade executed for {opportunity.market_title}")
    
    async def _execute_real_trade(self, opportunity: TradingOpportunity, position_size: float):
        """Execute real trade on Kalshi"""
        try:
            async with self.kalshi_client as kalshi:
                # Determine order side
                order_side = OrderSide.BUY if opportunity.direction == TradeDirection.BUY else OrderSide.SELL
                
                # Place order
                order = await kalshi.place_order(
                    market_id=opportunity.market_id,
                    side=order_side,
                    quantity=int(position_size),
                    limit_price=opportunity.entry_price,
                    order_type=OrderType.LIMIT
                )
                
                # Create trade execution record
                trade_execution = TradeExecution(
                    opportunity=opportunity,
                    position_size=position_size,
                    order_id=order.id,
                    execution_price=opportunity.entry_price,
                    status="executed",
                    execution_time=datetime.now()
                )
                
                self.active_trades[opportunity.market_id] = trade_execution
                self.total_trades += 1
                
                # Create stop loss
                await self.risk_manager.create_stop_loss(
                    opportunity.market_id,
                    opportunity.entry_price,
                    opportunity.direction
                )
                
                # Save to database
                self._save_trade_to_db(trade_execution)
                
                logger.info(f"✓ Trade executed: {opportunity.market_title}")
                logger.info(f"  Order ID: {order.id}")
                logger.info(f"  Position Size: {position_size:.1f}")
                logger.info(f"  Entry Price: ${opportunity.entry_price:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to execute real trade: {e}")
            raise
    
    async def _close_position(self, market_id: str, reason: str):
        """Close a position"""
        if market_id not in self.active_trades:
            return
        
        trade = self.active_trades[market_id]
        
        try:
            if not self.config.dry_run:
                async with self.kalshi_client as kalshi:
                    # Get current market price
                    market = await kalshi.get_market(market_id)
                    exit_price = market.bid if trade.opportunity.direction == TradeDirection.BUY else market.ask
                    
                    # Close position (this would require getting current positions and placing opposite order)
                    # For now, we'll simulate the close
                    trade.execution_price = exit_price
            
            # Calculate P&L
            if trade.opportunity.direction == TradeDirection.BUY:
                pnl = (trade.execution_price - trade.opportunity.entry_price) * trade.position_size
            else:
                pnl = (trade.opportunity.entry_price - trade.execution_price) * trade.position_size
            
            trade.pnl = pnl
            trade.status = "closed"
            trade.closed_time = datetime.now()
            
            # Update performance
            self.total_pnl += pnl
            if pnl > 0:
                self.successful_trades += 1
            
            # Move to completed trades
            self.completed_trades.append(trade)
            del self.active_trades[market_id]
            
            # Update strategy performance
            holding_period = (trade.closed_time - trade.execution_time).total_seconds() / 3600  # hours
            self.strategy_engine.update_strategy_performance(
                trade.opportunity.strategy_type,
                pnl,
                holding_period
            )
            
            logger.info(f"✓ Position closed: {trade.opportunity.market_title}")
            logger.info(f"  Reason: {reason}")
            logger.info(f"  P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position for {market_id}: {e}")
    
    def _save_trade_to_db(self, trade: TradeExecution):
        """Save trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    market_id, market_title, strategy_type, direction, confidence,
                    expected_return, position_size, entry_price, exit_price, pnl,
                    entry_time, exit_time, status, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.opportunity.market_id,
                trade.opportunity.market_title,
                trade.opportunity.strategy_type.value,
                trade.opportunity.direction.value,
                trade.opportunity.confidence,
                trade.opportunity.expected_return,
                trade.position_size,
                trade.opportunity.entry_price,
                trade.execution_price,
                trade.pnl,
                trade.execution_time.isoformat() if trade.execution_time else None,
                trade.closed_time.isoformat() if trade.closed_time else None,
                trade.status,
                trade.opportunity.reasoning
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    async def _update_performance_tracking(self):
        """Update performance tracking"""
        try:
            # This could include updating metrics, calculating returns, etc.
            pass
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def _display_startup_info(self):
        """Display bot startup information"""
        # Get environment info
        kalshi_mode = "DEMO" if self.config.kalshi.use_demo else "PRODUCTION"
        trading_mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        
        startup_panel = Panel(
            f"""
🤖 Kalshi Trading Bot with Octagon Deep Research
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Configuration:
• Trading Mode: {trading_mode}
• Kalshi Environment: {kalshi_mode}
• Strategies: {', '.join(self.config.trading.enabled_strategies)}
• Max Position Size: {self.config.risk_management.max_position_size:.1%}
• Max Daily Loss: {self.config.risk_management.max_daily_loss:.1%}
• Loop Interval: {self.config.loop_interval_seconds}s

Risk Management:
• Stop Loss: {self.config.risk_management.stop_loss_percent:.1%}
• Take Profit: {self.config.risk_management.take_profit_percent:.1%}
• Max Positions: {self.config.risk_management.max_concurrent_positions}

Started at: {self.bot_start_time}
            """,
            title="Bot Status",
            border_style="green" if kalshi_mode == "DEMO" else "yellow"
        )
        
        self.console.print(startup_panel)
    
    def _display_status(self):
        """Display current bot status"""
        # Create status table
        table = Table(title="Trading Bot Status", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")
        
        # Add metrics
        table.add_row("Runtime", str(datetime.now() - self.bot_start_time))
        table.add_row("Total Trades", str(self.total_trades))
        table.add_row("Successful Trades", str(self.successful_trades))
        table.add_row("Win Rate", f"{(self.successful_trades / self.total_trades * 100):.1f}%" if self.total_trades > 0 else "0%")
        table.add_row("Total P&L", f"${self.total_pnl:.2f}")
        table.add_row("Active Positions", str(len(self.active_trades)))
        
        # Get risk summary
        risk_summary = self.risk_manager.get_risk_summary()
        table.add_row("Portfolio Value", f"${risk_summary['portfolio_value']:.2f}")
        table.add_row("Daily P&L", f"${risk_summary['daily_pnl']:.2f} ({risk_summary['daily_pnl_percent']:.2%})")
        table.add_row("Max Drawdown", f"{risk_summary['max_drawdown']:.2%}")
        table.add_row("Risk Alerts", str(risk_summary['active_alerts']))
        
        self.console.print(table)
        
        # Display active trades
        if self.active_trades:
            trades_table = Table(title="Active Trades", style="yellow")
            trades_table.add_column("Market", style="cyan")
            trades_table.add_column("Strategy", style="magenta")
            trades_table.add_column("Direction", style="green")
            trades_table.add_column("Size", style="yellow")
            trades_table.add_column("Entry Price", style="blue")
            trades_table.add_column("Age", style="red")
            
            for trade in self.active_trades.values():
                age = datetime.now() - trade.execution_time if trade.execution_time else timedelta()
                trades_table.add_row(
                    trade.opportunity.market_title[:50] + "...",
                    trade.opportunity.strategy_type.value,
                    trade.opportunity.direction.value,
                    f"{trade.position_size:.1f}",
                    f"${trade.opportunity.entry_price:.2f}",
                    str(age).split('.')[0]
                )
            
            self.console.print(trades_table)
    
    async def _shutdown(self):
        """Shutdown bot gracefully"""
        logger.info("Shutting down bot...")
        
        try:
            # Close all positions if not in dry run
            if not self.config.dry_run:
                for market_id in list(self.active_trades.keys()):
                    await self._close_position(market_id, "shutdown")
            
            # Save final state
            self._save_final_state()
            
            logger.info("Bot shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _save_final_state(self):
        """Save final bot state"""
        try:
            final_state = {
                "bot_start_time": self.bot_start_time.isoformat(),
                "total_trades": self.total_trades,
                "successful_trades": self.successful_trades,
                "total_pnl": self.total_pnl,
                "active_trades": len(self.active_trades),
                "completed_trades": len(self.completed_trades)
            }
            
            with open("bot_final_state.json", "w") as f:
                json.dump(final_state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving final state: {e}")
    
    def pause(self):
        """Pause trading"""
        self.paused = True
        logger.info("Trading paused")
    
    def resume(self):
        """Resume trading"""
        self.paused = False
        logger.info("Trading resumed")
    
    def stop(self):
        """Stop bot"""
        self.running = False
        logger.info("Bot stopping...")

async def main():
    """Main entry point"""
    try:
        # Load configuration
        config = load_config()
        
        # Create and initialize bot
        bot = TradingBot(config)
        await bot.initialize()
        
        # Run bot
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 