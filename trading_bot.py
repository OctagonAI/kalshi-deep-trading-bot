"""
Simple Kalshi trading bot with Octagon research and OpenAI decision making.
"""
import asyncio
import json
from typing import List, Dict, Any
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import openai

from config import load_config
from kalshi_client import KalshiClient
from research_client import OctagonClient
from betting_models import BettingDecision, MarketAnalysis


class SimpleTradingBot:
    """Simple trading bot that follows a clear workflow."""
    
    def __init__(self):
        self.config = load_config()
        self.console = Console()
        self.kalshi_client = None
        self.research_client = None
        self.openai_client = None
        
    async def initialize(self):
        """Initialize all API clients."""
        self.console.print("[bold blue]Initializing trading bot...[/bold blue]")
        
        # Initialize clients
        self.kalshi_client = KalshiClient(self.config.kalshi)
        self.research_client = OctagonClient(self.config.octagon)
        self.openai_client = openai.AsyncOpenAI(api_key=self.config.openai.api_key)
        
        # Test connections
        await self.kalshi_client.login()
        self.console.print("[green]✓ Kalshi API connected[/green]")
        self.console.print("[green]✓ Octagon API ready[/green]")
        self.console.print("[green]✓ OpenAI API ready[/green]")
        
        # Show environment info
        env_color = "green" if self.config.kalshi.use_demo else "yellow"
        env_name = "DEMO" if self.config.kalshi.use_demo else "PRODUCTION"
        mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        
        self.console.print(f"\n[{env_color}]Environment: {env_name}[/{env_color}]")
        self.console.print(f"[blue]Mode: {mode}[/blue]")
        self.console.print(f"[blue]Max markets: {self.config.max_markets}[/blue]")
        self.console.print(f"[blue]Max bet amount: ${self.config.max_bet_amount}[/blue]\n")
    
    async def get_active_markets(self) -> List[Dict[str, Any]]:
        """Get active markets sorted by volume."""
        self.console.print("[bold]Step 1: Fetching active markets...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching markets...", total=None)
            
            try:
                markets = await self.kalshi_client.get_markets()
                
                # Filter for active markets and sort by volume
                active_markets = [
                    market for market in markets 
                    if market.get('status') == 'active'
                ]
                
                # Sort by volume (descending)
                active_markets.sort(
                    key=lambda x: x.get('volume', 0), 
                    reverse=True
                )
                
                # Limit to max_markets
                limited_markets = active_markets[:self.config.max_markets]
                
                self.console.print(f"[green]✓ Found {len(limited_markets)} active markets[/green]")
                
                # Show top 10 markets
                table = Table(title="Top 10 Markets by Volume")
                table.add_column("Ticker", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("Volume", style="magenta", justify="right")
                table.add_column("Status", style="green")
                
                for market in limited_markets[:10]:
                    table.add_row(
                        market.get('ticker', 'N/A'),
                        market.get('title', 'N/A')[:50] + "...",
                        str(market.get('volume', 0)),
                        market.get('status', 'N/A')
                    )
                
                self.console.print(table)
                return limited_markets
                
            except Exception as e:
                self.console.print(f"[red]Error fetching markets: {e}[/red]")
                return []
    
    async def research_markets(self, markets: List[Dict[str, Any]]) -> Dict[str, str]:
        """Research each market using Octagon Deep Research."""
        self.console.print(f"\n[bold]Step 2: Researching {len(markets)} markets...[/bold]")
        
        research_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Researching markets...", total=len(markets))
            
            # Research markets in batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(markets), batch_size):
                batch = markets[i:i + batch_size]
                
                # Research batch in parallel
                tasks = []
                for market in batch:
                    ticker = market.get('ticker', '')
                    title = market.get('title', '')
                    if ticker and title:
                        tasks.append(self.research_client.research_market(title, ticker))
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for market, result in zip(batch, results):
                        ticker = market.get('ticker', '')
                        if not isinstance(result, Exception):
                            research_results[ticker] = result
                            progress.update(task, advance=1)
                            self.console.print(f"[green]✓ Researched {ticker}[/green]")
                        else:
                            self.console.print(f"[red]✗ Failed to research {ticker}: {result}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch research error: {e}[/red]")
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(1)
        
        self.console.print(f"[green]✓ Completed research on {len(research_results)} markets[/green]")
        return research_results
    
    async def get_betting_decisions(self, markets: List[Dict[str, Any]], 
                                   research_results: Dict[str, str]) -> MarketAnalysis:
        """Use OpenAI to make structured betting decisions."""
        self.console.print(f"\n[bold]Step 3: Generating betting decisions...[/bold]")
        
        # Prepare market and research data for OpenAI
        market_data = []
        for market in markets:
            ticker = market.get('ticker', '')
            if ticker in research_results:
                market_data.append({
                    'ticker': ticker,
                    'title': market.get('title', ''),
                    'volume': market.get('volume', 0),
                    'yes_price': market.get('yes_bid', 0),
                    'no_price': market.get('no_bid', 0),
                    'research': research_results[ticker]
                })
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a professional prediction market trader. Based on the research provided, 
        make betting decisions for each market.
        
        Available budget: ${self.config.max_bet_amount * len(market_data)}
        Max bet per market: ${self.config.max_bet_amount}
        
        Markets and Research:
        {json.dumps(market_data, indent=2)}
        
        For each market, decide:
        1. Action: "buy_yes", "buy_no", or "skip"
        2. Confidence: 0-1 (only bet if confidence > 0.6)
        3. Amount: How much to bet (max ${self.config.max_bet_amount})
        4. Reasoning: Brief explanation
        
        Focus on:
        - High-volume markets with good research insights
        - Clear directional signals from the research
        - Risk management (don't bet on everything)
        - Diversification across different market types
        
        Return your analysis in the specified JSON format.
        """
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating betting decisions...", total=None)
            
            try:
                response = await self.openai_client.beta.chat.completions.parse(
                    model=self.config.openai.model,
                    messages=[
                        {"role": "system", "content": "You are a professional prediction market trader."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=MarketAnalysis,
                    temperature=0.1
                )
                
                analysis = response.choices[0].parsed
                
                # Show decision summary
                self.console.print(f"[green]✓ Generated {len(analysis.decisions)} betting decisions[/green]")
                
                # Display decisions table
                table = Table(title="Betting Decisions")
                table.add_column("Ticker", style="cyan")
                table.add_column("Action", style="yellow")
                table.add_column("Confidence", style="magenta", justify="right")
                table.add_column("Amount", style="green", justify="right")
                table.add_column("Reasoning", style="blue")
                
                for decision in analysis.decisions:
                    if decision.action != "skip":
                        table.add_row(
                            decision.ticker,
                            decision.action,
                            f"{decision.confidence:.2f}",
                            f"${decision.amount:.2f}",
                            decision.reasoning[:50] + "..."
                        )
                
                self.console.print(table)
                
                # Show summary
                self.console.print(f"\n[blue]Total recommended bet: ${analysis.total_recommended_bet:.2f}[/blue]")
                self.console.print(f"[blue]High confidence bets: {analysis.high_confidence_bets}[/blue]")
                self.console.print(f"[blue]Strategy: {analysis.summary}[/blue]")
                
                return analysis
                
            except Exception as e:
                self.console.print(f"[red]Error generating decisions: {e}[/red]")
                # Return empty analysis
                return MarketAnalysis(
                    decisions=[],
                    total_recommended_bet=0.0,
                    high_confidence_bets=0,
                    summary="Error generating decisions"
                )
    
    async def place_bets(self, analysis: MarketAnalysis):
        """Place bets based on the analysis."""
        self.console.print(f"\n[bold]Step 4: Placing bets...[/bold]")
        
        if not analysis.decisions:
            self.console.print("[yellow]No betting decisions to execute[/yellow]")
            return
        
        # Filter to only actionable decisions
        actionable_decisions = [
            decision for decision in analysis.decisions 
            if decision.action != "skip" and decision.amount > 0
        ]
        
        if not actionable_decisions:
            self.console.print("[yellow]No actionable bets to place[/yellow]")
            return
        
        if self.config.dry_run:
            self.console.print("[yellow]DRY RUN MODE - No actual bets will be placed[/yellow]")
        
        placed_bets = 0
        total_bet = 0.0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Placing bets...", total=len(actionable_decisions))
            
            for decision in actionable_decisions:
                try:
                    if self.config.dry_run:
                        # Simulate bet placement
                        self.console.print(f"[yellow]DRY RUN: Would place {decision.action} bet on {decision.ticker} for ${decision.amount:.2f}[/yellow]")
                        placed_bets += 1
                        total_bet += decision.amount
                    else:
                        # Place actual bet
                        side = "yes" if decision.action == "buy_yes" else "no"
                        result = await self.kalshi_client.place_order(
                            ticker=decision.ticker,
                            side=side,
                            amount=decision.amount
                        )
                        
                        if result.get('success', False):
                            self.console.print(f"[green]✓ Placed {decision.action} bet on {decision.ticker} for ${decision.amount:.2f}[/green]")
                            placed_bets += 1
                            total_bet += decision.amount
                        else:
                            self.console.print(f"[red]✗ Failed to place bet on {decision.ticker}: {result.get('error', 'Unknown error')}[/red]")
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    self.console.print(f"[red]Error placing bet on {decision.ticker}: {e}[/red]")
                    progress.update(task, advance=1)
                
                # Brief pause between bets
                await asyncio.sleep(0.5)
        
        # Summary
        self.console.print(f"\n[green]✓ Successfully placed {placed_bets} bets[/green]")
        self.console.print(f"[green]✓ Total amount bet: ${total_bet:.2f}[/green]")
    
    async def run(self):
        """Main bot execution."""
        try:
            await self.initialize()
            
            # Execute the main workflow
            markets = await self.get_active_markets()
            if not markets:
                self.console.print("[red]No markets found. Exiting.[/red]")
                return
            
            research_results = await self.research_markets(markets)
            if not research_results:
                self.console.print("[red]No research results. Exiting.[/red]")
                return
            
            analysis = await self.get_betting_decisions(markets, research_results)
            await self.place_bets(analysis)
            
            self.console.print("\n[bold green]Bot execution completed![/bold green]")
            
        except Exception as e:
            self.console.print(f"[red]Bot execution error: {e}[/red]")
            logger.exception("Bot execution failed")
        
        finally:
            # Clean up
            if self.research_client:
                await self.research_client.close()
            if self.kalshi_client:
                await self.kalshi_client.close()


async def main():
    """Main entry point."""
    bot = SimpleTradingBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main()) 