"""
Simple Kalshi trading bot with Octagon research and OpenAI decision making.
"""
import asyncio
import json
import re
import sys
import argparse
from typing import List, Dict, Any, Optional
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
        self.console.print(f"[blue]Max events to analyze: {self.config.max_events_to_analyze}[/blue]")
        self.console.print(f"[blue]Research batch size: {self.config.research_batch_size}[/blue]")
        self.console.print(f"[blue]Max bet amount: ${self.config.max_bet_amount}[/blue]\n")
    
    async def get_top_events(self) -> List[Dict[str, Any]]:
        """Get top events sorted by 24-hour volume."""
        self.console.print("[bold]Step 1: Fetching top events...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching events...", total=None)
            
            try:
                events = await self.kalshi_client.get_events(limit=self.config.max_events_to_analyze)
                
                self.console.print(f"[green]✓ Found {len(events)} events[/green]")
                
                # Show top 10 events
                table = Table(title="Top 10 Events by 24h Volume")
                table.add_column("Event Ticker", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("24h Volume", style="magenta", justify="right")
                table.add_column("Category", style="green")
                
                for event in events[:10]:
                    table.add_row(
                        event.get('event_ticker', 'N/A'),
                        event.get('title', 'N/A')[:50] + ("..." if len(event.get('title', '')) > 50 else ""),
                        f"{event.get('volume_24h', 0):,}",
                        event.get('category', 'N/A')
                    )
                
                self.console.print(table)
                return events
                
            except Exception as e:
                self.console.print(f"[red]Error fetching events: {e}[/red]")
                return []
    
    async def get_markets_for_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get all markets for each event."""
        self.console.print(f"\n[bold]Step 2: Fetching markets for {len(events)} events...[/bold]")
        
        event_markets = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching markets...", total=len(events))
            
            for event in events:
                event_ticker = event.get('event_ticker', '')
                if not event_ticker:
                    progress.update(task, advance=1)
                    continue
                
                try:
                    markets = await self.kalshi_client.get_markets_for_event(event_ticker)
                    
                    if markets:
                        event_markets[event_ticker] = {
                            'event': event,
                            'markets': markets
                        }
                        self.console.print(f"[green]✓ Found {len(markets)} markets for {event_ticker}[/green]")
                    else:
                        self.console.print(f"[yellow]⚠ No markets found for {event_ticker}[/yellow]")
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    self.console.print(f"[red]✗ Failed to get markets for {event_ticker}: {e}[/red]")
                    progress.update(task, advance=1)
                
                # Brief pause between requests
                await asyncio.sleep(0.1)
        
        total_markets = sum(len(data['markets']) for data in event_markets.values())
        self.console.print(f"[green]✓ Found {total_markets} total markets across {len(event_markets)} events[/green]")
        return event_markets
    
    def _parse_probabilities_from_research(self, research_text: str, markets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parse probability predictions from Octagon research text."""
        probabilities = {}
        
        # Look for patterns like "TICKER: 75%" or "Market TICKER has 65% probability"
        for market in markets:
            ticker = market.get('ticker', '')
            if not ticker:
                continue
                
            # Try different patterns to find probability for this ticker
            patterns = [
                rf"{re.escape(ticker)}[:\s]*(\d+)%",
                rf"{re.escape(ticker)}[:\s]*(\d+\.?\d*)%",
                rf"probability.*{re.escape(ticker)}[:\s]*(\d+\.?\d*)%",
                rf"{re.escape(ticker)}.*probability.*?(\d+\.?\d*)%",
                rf"{re.escape(ticker)}.*(\d+\.?\d*)%.*probability",
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, research_text, re.IGNORECASE)
                if matches:
                    try:
                        prob = float(matches[0])
                        if 0 <= prob <= 100:
                            probabilities[ticker] = prob
                            break
                    except ValueError:
                        continue
        
        return probabilities
    
    async def research_events(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Research each event and its markets using Octagon Deep Research."""
        self.console.print(f"\n[bold]Step 3: Researching {len(event_markets)} events...[/bold]")
        
        research_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Researching events...", total=len(event_markets))
            
            # Research events in batches to avoid rate limits
            batch_size = self.config.research_batch_size
            event_items = list(event_markets.items())
            
            for i in range(0, len(event_items), batch_size):
                batch = event_items[i:i + batch_size]
                
                # Research batch in parallel
                tasks = []
                for event_ticker, data in batch:
                    event = data['event']
                    markets = data['markets']
                    if event and markets:
                        tasks.append(self.research_client.research_event(event, markets))
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for (event_ticker, data), result in zip(batch, results):
                        if not isinstance(result, Exception):
                            research_results[event_ticker] = result
                            progress.update(task, advance=1)
                            self.console.print(f"[green]✓ Researched {event_ticker}[/green]")
                            
                            # Parse and display probabilities
                            probabilities = self._parse_probabilities_from_research(result, data['markets'])
                            if probabilities:
                                self.console.print(f"[blue]Predicted probabilities for {event_ticker}:[/blue]")
                                for ticker, prob in probabilities.items():
                                    self.console.print(f"  {ticker}: {prob:.1f}%")
                            
                        else:
                            self.console.print(f"[red]✗ Failed to research {event_ticker}: {result}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch research error: {e}[/red]")
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(1)
        
        self.console.print(f"[green]✓ Completed research on {len(research_results)} events[/green]")
        
        # Show summary table of all predicted probabilities
        self._display_probability_summary(research_results, event_markets)
        
        return research_results
    
    def _display_probability_summary(self, research_results: Dict[str, str], event_markets: Dict[str, Dict[str, Any]]):
        """Display a summary table of all predicted probabilities."""
        table = Table(title="Octagon Probability Predictions Summary")
        table.add_column("Event", style="cyan")
        table.add_column("Market", style="yellow")
        table.add_column("Ticker", style="magenta")
        table.add_column("Predicted Probability", style="green", justify="right")
        
        for event_ticker, research_text in research_results.items():
            if event_ticker in event_markets:
                markets = event_markets[event_ticker]['markets']
                probabilities = self._parse_probabilities_from_research(research_text, markets)
                
                for market in markets:
                    ticker = market.get('ticker', '')
                    prob = probabilities.get(ticker, 0)
                    
                    table.add_row(
                        event_ticker,
                        market.get('title', 'N/A')[:40] + ("..." if len(market.get('title', '')) > 40 else ""),
                        ticker,
                        f"{prob:.1f}%" if prob > 0 else "N/A"
                    )
        
        self.console.print(table)
    
    async def get_betting_decisions(self, event_markets: Dict[str, Dict[str, Any]], 
                                   research_results: Dict[str, str]) -> MarketAnalysis:
        """Use OpenAI to make structured betting decisions."""
        self.console.print(f"\n[bold]Step 4: Generating betting decisions...[/bold]")
        
        # Prepare event and research data for OpenAI
        event_data = []
        for event_ticker, data in event_markets.items():
            if event_ticker in research_results:
                event_info = data['event']
                markets = data['markets']
                
                event_data.append({
                    'event_ticker': event_ticker,
                    'event_title': event_info.get('title', ''),
                    'event_category': event_info.get('category', ''),
                    'event_volume': event_info.get('volume', 0),
                    'markets': [
                        {
                            'ticker': market.get('ticker', ''),
                            'title': market.get('title', ''),
                            'volume': market.get('volume', 0)
                        }
                        for market in markets
                    ],
                    'research': research_results[event_ticker]
                })
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a professional prediction market trader. Based on the research provided for each event, 
        make betting decisions for the individual markets.
        
        Available budget: ${self.config.max_bet_amount * sum(len(data['markets']) for data in event_data)}
        Max bet per market: ${self.config.max_bet_amount}
        
        Events and Research:
        {json.dumps(event_data, indent=2)}
        
        For each market, decide:
        1. Action: "buy_yes", "buy_no", or "skip"
        2. Confidence: 0-1 (only bet if confidence > 0.6)
        3. Amount: How much to bet (max ${self.config.max_bet_amount})
        4. Reasoning: Brief explanation based on the research
        
        Focus on:
        - Markets with clear research insights and probability predictions
        - High-volume events with good research confidence
        - Risk management (don't bet on everything)
        - Consider correlations within mutually exclusive events
        
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
        self.console.print(f"\n[bold]Step 5: Placing bets...[/bold]")
        
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
        
        # Display bets to be placed with reasoning
        self.console.print("\n[bold]Bets to be placed:[/bold]")
        for decision in actionable_decisions:
            self.console.print(f"[cyan]{decision.ticker}[/cyan]: {decision.action} ${decision.amount:.2f}")
            self.console.print(f"  [blue]Reasoning: {decision.reasoning}[/blue]")
            self.console.print(f"  [magenta]Confidence: {decision.confidence:.2f}[/magenta]\n")
        
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
                        self.console.print(f"[yellow]  Reasoning: {decision.reasoning}[/yellow]")
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
                            self.console.print(f"[green]  Reasoning: {decision.reasoning}[/green]")
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
            events = await self.get_top_events()
            if not events:
                self.console.print("[red]No events found. Exiting.[/red]")
                return
            
            event_markets = await self.get_markets_for_events(events)
            if not event_markets:
                self.console.print("[red]No markets found. Exiting.[/red]")
                return
            
            research_results = await self.research_events(event_markets)
            if not research_results:
                self.console.print("[red]No research results. Exiting.[/red]")
                return
            
            analysis = await self.get_betting_decisions(event_markets, research_results)
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


def cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Kalshi trading bot with Octagon research and OpenAI decision making",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run trading-bot                    # Run bot with default settings
  uv run trading-bot --help            # Show this help message
  
Configuration:
  Create a .env file with your API keys:
    KALSHI_API_KEY=your_kalshi_api_key
    KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----"
    OCTAGON_API_KEY=your_octagon_api_key
    OPENAI_API_KEY=your_openai_api_key
    
  Optional settings:
    KALSHI_USE_DEMO=true               # Use demo environment (default: true)
    DRY_RUN=true                       # Simulate trades (default: true)
    MAX_MARKETS=50                     # Max events to process (default: 50)
    MAX_BET_AMOUNT=25.0                # Max bet per market (default: 25.0)
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Kalshi Trading Bot 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Try to load config and run bot
    try:
        asyncio.run(main())
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Please check your .env file configuration.[/yellow]")
        console.print("[yellow]Run with --help for more information.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    cli() 