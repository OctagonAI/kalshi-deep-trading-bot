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
from betting_models import BettingDecision, MarketAnalysis, MarketProbability, ProbabilityExtraction


class SimpleTradingBot:
    """Simple trading bot that follows a clear workflow."""
    
    def __init__(self, live_trading: bool = False):
        self.config = load_config()
        # Override dry_run based on CLI parameter
        self.config.dry_run = not live_trading
        self.console = Console()
        self.kalshi_client = None
        self.research_client = None
        self.openai_client = None
        
    async def initialize(self):
        """Initialize all API clients."""
        self.console.print("[bold blue]Initializing trading bot...[/bold blue]")
        
        # Initialize clients
        self.kalshi_client = KalshiClient(self.config.kalshi, self.config.minimum_time_remaining_hours, self.config.max_markets_per_event)
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
        self.console.print(f"[blue]Skip existing positions: {self.config.skip_existing_positions}[/blue]")
        self.console.print(f"[blue]Minimum time to event strike: {self.config.minimum_time_remaining_hours} hours (for events with strike_date)[/blue]")
        self.console.print(f"[blue]Max markets per event: {self.config.max_markets_per_event}[/blue]")
        self.console.print(f"[blue]Max bet amount: ${self.config.max_bet_amount}[/blue]")
        self.console.print(f"[blue]Minimum alpha threshold: {self.config.minimum_alpha_threshold}x[/blue]\n")
    
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
                # Get a larger pool of events to ensure we have enough after filtering positions
                # Use 3x the target amount to account for events with existing positions
                fetch_limit = self.config.max_events_to_analyze * 3
                events = await self.kalshi_client.get_events(limit=fetch_limit)
                self.console.print(f"[blue]• Fetched {len(events)} events (will filter to top {self.config.max_events_to_analyze} after position filtering)[/blue]")
                
                self.console.print(f"[green]✓ Found {len(events)} events[/green]")
                
                # Show top 10 events
                table = Table(title="Top 10 Events by 24h Volume")
                table.add_column("Event Ticker", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("24h Volume", style="magenta", justify="right")
                table.add_column("Time Remaining", style="blue", justify="right")
                table.add_column("Category", style="green")
                table.add_column("Mutually Exclusive", style="red", justify="center")
                
                for event in events[:10]:
                    time_remaining = event.get('time_remaining_hours')
                    if time_remaining is None:
                        time_str = "No date set"
                    elif time_remaining > 24:
                        time_str = f"{time_remaining/24:.1f} days"
                    else:
                        time_str = f"{time_remaining:.1f} hours"
                    
                    table.add_row(
                        event.get('event_ticker', 'N/A'),
                        event.get('title', 'N/A')[:35] + ("..." if len(event.get('title', '')) > 35 else ""),
                        f"{event.get('volume_24h', 0):,}",
                        time_str,
                        event.get('category', 'N/A'),
                        "YES" if event.get('mutually_exclusive', False) else "NO"
                    )
                
                self.console.print(table)
                return events
                
            except Exception as e:
                self.console.print(f"[red]Error fetching events: {e}[/red]")
                return []
    
    async def get_markets_for_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get markets for each event (uses pre-loaded markets from events)."""
        self.console.print(f"\n[bold]Step 2: Processing markets for {len(events)} events...[/bold]")
        
        event_markets = {}
        
        for event in events:
            event_ticker = event.get('event_ticker', '')
            if not event_ticker:
                continue
            
            # Use pre-loaded markets from the event data
            markets = event.get('markets', [])
            total_markets = event.get('total_markets', len(markets))
            
            if markets:
                # Convert to the format expected by the rest of the system
                simple_markets = []
                for market in markets:
                    simple_markets.append({
                        "ticker": market.get("ticker", ""),
                        "title": market.get("title", ""),
                        "subtitle": market.get("subtitle", ""),
                        "volume": market.get("volume", 0),
                        "open_time": market.get("open_time", ""),
                        "close_time": market.get("close_time", ""),
                    })
                
                event_markets[event_ticker] = {
                    'event': event,
                    'markets': simple_markets
                }
                
                if total_markets > len(markets):
                    self.console.print(f"[green]✓ Using top {len(markets)} markets for {event_ticker} (from {total_markets} total)[/green]")
                else:
                    self.console.print(f"[green]✓ Using {len(markets)} markets for {event_ticker}[/green]")
            else:
                self.console.print(f"[yellow]⚠ No markets found for {event_ticker}[/yellow]")
        
        total_markets = sum(len(data['markets']) for data in event_markets.values())
        self.console.print(f"[green]✓ Processing {total_markets} total markets across {len(event_markets)} events[/green]")
        return event_markets
    
    async def filter_markets_by_positions(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter out markets where we already have positions to save research time."""
        if self.config.dry_run or not self.config.skip_existing_positions:
            # Skip position filtering in dry run mode or if disabled
            return event_markets
            
        self.console.print(f"\n[bold]Step 2.5: Filtering markets by existing positions...[/bold]")
        
        filtered_event_markets = {}
        total_markets_before = 0
        total_markets_after = 0
        skipped_markets = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Count total markets for progress
            total_markets = sum(len(data['markets']) for data in event_markets.values())
            task = progress.add_task("Checking existing positions...", total=total_markets)
            
            for event_ticker, data in event_markets.items():
                event = data['event']
                markets = data['markets']
                total_markets_before += len(markets)
                
                # Check if we have positions in ANY market of this event
                event_has_positions = False
                markets_checked = 0
                
                for market in markets:
                    ticker = market.get('ticker', '')
                    if not ticker:
                        progress.update(task, advance=1)
                        markets_checked += 1
                        continue
                        
                    try:
                        # Check if we already have a position in this market
                        has_position = await self.kalshi_client.has_position_in_market(ticker)
                        if has_position:
                            self.console.print(f"[yellow]⚠ Found position in {ticker}[/yellow]")
                            event_has_positions = True
                            # Update progress for remaining unchecked markets in this event
                            remaining_markets = len(markets) - markets_checked - 1
                            progress.update(task, advance=remaining_markets + 1)
                            break  # No need to check other markets in this event
                            
                    except Exception as e:
                        logger.warning(f"Could not check position for {ticker}: {e}")
                        # If we can't check, assume no position and continue checking other markets
                    
                    progress.update(task, advance=1)
                    markets_checked += 1
                
                if event_has_positions:
                    skipped_markets += len(markets)  # Count all markets in event as skipped
                    self.console.print(f"[yellow]⚠ Skipping entire event {event_ticker}: Has existing positions[/yellow]")
                else:
                    # No positions found, keep the entire event
                    filtered_event_markets[event_ticker] = {
                        'event': event,
                        'markets': markets
                    }
                    total_markets_after += len(markets)
                    self.console.print(f"[green]✓ Keeping entire event {event_ticker}: No existing positions[/green]")
        
        # Show filtering summary
        events_skipped = len(event_markets) - len(filtered_event_markets)
        self.console.print(f"\n[blue]Position filtering summary:[/blue]")
        self.console.print(f"[blue]• Events before filtering: {len(event_markets)}[/blue]")
        self.console.print(f"[blue]• Events after filtering: {len(filtered_event_markets)}[/blue]")
        self.console.print(f"[blue]• Events skipped (existing positions): {events_skipped}[/blue]")
        self.console.print(f"[blue]• Markets in skipped events: {skipped_markets}[/blue]")
        self.console.print(f"[blue]• Markets remaining for research: {total_markets_after}[/blue]")
        
        if len(filtered_event_markets) == 0:
            self.console.print("[yellow]⚠ No events remaining after position filtering[/yellow]")
        elif events_skipped > 0:
            time_saved_estimate = events_skipped * 3  # Rough estimate: 3 minutes per event research
            self.console.print(f"[green]✓ Estimated time saved by skipping research: ~{time_saved_estimate} minutes[/green]")
            
        return filtered_event_markets

    def _parse_probabilities_from_research(self, research_text: str, markets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parse probability predictions from Octagon research text."""
        probabilities = {}
        
        # Look for patterns with both ticker names and market titles
        for market in markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            if not ticker:
                continue
                
            # Try different patterns to find probability for this market
            # Look for both ticker and title patterns
            search_terms = [ticker]
            if title:
                # Add key words from the title for better matching
                search_terms.append(title)
                # Extract key identifying words (avoid common words)
                title_words = [w for w in title.split() if len(w) > 3 and w.lower() not in ['will', 'the', 'win', 'be', 'a', 'be', 'of', 'and', 'or', 'for', 'to', 'in', 'on', 'at', 'with', 'from', 'by', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once']]
                search_terms.extend(title_words)
            
            found_probability = None
            for term in search_terms:
                if not term:
                    continue
                    
                # Try different patterns to find probability for this term
                patterns = [
                    rf"{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}[:\s]*(\d+)%",
                    rf"(\d+\.?\d*)%[:\s]*{re.escape(term)}",
                    rf"(\d+)%[:\s]*{re.escape(term)}",
                    rf"probability.*{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*probability.*?(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*(\d+\.?\d*)%.*probability",
                    rf"probability.*(\d+\.?\d*)%.*{re.escape(term)}",
                    # More flexible patterns for natural language
                    rf"{re.escape(term)}.*?(\d+\.?\d*)%",
                    rf"(\d+\.?\d*)%.*?{re.escape(term)}",
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, research_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        try:
                            prob = float(matches[0])
                            if 0 <= prob <= 100:
                                found_probability = prob
                                break
                        except ValueError:
                            continue
                
                if found_probability is not None:
                    break
            
            if found_probability is not None:
                probabilities[ticker] = found_probability
                logger.info(f"Found probability for {ticker}: {found_probability}%")
            else:
                logger.warning(f"No probability found for {ticker} (title: {title})")
                # Show first 200 chars of research text for debugging
                sample_text = research_text[:200].replace('\n', ' ')
                logger.debug(f"Research sample: {sample_text}...")
             
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
                            
                            
                        else:
                            self.console.print(f"[red]✗ Failed to research {event_ticker}: {result}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch research error: {e}[/red]")
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(1)
        
        self.console.print(f"[green]✓ Completed research on {len(research_results)} events[/green]")
    
        
        return research_results
    
    async def _extract_probabilities_for_event(self, event_ticker: str, research_text: str, 
                                              event_markets: Dict[str, Dict[str, Any]]) -> tuple[str, Optional[ProbabilityExtraction]]:
        """Extract probabilities for a single event."""
        try:
            # Get market information for this event
            event_data = event_markets.get(event_ticker, {})
            markets = event_data.get('markets', [])
            event_info = event_data.get('event', {})
            
            # Prepare market information for the prompt
            market_info = []
            for market in markets:
                market_info.append({
                    'ticker': market.get('ticker', ''),
                    'title': market.get('title', ''),
                    'yes_mid_price': market.get('yes_mid_price', 0),
                    'no_mid_price': market.get('no_mid_price', 0)
                })
            
            # Create prompt for probability extraction
            prompt = f"""
            Based on the following deep research, extract the probability estimates for each market.
            
            Event: {event_info.get('title', event_ticker)}
            
            Markets:
            {json.dumps(market_info, indent=2)}
            
            Research Results:
            {research_text}
            
            For each market, provide:
            1. The research-based probability estimate (0-100%)
            2. Clear reasoning for that probability
            3. Confidence level in the estimate (0-1)
            
            Focus on extracting concrete probability estimates from the research, not market prices.
            If the research doesn't provide a clear probability for a market, make your best estimate based on the available information.
            """
            
            # Use GPT-4o to extract probabilities with structured output
            response = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional prediction market analyst. Extract probability estimates from research with structured output."},
                    {"role": "user", "content": prompt}
                ],
                response_format=ProbabilityExtraction,
            )
            
            # Return the extracted probabilities
            return event_ticker, response.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"Error extracting probabilities for {event_ticker}: {e}")
            return event_ticker, None

    async def extract_probabilities(self, research_results: Dict[str, str], 
                                  event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, ProbabilityExtraction]:
        """Extract structured probabilities from research results using GPT-4o in parallel."""
        self.console.print(f"\n[bold]Step 3.5: Extracting probabilities from research...[/bold]")
        
        probability_extractions = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Extracting probabilities...", total=len(research_results))
            
            # Create tasks for all events to run in parallel
            tasks = []
            for event_ticker, research_text in research_results.items():
                task_coroutine = self._extract_probabilities_for_event(event_ticker, research_text, event_markets)
                tasks.append(task_coroutine)
            
            # Run all probability extractions in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in probability extraction: {result}")
                    progress.update(task, advance=1)
                    continue
                    
                event_ticker, extraction = result
                if extraction is not None:
                    probability_extractions[event_ticker] = extraction
                    self.console.print(f"[green]✓ Extracted probabilities for {event_ticker}[/green]")
                    
                    # Display extracted probabilities
                    self.console.print(f"[blue]Extracted probabilities for {event_ticker}:[/blue]")
                    for market_prob in extraction.markets:
                        self.console.print(f"  {market_prob.ticker}: {market_prob.research_probability:.1f}%")
                else:
                    self.console.print(f"[red]✗ Failed to extract probabilities for {event_ticker}[/red]")
                
                progress.update(task, advance=1)
        
        self.console.print(f"[green]✓ Extracted probabilities for {len(probability_extractions)} events[/green]")
        return probability_extractions
    
    
    
    async def get_market_odds(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Fetch current market odds for all markets."""
        self.console.print(f"\n[bold]Step 4: Fetching current market odds...[/bold]")
        
        market_odds = {}
        all_tickers = []
        
        # Collect all market tickers
        for event_ticker, data in event_markets.items():
            markets = data['markets']
            for market in markets:
                ticker = market.get('ticker', '')
                if ticker:
                    all_tickers.append(ticker)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching market odds...", total=len(all_tickers))
            
            # Fetch odds in batches to avoid overwhelming the API
            batch_size = 20
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i:i + batch_size]
                
                # Fetch batch in parallel
                tasks = []
                for ticker in batch:
                    tasks.append(self.kalshi_client.get_market_with_odds(ticker))
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for ticker, result in zip(batch, results):
                        if not isinstance(result, Exception) and result:
                            market_odds[ticker] = result
                            progress.update(task, advance=1)
                        else:
                            self.console.print(f"[red]✗ Failed to get odds for {ticker}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch odds fetch error: {e}[/red]")
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(0.2)
        
        self.console.print(f"[green]✓ Fetched odds for {len(market_odds)} markets[/green]")
        return market_odds
    
    async def _get_betting_decisions_for_event(self, event_ticker: str, data: Dict[str, Any], 
                                             probability_extraction: ProbabilityExtraction, 
                                             market_odds: Dict[str, Dict[str, Any]]) -> tuple[str, Optional[MarketAnalysis]]:
        """Get betting decisions for a single event with error handling."""
        try:
            # Get event-specific decisions
            event_analysis = await self._get_event_betting_decisions(
                event_ticker, data, probability_extraction, market_odds
            )
            return event_ticker, event_analysis
        except Exception as e:
            logger.error(f"Error generating decisions for {event_ticker}: {e}")
            return event_ticker, None

    async def get_betting_decisions(self, event_markets: Dict[str, Dict[str, Any]], 
                                   probability_extractions: Dict[str, ProbabilityExtraction], 
                                   market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Use OpenAI to make structured betting decisions per event in parallel."""
        self.console.print(f"\n[bold]Step 5: Generating betting decisions...[/bold]")
        
        # Process events in parallel for better performance
        all_decisions = []
        total_recommended_bet = 0.0
        high_confidence_bets = 0
        event_summaries = []
        
        # Filter to events that have both research results and markets
        processable_events = [
            (event_ticker, data) for event_ticker, data in event_markets.items()
            if event_ticker in probability_extractions and data['markets']
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Generating betting decisions...", total=len(processable_events))
            
            # Create tasks for all events to run in parallel
            tasks = []
            for event_ticker, data in processable_events:
                task_coroutine = self._get_betting_decisions_for_event(
                    event_ticker, data, probability_extractions[event_ticker], market_odds
                )
                tasks.append(task_coroutine)
            
            # Run all betting decision generations in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in betting decisions generation: {result}")
                    progress.update(task, advance=1)
                    continue
                
                event_ticker, event_analysis = result
                if event_analysis is not None:
                    # Display decisions for this event
                    self._display_event_decisions(event_ticker, event_analysis)
                    
                    # Aggregate results
                    all_decisions.extend(event_analysis.decisions)
                    total_recommended_bet += event_analysis.total_recommended_bet
                    high_confidence_bets += event_analysis.high_confidence_bets
                    event_summaries.append(f"{event_ticker}: {event_analysis.summary}")
                    
                    self.console.print(f"[green]✓ Generated {len(event_analysis.decisions)} decisions for {event_ticker}[/green]")
                else:
                    self.console.print(f"[red]✗ Failed to generate decisions for {event_ticker}[/red]")
                
                progress.update(task, advance=1)
        
        # Create combined analysis
        analysis = MarketAnalysis(
            decisions=all_decisions,
            total_recommended_bet=total_recommended_bet,
            high_confidence_bets=high_confidence_bets,
            summary=f"Analyzed {len(processable_events)} events. " + " | ".join(event_summaries[:3]) + 
                   (f" and {len(event_summaries) - 3} more..." if len(event_summaries) > 3 else "")
        )
        
        # Show overall decision summary
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        self.console.print(f"\n[green]✓ Generated {len(analysis.decisions)} total decisions ({len(actionable_decisions)} actionable)[/green]")
        
        # Display consolidated summary table
        if actionable_decisions:
            table = Table(title="📊 All Betting Decisions Summary", show_lines=True)
            table.add_column("Event", style="bright_blue", width=25)
            table.add_column("Market", style="cyan", width=35)
            table.add_column("Action", style="yellow", justify="center", width=10)
            table.add_column("Confidence", style="magenta", justify="right", width=10)
            table.add_column("Amount", style="green", justify="right", width=10)
            table.add_column("Reasoning", style="blue", width=70)
            
            for decision in actionable_decisions:
                # Use human-readable names if available
                event_name = decision.event_name if decision.event_name else "Unknown Event"
                market_name = decision.market_name if decision.market_name else decision.ticker
                
                table.add_row(
                    event_name,
                    market_name,
                    decision.action.upper().replace('_', ' '),
                    f"{decision.confidence:.2f}",
                    f"${decision.amount:.2f}",
                    decision.reasoning
                )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No actionable betting decisions generated[/yellow]")
        
        # Show summary
        self.console.print(f"\n[blue]Total recommended bet: ${analysis.total_recommended_bet:.2f}[/blue]")
        self.console.print(f"[blue]High confidence bets: {analysis.high_confidence_bets}[/blue]")
        self.console.print(f"[blue]Strategy: {analysis.summary}[/blue]")
        
        return analysis
    
    def _display_event_decisions(self, event_ticker: str, event_analysis: MarketAnalysis):
        """Display the betting decisions for a single event."""
        # Filter to actionable decisions (not skip)
        actionable_decisions = [
            decision for decision in event_analysis.decisions 
            if decision.action != "skip"
        ]
        
        # Check if any decisions were adjusted due to mutually exclusive constraint
        mutually_exclusive_adjustments = [
            decision for decision in event_analysis.decisions 
            if decision.action != "skip" and "Mutually exclusive hedge" in decision.reasoning
        ]
        
        # Check if strategic filtering was applied
        strategic_filtering_skips = [
            decision for decision in event_analysis.decisions 
            if decision.action == "skip" and "Strategic filter" in decision.reasoning
        ]
        
        if not actionable_decisions:
            self.console.print(f"[yellow]No actionable decisions for {event_ticker}[/yellow]")
            return
        
        # Create event-specific table
        event_name = actionable_decisions[0].event_name if actionable_decisions else "Unknown Event"
        table = Table(title=f"Betting Decisions for {event_name}", show_lines=True)
        table.add_column("Market", style="cyan", width=45)
        table.add_column("Action", style="yellow", justify="center", width=10)
        table.add_column("Confidence", style="magenta", justify="right", width=10)
        table.add_column("Amount", style="green", justify="right", width=10)
        table.add_column("Reasoning", style="blue", width=75)
        
        for decision in actionable_decisions:
            # Use human-readable market name if available, otherwise generate from ticker
            market_display = decision.market_name if decision.market_name else self._generate_readable_market_name(decision.ticker)
            
            table.add_row(
                market_display,
                decision.action.upper().replace('_', ' '),
                f"{decision.confidence:.2f}",
                f"${decision.amount:.2f}",
                decision.reasoning
            )
        
        self.console.print(table)
        
        # Show mutually exclusive strategy info if applicable
        if mutually_exclusive_adjustments:
            self.console.print(f"[blue]ℹ Strategic hedge betting: {len(mutually_exclusive_adjustments)} positions sized for mutually exclusive event[/blue]")
        
        # Show strategic filtering info if applicable
        if strategic_filtering_skips:
            self.console.print(f"[yellow]⚡ Strategic filtering: {len(strategic_filtering_skips)} lower-value opportunities skipped, focused on best positions[/yellow]")
        
        # Show event summary
        if event_analysis.total_recommended_bet > 0:
            self.console.print(f"[blue]Event total: ${event_analysis.total_recommended_bet:.2f} | High confidence: {event_analysis.high_confidence_bets}[/blue]")
    
    async def _get_event_betting_decisions(self, event_ticker: str, event_data: Dict[str, Any], 
                                         probability_extraction: ProbabilityExtraction, market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Get betting decisions for a single event."""
        event_info = event_data['event']
        markets = event_data['markets']
        
        # Include market odds in the data
        markets_with_odds = []
        for market in markets:
            ticker = market.get('ticker', '')
            market_data = {
                'ticker': ticker,
                'title': market.get('title', ''),
                'volume': market.get('volume', 0)
            }
            
            # Add current market odds if available
            if ticker in market_odds:
                odds = market_odds[ticker]
                
                yes_bid = odds.get('yes_bid', 0)
                no_bid = odds.get('no_bid', 0)
                yes_ask = odds.get('yes_ask', 0)
                no_ask = odds.get('no_ask', 0)
                
                # Calculate mid-prices
                yes_mid_price = (yes_bid + yes_ask) / 2
                no_mid_price = (no_bid + no_ask) / 2
                
                market_data.update({
                    'yes_bid': yes_bid,
                    'no_bid': no_bid,
                    'yes_ask': yes_ask,
                    'no_ask': no_ask,
                    'status': odds.get('status', ''),
                    # Calculate implied probabilities from mid-prices
                    'yes_mid_price': yes_mid_price,
                    'no_mid_price': no_mid_price,
                })
            
            markets_with_odds.append(market_data)
        
        # Create single event data with structured probabilities
        is_mutually_exclusive = event_info.get('mutually_exclusive', False)
        single_event_data = {
            'event_ticker': event_ticker,
            'event_title': event_info.get('title', ''),
            'event_category': event_info.get('category', ''),
            'event_volume': event_info.get('volume', 0),
            'time_remaining_hours': event_info.get('time_remaining_hours'),
            'strike_date': event_info.get('strike_date', ''),
            'strike_period': event_info.get('strike_period', ''),
            'mutually_exclusive': is_mutually_exclusive,
            'markets': markets_with_odds,
            'research_summary': probability_extraction.overall_summary,
            'market_probabilities': [
                {
                    'ticker': mp.ticker,
                    'title': mp.title,
                    'research_probability': mp.research_probability,
                    'reasoning': mp.reasoning,
                    'confidence': mp.confidence
                }
                for mp in probability_extraction.markets
            ]
        }
        
        # Create prompt for OpenAI
        mutually_exclusive_guidance = ""
        if is_mutually_exclusive:
            mutually_exclusive_guidance = """
            
            🚨 MUTUALLY EXCLUSIVE EVENT - STRATEGIC HEDGE BETTING:
            This event is MUTUALLY EXCLUSIVE - only ONE outcome can be true.
            
            STRATEGIC BETTING APPROACH:
            - You CAN place multiple bets (YES and NO) with different position sizes
            - Focus on creating a profitable hedge portfolio across outcomes
            - Primary strategy: Find the best value opportunity for your largest YES bet
            - Secondary strategy: Place smaller YES bets on other good value opportunities
            - Tertiary strategy: Place NO bets on clearly overpriced outcomes
            - Key principle: Position sizing should reflect probability and value, not equal amounts
            
            POSITION SIZING GUIDELINES:
            - Largest bet: Best value opportunity (highest edge)
            - Medium bets: Good value opportunities (moderate edge)
            - Small bets: Decent value or hedge positions
            - Consider potential profit/loss scenarios across different outcomes
            
            Example: If researching shows A=50%, B=30%, C=20% but market prices A=30%, B=40%, C=30%:
            - BUY YES A (large) - undervalued by 20 points
            - BUY NO B (medium) - overvalued by 10 points  
            - BUY YES C (small) - undervalued by 10 points
            """
        else:
            mutually_exclusive_guidance = """
            
            NON-MUTUALLY EXCLUSIVE EVENT:
            Multiple outcomes in this event can be true simultaneously.
            You can place multiple YES bets if there are multiple good opportunities.
            Position sizing should reflect individual value opportunities.
            """
        
        prompt = f"""
        You are a professional prediction market trader. Based on the research provided for this event AND the current market odds, 
        make betting decisions for the individual markets within this event.
        
        Max bet per market: ${self.config.max_bet_amount}
        {mutually_exclusive_guidance}
        
        IMPORTANT TRADING CONSTRAINTS:
        - You can ONLY buy YES or buy NO positions (no shorting or sophisticated selling)
        - When you buy YES, you profit if the outcome is YES
        - When you buy NO, you profit if the outcome is NO
        - Prices are in cents (0-100, where 50 = 50% probability)
        - Look for value opportunities where research probability differs significantly from market odds
        
        STRUCTURED PROBABILITY DATA AVAILABLE:
        - Each market has a research_probability (0-100%) extracted from deep research
        - Each market has detailed reasoning for the probability estimate
        - Use these precise probabilities to calculate edges and alpha ratios
        - The research_summary provides overall context
        
        Event Data, Research Probabilities, and Current Market Odds:
        {json.dumps(single_event_data, indent=2)}
        
        For each market, decide:
        1. Action: "buy_yes", "buy_no", or "skip"
        2. Confidence: 0-1 (only bet if confidence > 0.75 for primary positions, > 0.85 for hedge positions)
        3. Amount: How much to bet (max ${self.config.max_bet_amount})
        4. Reasoning: Brief explanation comparing research prediction to current market odds
        
        STRATEGIC BETTING APPROACH - BE HIGHLY SELECTIVE:
        - SKIP most markets - only bet on exceptional opportunities
        - Primary position: Find the ONE best value opportunity (highest edge × confidence)
        - Secondary positions: Maximum 1-2 hedge bets only if they're truly exceptional
        - Minimum edge requirement: Research probability must differ by at least 5 percentage points from market odds
        - Focus on quality over quantity - better to make 1 great bet than 5 mediocre ones
        
        MINIMUM ALPHA THRESHOLD REQUIREMENT:
        - Only place bets when there's at least a {self.config.minimum_alpha_threshold}x difference between research and market price
        - For YES bets: Research probability must be >= {self.config.minimum_alpha_threshold}x the market yes_mid_price
        - For NO bets: Research probability must be <= market no_mid_price / {self.config.minimum_alpha_threshold}
        - Example: If market yes_mid_price is 25% and minimum_alpha_threshold is 2.0x, research probability must be >= 50%
        - Example: If market no_mid_price is 75% and minimum_alpha_threshold is 2.0x, research probability must be <= 37.5%
        - SKIP all bets that don't meet this minimum alpha threshold
        
        POSITION SIZING STRATEGY:
        - Primary bet: Largest position (${self.config.max_bet_amount}) on the single best opportunity
        - Secondary bets: Much smaller positions (≤60% of primary) only for exceptional hedge opportunities
        - Most markets: SKIP - don't bet unless there's a clear, substantial edge
        
        EDGE CALCULATION:
        - Compare research predicted probability to yes_mid_price and no_mid_price
        - Look for market mispricing of 5+ percentage points
        - Higher confidence required for smaller edges
        - MANDATORY: Check minimum alpha threshold before considering any bet
        
        Return your analysis in the specified JSON format.
        """
        
        try:
            response = await self.openai_client.beta.chat.completions.parse(
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a professional prediction market trader."},
                    {"role": "user", "content": prompt}
                ],
                response_format=MarketAnalysis,
            )
            
            # Access the parsed response (correct OpenAI API format)
            analysis = response.choices[0].message.parsed
            
            # Enrich decisions with human-readable names
            analysis = self._add_human_readable_names(analysis, event_info, markets)
            
            # Apply alpha threshold validation to ensure minimum edge requirements
            analysis = self._apply_alpha_threshold_validation(analysis, event_ticker, markets, probability_extraction)
            
            # Apply strategic filtering to ensure selective betting
            analysis = self._apply_strategic_filtering(analysis, event_ticker)
            
            # Post-process for mutually exclusive events: ensure only one YES bet
            if is_mutually_exclusive:
                analysis = self._enforce_mutually_exclusive_constraint(analysis, event_ticker)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating decisions for {event_ticker}: {e}")
            # Return empty analysis for this event
            return MarketAnalysis(
                decisions=[],
                total_recommended_bet=0.0,
                high_confidence_bets=0,
                summary=f"Error generating decisions for {event_ticker}"
            )
    
    def _add_human_readable_names(self, analysis: MarketAnalysis, event_info: Dict[str, Any], markets: List[Dict[str, Any]]) -> MarketAnalysis:
        """Enrich BettingDecision objects with human-readable event and market names."""
        enriched_decisions = []
        event_name = event_info.get('title', 'Unknown Event')
        
        for decision in analysis.decisions:
            # Find the market title from the original markets list
            market_title = None
            for market in markets:
                if market.get('ticker') == decision.ticker:
                    market_title = market.get('title', '')
                    break
            
            # If no market title found, generate a readable name from ticker
            if not market_title:
                market_title = self._generate_readable_market_name(decision.ticker)
            
            enriched_decisions.append(
                BettingDecision(
                    ticker=decision.ticker,
                    action=decision.action,
                    confidence=decision.confidence,
                    amount=decision.amount,
                    reasoning=decision.reasoning,
                    event_name=event_name,
                    market_name=market_title
                )
            )
        return MarketAnalysis(
            decisions=enriched_decisions,
            total_recommended_bet=analysis.total_recommended_bet,
            high_confidence_bets=analysis.high_confidence_bets,
            summary=analysis.summary
        )
    
    def _enforce_mutually_exclusive_constraint(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Validate and optimize betting strategy for mutually exclusive events."""
        yes_bets = [d for d in analysis.decisions if d.action == "buy_yes"]
        no_bets = [d for d in analysis.decisions if d.action == "buy_no"]
        skip_bets = [d for d in analysis.decisions if d.action == "skip"]
        
        # If no multiple YES bets, return as-is (already fine)
        if len(yes_bets) <= 1:
            return analysis
        
        # For mutually exclusive events with multiple YES bets, validate position sizing
        # Sort YES bets by value (confidence * amount as proxy for expected value)
        yes_bets_sorted = sorted(yes_bets, key=lambda d: d.confidence * d.amount, reverse=True)
        
        # Check if position sizing makes strategic sense
        # Primary bet should be significantly larger than secondary bets
        primary_bet = yes_bets_sorted[0]
        secondary_bets = yes_bets_sorted[1:]
        
        # Calculate position sizing ratios
        primary_value = primary_bet.confidence * primary_bet.amount
        modified_decisions = []
        valid_secondary_bets = []
        
        for decision in analysis.decisions:
            if decision.action == "buy_yes" and decision.ticker != primary_bet.ticker:
                secondary_value = decision.confidence * decision.amount
                
                # Allow secondary YES bets if they're appropriately sized relative to primary
                # Secondary bets should be smaller (< 80% of primary bet value)
                if secondary_value < (primary_value * 0.8):
                    valid_secondary_bets.append(decision)
                    modified_decisions.append(decision)
                else:
                    # Convert oversized secondary bets to smaller positions
                    adjusted_amount = min(decision.amount, primary_bet.amount * 0.6)
                    adjusted_decision = BettingDecision(
                        ticker=decision.ticker,
                        action=decision.action,
                        confidence=decision.confidence,
                        amount=adjusted_amount,
                        reasoning=f"Mutually exclusive hedge: reduced position size for strategic betting. {decision.reasoning}",
                        event_name=decision.event_name,
                        market_name=decision.market_name
                    )
                    valid_secondary_bets.append(adjusted_decision)
                    modified_decisions.append(adjusted_decision)
            else:
                # Keep primary YES bet, NO bets, and skip decisions as-is
                modified_decisions.append(decision)
        
        # Recalculate totals
        new_total_bet = sum(d.amount for d in modified_decisions if d.action != "skip")
        new_high_confidence = sum(1 for d in modified_decisions if d.action != "skip" and d.confidence >= 0.8)
        
        # Log the strategy
        logger.info(f"Mutually exclusive strategic betting for {event_ticker}: "
                   f"Primary YES bet: {primary_bet.ticker} (${primary_bet.amount}), "
                   f"Secondary YES bets: {len(valid_secondary_bets)}, "
                   f"NO bets: {len(no_bets)}")
        
        return MarketAnalysis(
            decisions=modified_decisions,
            total_recommended_bet=new_total_bet,
            high_confidence_bets=new_high_confidence,
            summary=f"Strategic hedge betting: 1 primary + {len(valid_secondary_bets)} secondary YES bets, {len(no_bets)} NO bets"
        )
    
    async def place_bets(self, analysis: MarketAnalysis, market_odds: Dict[str, Dict[str, Any]], 
                        probability_extractions: Dict[str, ProbabilityExtraction]):
        """Place bets based on the analysis with enhanced table showing probabilities."""
        self.console.print(f"\n[bold]Step 6: Placing bets...[/bold]")
        
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
        
        # Create a lookup for probability data across all events
        prob_lookup = {}
        for event_ticker, prob_extraction in probability_extractions.items():
            for market_prob in prob_extraction.markets:
                prob_lookup[market_prob.ticker] = market_prob
        
        # Display bets to be placed in table format with probabilities
        table = Table(title="🎯 Bets to be Placed", show_lines=True)
        table.add_column("Event", style="bright_blue", width=20)
        table.add_column("Market", style="cyan", width=25)
        table.add_column("Action", style="yellow", justify="center", width=8)
        table.add_column("Amount", style="green", justify="right", width=8)
        table.add_column("Research %", style="magenta", justify="right", width=10)
        table.add_column("Market %", style="red", justify="right", width=10)
        table.add_column("Confidence", style="bright_magenta", justify="right", width=10)
        table.add_column("Reasoning", style="blue", width=45)
        
        for decision in actionable_decisions:
            # Use human-readable names if available, otherwise use ticker
            event_name = decision.event_name if decision.event_name else "Unknown Event"
            market_name = decision.market_name if decision.market_name else decision.ticker
            
            # Get structured research probability (raw, not confidence-scaled)
            prob_data = prob_lookup.get(decision.ticker)
            if prob_data:
                research_prob_str = f"{prob_data.research_probability:.1f}%"
            else:
                research_prob_str = "N/A"
            
            # Extract market probability from market odds
            market_prob = self._extract_market_probability(decision.ticker, decision.action, market_odds)
            market_prob_str = f"{market_prob:.1f}%" if market_prob is not None else "N/A"
            
            table.add_row(
                event_name,
                market_name,
                decision.action.upper().replace('_', ' '),
                f"${decision.amount:.2f}",
                research_prob_str,
                market_prob_str,
                f"{decision.confidence:.2f}",
                decision.reasoning
            )
        
        self.console.print(table)
        
        # Show betting summary
        total_amount = sum(decision.amount for decision in actionable_decisions)
        self.console.print(f"\n[blue]Total bets to place: {len(actionable_decisions)} | Total amount: ${total_amount:.2f}[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Placing bets...", total=len(actionable_decisions))
            
            for decision in actionable_decisions:
                try:
                    # Position checking already done earlier in filter_markets_by_positions()
                    
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
    
    def _extract_market_name_from_ticker(self, ticker: str) -> str:
        """Extract a readable market name from ticker as fallback."""
        # Remove common prefixes to make it more readable
        if '-' in ticker:
            parts = ticker.split('-')
            if len(parts) >= 3:
                return f"{parts[0]}-{parts[1]}-{parts[-1]}"
        return ticker
    
    def _generate_readable_market_name(self, ticker: str) -> str:
        """Generate a readable market name from ticker when original title isn't available."""
        # Extract the market-specific part after the event ticker
        parts = ticker.split('-')
        if len(parts) >= 3:
            # For tickers like KXMLBHRDERBY-25-CRAL, extract CRAL
            market_code = parts[-1]
            
            # Try to make common codes more readable
            readable_names = {
                'CRAL': 'Carlos Correa',
                'JWOO': 'James Wood', 
                'OCRU': 'Oneil Cruz',
                'MOLS': 'Matt Olson',
                'BBUX': 'Byron Buxton',
                'JCHI': 'Jazz Chisholm Jr.',
                'JAC': 'Jacquemot',
                'SHA': 'Sharma',
                'ROM': 'Romero Gormaz',
                'LYS': 'Lys',
                'EADA': 'Eric Adams',
                'D': 'Democrat',
                'R': 'Republican',
                'AC': 'Andrew Cuomo'
            }
            
            if market_code in readable_names:
                return f"Will {readable_names[market_code]} win?"
            else:
                return f"Market {market_code}"
        
        # Fallback to ticker if we can't parse it
        return ticker
    
    def _apply_strategic_filtering(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Apply strategic filtering to ensure selective, high-quality betting."""
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        
        if not actionable_decisions:
            return analysis
        
        # Calculate value scores for each decision (edge × confidence approximation)
        scored_decisions = []
        for decision in actionable_decisions:
            # Estimate edge based on confidence and amount (higher confidence and amount suggests better edge)
            value_score = decision.confidence * decision.amount
            scored_decisions.append((value_score, decision))
        
        # Sort by value score (highest first)
        scored_decisions.sort(key=lambda x: x[0], reverse=True)
        
        # Apply strategic filtering rules
        filtered_decisions = []
        skip_decisions = []
        
        # Rule 1: Maximum 2 bets per event
        max_bets = 2
        
        # Rule 2: Primary bet should be high confidence (>0.75) and substantial amount
        primary_threshold = 0.75
        hedge_threshold = 0.85
        
        for i, (score, decision) in enumerate(scored_decisions):
            if i == 0:
                # Primary position: must meet higher standards
                if decision.confidence >= primary_threshold and decision.amount >= self.config.max_bet_amount * 0.4:
                    filtered_decisions.append(decision)
                else:
                    skip_decisions.append(BettingDecision(
                        ticker=decision.ticker,
                        action="skip",
                        confidence=decision.confidence,
                        amount=0.0,
                        reasoning=f"Strategic filter: Primary position needs confidence ≥{primary_threshold} and substantial amount. Original: {decision.reasoning}",
                        event_name=decision.event_name,
                        market_name=decision.market_name
                    ))
            elif i == 1 and len(filtered_decisions) > 0:
                # Secondary position: must be exceptional and smaller than primary
                primary_amount = filtered_decisions[0].amount
                if (decision.confidence >= hedge_threshold and 
                    decision.amount <= primary_amount * 0.6):
                    filtered_decisions.append(decision)
                else:
                    skip_decisions.append(BettingDecision(
                        ticker=decision.ticker,
                        action="skip",
                        confidence=decision.confidence,
                        amount=0.0,
                        reasoning=f"Strategic filter: Secondary position needs confidence ≥{hedge_threshold} and amount ≤60% of primary. Original: {decision.reasoning}",
                        event_name=decision.event_name,
                        market_name=decision.market_name
                    ))
            else:
                # Skip additional positions
                skip_decisions.append(BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Strategic filter: Limited to {max_bets} bets per event, focusing on highest value opportunities. Original: {decision.reasoning}",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                ))
        
        # Add back the original skip decisions
        original_skips = [d for d in analysis.decisions if d.action == "skip"]
        all_decisions = filtered_decisions + skip_decisions + original_skips
        
        # Recalculate totals
        new_total_bet = sum(d.amount for d in all_decisions if d.action != "skip")
        new_high_confidence = sum(1 for d in all_decisions if d.action != "skip" and d.confidence >= 0.8)
        
        # Log the filtering results
        logger.info(f"Strategic filtering for {event_ticker}: "
                   f"reduced from {len(actionable_decisions)} to {len(filtered_decisions)} bets "
                   f"(kept top {len(filtered_decisions)} highest-value opportunities)")
        
        return MarketAnalysis(
            decisions=all_decisions,
            total_recommended_bet=new_total_bet,
            high_confidence_bets=new_high_confidence,
            summary=f"Strategic betting: {len(filtered_decisions)} high-value positions selected from {len(actionable_decisions)} opportunities"
        )
    
    def _apply_alpha_threshold_validation(self, analysis: MarketAnalysis, event_ticker: str, 
                                        markets: List[Dict[str, Any]], 
                                        probability_extraction: ProbabilityExtraction) -> MarketAnalysis:
        """Apply minimum alpha threshold validation to ensure only high-alpha opportunities are bet on."""
        validated_decisions = []
        skip_decisions = []
        
        # Create a lookup for market data by ticker
        market_lookup = {market['ticker']: market for market in markets}
        
        # Create a lookup for probability data by ticker
        prob_lookup = {mp.ticker: mp for mp in probability_extraction.markets}
        
        for decision in analysis.decisions:
            if decision.action == "skip":
                validated_decisions.append(decision)
                continue
            
            # Get market data for this ticker
            market_data = market_lookup.get(decision.ticker)
            if not market_data:
                logger.warning(f"Could not find market data for {decision.ticker}")
                validated_decisions.append(decision)
                continue
            
            # Get probability data for this ticker
            prob_data = prob_lookup.get(decision.ticker)
            if not prob_data:
                logger.warning(f"Could not find probability data for {decision.ticker}")
                validated_decisions.append(decision)
                continue
            
            # Check if alpha threshold is met
            if self._meets_alpha_threshold(decision, market_data, prob_data):
                validated_decisions.append(decision)
            else:
                # Get relevant market price for error message
                market_price = market_data.get('yes_mid_price', 0) if decision.action == "buy_yes" else market_data.get('no_mid_price', 0)
                
                # Convert to skip decision with alpha threshold reasoning
                skip_decisions.append(BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Alpha threshold not met (requires {self.config.minimum_alpha_threshold}x difference). Research: {prob_data.research_probability:.1f}%, Market: {market_price:.1f}%",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                ))
        
        # Show alpha threshold filtering results
        if skip_decisions:
            logger.info(f"Alpha threshold filter for {event_ticker}: {len(skip_decisions)} bets skipped for insufficient alpha")
        
        # Update analysis with validated decisions
        validated_analysis = MarketAnalysis(
            decisions=validated_decisions + skip_decisions,
            total_recommended_bet=sum(d.amount for d in validated_decisions if d.action != "skip"),
            high_confidence_bets=len([d for d in validated_decisions if d.action != "skip" and d.confidence > 0.7]),
            summary=analysis.summary
        )
        
        return validated_analysis
    
    def _meets_alpha_threshold(self, decision: BettingDecision, market_data: Dict[str, Any], prob_data: MarketProbability) -> bool:
        """Check if a betting decision meets the minimum alpha threshold.
        
        Note: This uses raw research probabilities. If calibration issues are identified,
        the proper approach is to:
        1. Generate a reliability diagram on validation data
        2. Apply Platt scaling or isotonic regression if needed
        3. Use calibrated probabilities, not ad-hoc scaling
        
        Simply multiplying probabilities by confidence scores introduces bias and
        doesn't address the underlying calibration problem.
        """
        # Get market odds
        yes_mid_price = market_data.get('yes_mid_price', 0)
        no_mid_price = market_data.get('no_mid_price', 0)
        
        # Use raw research probability (not confidence-scaled)
        research_probability = prob_data.research_probability
        confidence = prob_data.confidence
        
        if research_probability is None:
            logger.warning(f"Could not extract research probability from probability_extraction: {prob_data.reasoning}")
            return True  # Allow bet if we can't determine probability
        
        # Convert to decimal (0-1 range) - using raw research probability
        research_prob = research_probability / 100.0
        market_yes_price = yes_mid_price / 100.0
        market_no_price = no_mid_price / 100.0
        
        if decision.action == "buy_yes":
            # For YES bets: research probability should be >= market_price * alpha_threshold
            if market_yes_price > 0:
                alpha_ratio = research_prob / market_yes_price
                logger.debug(f"YES bet alpha check for {decision.ticker}: "
                            f"research={research_probability:.1f}%, confidence={confidence:.2f}, "
                            f"market={market_yes_price*100:.1f}%, "
                            f"alpha={alpha_ratio:.2f}x, required={self.config.minimum_alpha_threshold}x")
                return alpha_ratio >= self.config.minimum_alpha_threshold
            else:
                return True  # Allow if market price is 0 (extreme edge case)
        
        elif decision.action == "buy_no":
            # For NO bets: research probability should be <= market_no_price / alpha_threshold
            if research_prob > 0:
                alpha_ratio = market_no_price / research_prob
                logger.debug(f"NO bet alpha check for {decision.ticker}: "
                            f"research={research_probability:.1f}%, confidence={confidence:.2f}, "
                            f"market_no={market_no_price*100:.1f}%, "
                            f"alpha={alpha_ratio:.2f}x, required={self.config.minimum_alpha_threshold}x")
                return alpha_ratio >= self.config.minimum_alpha_threshold
            else:
                return True  # Allow if research probability is 0 (extreme edge case)
        
        return True  # Default to allow if action is unclear
    
    def _extract_market_probability(self, ticker: str, action: str, market_odds: Dict[str, Dict[str, Any]]) -> Optional[float]:
        """Extract market probability from market odds based on the action."""
        odds = market_odds.get(ticker, {})
        
        if not odds:
            logger.debug(f"No odds found for ticker {ticker}")
            return None
        
        # Calculate mid-prices from bid/ask
        yes_bid = odds.get('yes_bid', 0)
        no_bid = odds.get('no_bid', 0)
        yes_ask = odds.get('yes_ask', 0)
        no_ask = odds.get('no_ask', 0)
        
        yes_mid_price = (yes_bid + yes_ask) / 2
        no_mid_price = (no_bid + no_ask) / 2
        
        if action == "buy_yes":
            # For YES bets, use the yes_mid_price as the market's implied probability
            return yes_mid_price if yes_mid_price > 0 else None
        elif action == "buy_no":
            # For NO bets, use the no_mid_price as the market's implied probability
            return no_mid_price if no_mid_price > 0 else None
        
        return None
    
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
            
            event_markets = await self.filter_markets_by_positions(event_markets)
            if not event_markets:
                self.console.print("[red]No markets remaining after position filtering. Exiting.[/red]")
                return
            
            # Limit to max_events_to_analyze after position filtering
            if len(event_markets) > self.config.max_events_to_analyze:
                # Sort filtered events by volume_24h and take top N
                filtered_events_list = []
                for event_ticker, data in event_markets.items():
                    event = data['event']
                    volume_24h = event.get('volume_24h', 0)
                    filtered_events_list.append((event_ticker, data, volume_24h))
                
                # Sort by volume_24h (descending) and take top max_events_to_analyze
                filtered_events_list.sort(key=lambda x: x[2], reverse=True)
                top_events = filtered_events_list[:self.config.max_events_to_analyze]
                
                # Rebuild event_markets dict with only top events
                event_markets = {event_ticker: data for event_ticker, data, _ in top_events}
                
                self.console.print(f"[blue]• Limited to top {len(event_markets)} events by volume after position filtering[/blue]")
            
            research_results = await self.research_events(event_markets)
            if not research_results:
                self.console.print("[red]No research results. Exiting.[/red]")
                return
            
            probability_extractions = await self.extract_probabilities(research_results, event_markets)
            if not probability_extractions:
                self.console.print("[red]No probability extractions. Exiting.[/red]")
                return
            
            market_odds = await self.get_market_odds(event_markets)
            if not market_odds:
                self.console.print("[red]No market odds found. Exiting.[/red]")
                return
            
            analysis = await self.get_betting_decisions(event_markets, probability_extractions, market_odds)
            await self.place_bets(analysis, market_odds, probability_extractions)
            
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


async def main(live_trading: bool = False):
    """Main entry point."""
    bot = SimpleTradingBot(live_trading=live_trading)
    await bot.run()


def cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Kalshi trading bot with Octagon research and OpenAI decision making",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run trading-bot                    # Run bot in dry run mode (default)
  uv run trading-bot --live             # Run bot with live trading enabled
  uv run trading-bot --help            # Show this help message
  
Configuration:
  Create a .env file with your API keys:
    KALSHI_API_KEY=your_kalshi_api_key
    KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----"
    OCTAGON_API_KEY=your_octagon_api_key
    OPENAI_API_KEY=your_openai_api_key
    
  Optional settings:
    KALSHI_USE_DEMO=true               # Use demo environment (default: true)
    MAX_EVENTS_TO_ANALYZE=50           # Max events to analyze (default: 50)
    MAX_BET_AMOUNT=25.0                # Max bet per market (default: 25.0)
    RESEARCH_BATCH_SIZE=10             # Parallel research requests (default: 10)
    SKIP_EXISTING_POSITIONS=true       # Skip markets with existing positions (default: true)
    MINIMUM_ALPHA_THRESHOLD=2.0        # Minimum alpha threshold for betting (default: 2.0)
    
  Trading modes:
    Default: Dry run mode - shows what trades would be made without placing real bets
    --live: Live trading mode - actually places bets (use with caution!)
        """
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (default: dry run mode)'
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
        asyncio.run(main(live_trading=args.live))
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Please check your .env file configuration.[/yellow]")
        console.print("[yellow]Run with --help for more information.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    cli() 