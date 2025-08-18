"""
Simple Kalshi trading bot with Octagon research and OpenAI decision making.
"""
import asyncio
import argparse
import json
import csv
import json
import csv
import datetime
import math
from pathlib import Path
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger
import re

from kalshi_client import KalshiClient
from research_client import OctagonClient
from betting_models import BettingDecision, MarketAnalysis, ProbabilityExtraction
from config import load_config
import openai


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
        hedging_status = "Enabled" if self.config.enable_hedging else "Disabled"
        self.console.print(f"[blue]Risk hedging: {hedging_status} (ratio: {self.config.hedge_ratio}, min confidence: {self.config.min_confidence_for_hedging})[/blue]")
        
        # Show risk-adjusted trading settings
        self.console.print(f"[blue]R-score filtering: Enabled (z-threshold: {self.config.z_threshold})[/blue]")
        if self.config.enable_kelly_sizing:
            self.console.print(f"[blue]Kelly sizing: Enabled (fraction: {self.config.kelly_fraction}, bankroll: ${self.config.bankroll})[/blue]")
        self.console.print(f"[blue]Portfolio selection: {self.config.portfolio_selection_method} (max positions: {self.config.max_portfolio_positions})[/blue]\n")
    
    def calculate_risk_adjusted_metrics(self, research_prob: float, market_price: float, action: str) -> dict:
        """
        Calculate hedge-fund style risk-adjusted metrics.
        
        Args:
            research_prob: Research probability (0-1)
            market_price: Market price (0-1) 
            action: "buy_yes" or "buy_no"
            
        Returns:
            dict with expected_return, r_score, kelly_fraction
        """
        try:
            # Adjust probabilities based on action
            if action == "buy_yes":
                p = research_prob  # Our probability of YES
                y = market_price   # Market price of YES
            elif action == "buy_no":
                p = 1 - research_prob  # Our probability of NO (1 - research_prob_of_yes)
                y = market_price       # Market price of NO
            else:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
            
            # Prevent division by zero and invalid probabilities
            if y <= 0 or y >= 1 or p <= 0 or p >= 1:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
            
            # Expected return on capital: E[R] = (p-y)/y
            expected_return = (p - y) / y
            
            # Risk-Adjusted Edge (R-score): (p-y)/sqrt(p*(1-p)) 
            # This is the z-score - how many standard deviations away from fair value
            variance = p * (1 - p)
            if variance <= 0:
                return {"expected_return": expected_return, "r_score": 0.0, "kelly_fraction": 0.0}
            
            r_score = (p - y) / math.sqrt(variance)
            
            # Kelly fraction: f_kelly = (p-y)/(1-y)
            # This gives the optimal fraction of bankroll to bet
            if y >= 1:
                kelly_fraction = 0.0
            else:
                kelly_fraction = (p - y) / (1 - y)
                # Ensure Kelly fraction is reasonable (between 0 and 1)
                kelly_fraction = max(0.0, min(1.0, kelly_fraction))
            
            return {
                "expected_return": expected_return,
                "r_score": r_score, 
                "kelly_fraction": kelly_fraction
            }
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
    
    def calculate_kelly_position_size(self, kelly_fraction: float) -> float:
        """
        Calculate position size using fractional Kelly criterion.
        
        Args:
            kelly_fraction: Optimal Kelly fraction (0-1)
            
        Returns:
            Position size in dollars
        """
        if not self.config.enable_kelly_sizing or kelly_fraction <= 0:
            return self.config.max_bet_amount
        
        # Apply fractional Kelly (e.g., half-Kelly)
        adjusted_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Calculate position size as fraction of bankroll
        kelly_bet_size = self.config.bankroll * adjusted_kelly
        
        # Apply maximum bet fraction constraint
        max_allowed = self.config.bankroll * self.config.max_kelly_bet_fraction
        kelly_bet_size = min(kelly_bet_size, max_allowed)
        
        # Apply absolute maximum bet limit
        kelly_bet_size = min(kelly_bet_size, self.config.max_bet_amount)
        
        # Ensure minimum bet size
        kelly_bet_size = max(kelly_bet_size, 1.0)
        
        return kelly_bet_size
    
    def apply_portfolio_selection(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """
        Apply portfolio selection to hold only the N highest R-scores.
        Step 4: Portfolio view - hold the N highest R-scores subject to limits.
        """
        if self.config.portfolio_selection_method == "legacy":
            # Skip portfolio optimization, use existing logic
            return analysis
        
        # Filter out skip decisions for ranking
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        skip_decisions = [d for d in analysis.decisions if d.action == "skip"]
        
        if not actionable_decisions:
            return analysis
        
        if self.config.portfolio_selection_method == "top_r_scores":
            # Sort by R-score (highest first)
            actionable_decisions.sort(key=lambda d: d.r_score or -999, reverse=True)
            
            # Select top N positions
            max_positions = self.config.max_portfolio_positions
            selected_decisions = actionable_decisions[:max_positions]
            
            # Convert remaining to skip decisions
            rejected_decisions = []
            for decision in actionable_decisions[max_positions:]:
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Portfolio limit: R-score {decision.r_score:.2f} ranked #{len(selected_decisions)+1}",
                    event_name=decision.event_name,
                    market_name=decision.market_name,
                    expected_return=decision.expected_return,
                    r_score=decision.r_score,
                    kelly_fraction=decision.kelly_fraction,
                    market_price=decision.market_price,
                    research_probability=decision.research_probability
                )
                rejected_decisions.append(skip_decision)
            
            if rejected_decisions:
                logger.info(f"Portfolio selection: kept top {len(selected_decisions)} positions, "
                           f"rejected {len(rejected_decisions)} lower R-score positions")
            
            # Combine selected decisions with all skip decisions
            analysis.decisions = selected_decisions + skip_decisions + rejected_decisions
            
        elif self.config.portfolio_selection_method == "diversified":
            # Future enhancement: could implement diversification by event category, etc.
            # For now, fall back to top R-scores
            return self.apply_portfolio_selection(analysis, event_ticker)
        
        return analysis
    
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
        
        # Generate hedge decisions for risk management
        hedge_decisions = self._generate_hedge_decisions(all_decisions)
        if hedge_decisions:
            all_decisions.extend(hedge_decisions)
            # Update totals to include hedge amounts
            hedge_bet_total = sum(d.amount for d in hedge_decisions)
            total_recommended_bet += hedge_bet_total
            self.console.print(f"[blue]💡 Generated {len(hedge_decisions)} hedge bets (${hedge_bet_total:.2f}) for risk management[/blue]")
        
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
            table.add_column("Type", style="bright_blue", justify="center", width=8)
            table.add_column("Event", style="bright_blue", width=22)
            table.add_column("Market", style="cyan", width=32)
            table.add_column("Action", style="yellow", justify="center", width=10)
            table.add_column("Confidence", style="magenta", justify="right", width=10)
            table.add_column("Amount", style="green", justify="right", width=10)
            table.add_column("Reasoning", style="blue", width=65)
            
            for decision in actionable_decisions:
                # Use human-readable names if available
                event_name = decision.event_name if decision.event_name else "Unknown Event"
                market_name = decision.market_name if decision.market_name else decision.ticker
                
                # Determine bet type
                bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"
                
                table.add_row(
                    bet_type,
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
    
    def _generate_hedge_decisions(self, main_decisions: List[BettingDecision]) -> List[BettingDecision]:
        """Generate hedge decisions to minimize risk for main betting decisions."""
        if not self.config.enable_hedging:
            return []
        
        hedge_decisions = []
        
        for main_decision in main_decisions:
            # Skip if this is already a hedge or a skip
            if main_decision.is_hedge or main_decision.action == "skip":
                continue
                
            # Only hedge if confidence is below threshold (higher risk bets)
            if main_decision.confidence >= self.config.min_confidence_for_hedging:
                continue
                
            # Calculate hedge amount
            hedge_amount = min(
                main_decision.amount * self.config.hedge_ratio,
                self.config.max_hedge_amount
            )
            
            # Only create hedge if amount is meaningful (at least $1)
            if hedge_amount < 1.0:
                continue
            
            # Create opposite hedge position
            hedge_action = "buy_no" if main_decision.action == "buy_yes" else "buy_yes"
            
            hedge_decision = BettingDecision(
                ticker=main_decision.ticker,
                action=hedge_action,
                confidence=0.8,  # Hedge has high confidence (it's risk management)
                amount=hedge_amount,
                reasoning=f"Risk hedge: {self.config.hedge_ratio*100:.0f}% hedge for {main_decision.action} (confidence {main_decision.confidence:.2f} < {self.config.min_confidence_for_hedging:.2f})",
                event_name=main_decision.event_name,
                market_name=main_decision.market_name,
                is_hedge=True,
                hedge_for=main_decision.ticker,
                hedge_ratio=self.config.hedge_ratio
            )
            
            hedge_decisions.append(hedge_decision)
        
        return hedge_decisions
    
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
        table.add_column("Type", style="bright_blue", justify="center", width=8)
        table.add_column("Market", style="cyan", width=40)
        table.add_column("Action", style="yellow", justify="center", width=10)
        table.add_column("Confidence", style="magenta", justify="right", width=10)
        table.add_column("Amount", style="green", justify="right", width=10)
        table.add_column("Reasoning", style="blue", width=70)
        
        for decision in actionable_decisions:
            # Use human-readable market name if available, otherwise generate from ticker
            market_display = decision.market_name if decision.market_name else self._generate_readable_market_name(decision.ticker)
            
            # Determine bet type
            bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"
            
            table.add_row(
                bet_type,
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
                
                # Calculate mid-prices with validation
                # Only calculate mid-price if both bid and ask are > 0
                yes_mid_price = None
                no_mid_price = None
                
                if yes_bid > 0 and yes_ask > 0:
                    yes_mid_price = (yes_bid + yes_ask) / 2
                elif yes_ask > 0:
                    # If only ask is available, use ask price as approximation
                    yes_mid_price = yes_ask
                elif yes_bid > 0:
                    # If only bid is available, use bid price as approximation
                    yes_mid_price = yes_bid
                
                if no_bid > 0 and no_ask > 0:
                    no_mid_price = (no_bid + no_ask) / 2
                elif no_ask > 0:
                    # If only ask is available, use ask price as approximation
                    no_mid_price = no_ask
                elif no_bid > 0:
                    # If only bid is available, use bid price as approximation
                    no_mid_price = no_bid
                
                # Log warning if we couldn't calculate proper mid-prices
                if yes_mid_price is None or no_mid_price is None:
                    logger.warning(f"Market {ticker}: Missing bid/ask data - yes_bid={yes_bid}, yes_ask={yes_ask}, no_bid={no_bid}, no_ask={no_ask}")
                    # Fallback: use 0 for missing prices
                    yes_mid_price = yes_mid_price or 0
                    no_mid_price = no_mid_price or 0
                
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
        
        # Skip one-sided markets where either YES or NO mid price is missing/zero
        filtered_markets_with_odds = []
        for md in markets_with_odds:
            yes_mid = md.get('yes_mid_price')
            no_mid = md.get('no_mid_price')
            if yes_mid in (None, 0) or no_mid in (None, 0):
                logger.info(f"Skipping market {md.get('ticker', '')}: missing yes/no price (yes_mid={yes_mid}, no_mid={no_mid})")
                continue
            filtered_markets_with_odds.append(md)
        markets_with_odds = filtered_markets_with_odds

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
        
        RISK-ADJUSTED FILTERING (HEDGE-FUND STYLE):
        - Only place bets with strong statistical edge: R-score (z-score) >= {self.config.z_threshold}
        - R-score measures how many standard deviations away the market is from fair value
        - Higher R-scores indicate higher conviction opportunities with better risk-adjusted returns
        - Example: R-score of 2.0 means the market is 2 standard deviations mis-priced (97.5th percentile opportunity)
        - SKIP all bets with R-score below the threshold - focus on exceptional statistical opportunities
        
        POSITION SIZING STRATEGY:
        - Use Kelly criterion for optimal position sizing based on edge and risk
        - Higher R-score opportunities get larger position sizes (within risk limits)
        - Maximum position size capped at ${self.config.max_bet_amount} and {self.config.max_kelly_bet_fraction*100}% of bankroll
        - Most markets: SKIP - only bet on the highest R-score opportunities
        
        RISK-ADJUSTED EDGE CALCULATION:
        - Bot automatically calculates R-score (z-score) for statistical edge measurement
        - R-score accounts for both probability difference AND volatility/risk
        - Higher R-scores indicate better risk-adjusted opportunities
        - Kelly sizing automatically optimizes position size based on edge and risk
        
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
            analysis = self._apply_alpha_threshold_validation(analysis, event_ticker, markets, probability_extraction, market_odds)
            
            # Apply strategic filtering to ensure selective betting
            analysis = self._apply_strategic_filtering(analysis, event_ticker)
            
            # Apply portfolio selection to hold only the N highest R-scores
            analysis = self.apply_portfolio_selection(analysis, event_ticker)
            
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
        """Add human-readable names to betting decisions."""
        # Create lookup for market names
        market_names = {market.get('ticker', ''): market.get('title', '') for market in markets}
        
        # Update each decision with readable names
        for decision in analysis.decisions:
            decision.event_name = event_info.get('title', '')
            decision.market_name = market_names.get(decision.ticker, decision.ticker)
        
        return analysis
    
    def _generate_readable_market_name(self, ticker: str) -> str:
        """Generate a readable market name from ticker."""
        return ticker.replace('-', ' ').replace('_', ' ').title()
    
    def _enforce_mutually_exclusive_constraint(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Ensure only one YES bet for mutually exclusive events."""
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        yes_bets = [d for d in actionable_decisions if d.action == "buy_yes"]
        
        if len(yes_bets) <= 1:
            # Already compliant or no YES bets
            return analysis
        
        # Multiple YES bets found - keep only the highest confidence one
        logger.warning(f"Event {event_ticker}: Multiple YES bets found in mutually exclusive event. Keeping only highest confidence bet.")
        
        # Sort YES bets by confidence (highest first)
        yes_bets.sort(key=lambda x: x.confidence, reverse=True)
        best_yes_bet = yes_bets[0]
        
        # Convert other YES bets to SKIP
        filtered_decisions = []
        for decision in analysis.decisions:
            if decision.action == "buy_yes" and decision.ticker != best_yes_bet.ticker:
                # Convert to skip
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to mutually exclusive constraint (kept {best_yes_bet.ticker} with higher confidence)",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(skip_decision)
            else:
                filtered_decisions.append(decision)
        
        # Update the analysis
        analysis.decisions = filtered_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in filtered_decisions)
        analysis.high_confidence_bets = len([d for d in filtered_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    def _apply_alpha_threshold_validation(self, analysis: MarketAnalysis, event_ticker: str, markets: List[Dict[str, Any]], probability_extraction: ProbabilityExtraction, market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Apply risk-adjusted threshold validation and enrich decisions with metrics."""
        validated_decisions = []
        
        for decision in analysis.decisions:
            if decision.action == "skip":
                validated_decisions.append(decision)
                continue
            
            # Find the market probability for this decision
            market_prob = None
            for market_data in probability_extraction.markets:
                if market_data.ticker == decision.ticker:
                    market_prob = market_data.research_probability
                    break
            
            if market_prob is None:
                # If we can't find probability data, skip the bet
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to missing probability data",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                validated_decisions.append(skip_decision)
                continue
            
            # Find current market price from market_odds dictionary
            market_odds_data = None
            if decision.ticker in market_odds:
                ticker_odds = market_odds[decision.ticker]
                if decision.action == "buy_yes":
                    market_odds_data = ticker_odds.get('yes_ask', 0) / 100.0  # Convert to probability
                elif decision.action == "buy_no":
                    market_odds_data = ticker_odds.get('no_ask', 0) / 100.0  # Convert to probability
            
            if market_odds_data is None or market_odds_data == 0:
                # If we can't find market odds, skip the bet
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to missing market odds",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                validated_decisions.append(skip_decision)
                continue
            
            # Convert probabilities to 0-1 range
            research_prob = market_prob / 100.0
            
            # Calculate risk-adjusted metrics
            risk_metrics = self.calculate_risk_adjusted_metrics(
                research_prob, market_odds_data, decision.action
            )
            
            # Apply threshold filtering - use R-score filtering by default
            should_accept = False
            rejection_reason = ""
            
            # Use R-score (z-score) filtering - the new standard
            if risk_metrics["r_score"] >= self.config.z_threshold:
                should_accept = True
            else:
                rejection_reason = f"R-score {risk_metrics['r_score']:.2f} below z-threshold {self.config.z_threshold:.2f}"
            
            if should_accept:
                # Calculate Kelly position size if enabled
                if self.config.enable_kelly_sizing:
                    kelly_size = self.calculate_kelly_position_size(risk_metrics["kelly_fraction"])
                    decision.amount = kelly_size
                
                # Enrich decision with risk metrics
                decision.expected_return = risk_metrics["expected_return"]
                decision.r_score = risk_metrics["r_score"]
                decision.kelly_fraction = risk_metrics["kelly_fraction"]
                decision.market_price = market_odds_data
                decision.research_probability = research_prob
                
                validated_decisions.append(decision)
            else:
                # Convert to skip if threshold not met
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped: {rejection_reason}",
                    event_name=decision.event_name,
                    market_name=decision.market_name,
                    expected_return=risk_metrics["expected_return"],
                    r_score=risk_metrics["r_score"],
                    kelly_fraction=risk_metrics["kelly_fraction"],
                    market_price=market_odds_data,
                    research_probability=research_prob
                )
                validated_decisions.append(skip_decision)
                logger.info(f"Filtered out {decision.ticker} - {rejection_reason}")
        
        # Update analysis with validated decisions
        analysis.decisions = validated_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in validated_decisions if d.action != "skip")
        analysis.high_confidence_bets = len([d for d in validated_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    def _apply_strategic_filtering(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Apply strategic filtering to ensure selective betting."""
        filtered_decisions = []
        
        for decision in analysis.decisions:
            if decision.action == "skip":
                filtered_decisions.append(decision)
                continue
            
            # Apply confidence threshold filter
            if decision.confidence < 0.6:
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to low confidence: {decision.confidence:.2f} < 0.6",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(skip_decision)
                continue
            
            # Apply bet amount limits
            max_bet = min(decision.amount, self.config.max_bet_amount)
            if max_bet != decision.amount:
                # Adjust bet amount to maximum allowed
                adjusted_decision = BettingDecision(
                    ticker=decision.ticker,
                    action=decision.action,
                    confidence=decision.confidence,
                    amount=max_bet,
                    reasoning=f"Amount adjusted to max limit: {decision.reasoning}",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(adjusted_decision)
            else:
                filtered_decisions.append(decision)
        
        # Update analysis with filtered decisions
        analysis.decisions = filtered_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in filtered_decisions if d.action != "skip")
        analysis.high_confidence_bets = len([d for d in filtered_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    async def place_bets(self, analysis: MarketAnalysis, market_odds: Dict[str, Dict[str, Any]], probability_extractions: Dict[str, ProbabilityExtraction]):
        """Place bets based on the analysis."""
        self.console.print(f"\n[bold]Step 6: Placing bets...[/bold]")
        
        if not analysis.decisions:
            self.console.print("[yellow]No betting decisions to execute[/yellow]")
            return
        
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        
        if not actionable_decisions:
            self.console.print("[yellow]No actionable betting decisions to execute[/yellow]")
            return
        
        self.console.print(f"Found {len(actionable_decisions)} actionable decisions")
        
        for decision in actionable_decisions:
            if self.config.dry_run:
                self.console.print(f"[blue]DRY RUN: Would place {decision.action} bet of ${decision.amount} on {decision.ticker}[/blue]")
            else:
                # Convert action to Kalshi side format
                side = "yes" if decision.action == "buy_yes" else "no"
                result = await self.kalshi_client.place_order(decision.ticker, side, decision.amount)
                
                if result.get("success"):
                    self.console.print(f"[green]✓ Placed {decision.action} bet of ${decision.amount} on {decision.ticker}[/green]")
                else:
                    self.console.print(f"[red]✗ Failed to place bet on {decision.ticker}: {result.get('error', 'Unknown error')}[/red]")
        
        if self.config.dry_run:
            self.console.print("\n[yellow]DRY RUN MODE: No actual bets were placed[/yellow]")
        else:
            self.console.print(f"\n[green]✓ Completed bet placement[/green]")
 
    def save_betting_decisions_to_csv(self, 
                                     analysis: MarketAnalysis, 
                                     research_results: Dict[str, str],
                                     probability_extractions: Dict[str, ProbabilityExtraction],
                                     market_odds: Dict[str, Dict[str, Any]],
                                     event_markets: Dict[str, Dict[str, Any]]) -> str:
        """
        Save betting decisions to a timestamped CSV file including raw research data.
        
        Args:
            analysis: The final betting decisions
            research_results: Raw research results by event ticker
            probability_extractions: Structured probability data by event ticker
            market_odds: Current market odds
            event_markets: Event and market information
            
        Returns:
            str: Path to the created CSV file
        """
        # Create output directory
        output_dir = Path("betting_decisions")
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"betting_decisions_{timestamp}.csv"
        filepath = output_dir / filename
        
        # Prepare CSV data
        csv_data = []
        
        for decision in analysis.decisions:
            # Find corresponding research and market data
            event_ticker = None
            raw_research = ""
            research_summary = ""
            research_probability = None
            research_reasoning = ""
            market_yes_price = None
            market_no_price = None
            event_title = ""
            market_title = ""
            
            # Find the event ticker for this market
            for evt_ticker, data in event_markets.items():
                for market in data['markets']:
                    if market.get('ticker') == decision.ticker:
                        event_ticker = evt_ticker
                        event_title = data['event'].get('title', '')
                        market_title = market.get('title', '')
                        break
                if event_ticker:
                    break
            
            # Get raw research
            if event_ticker and event_ticker in research_results:
                raw_research = research_results[event_ticker]
            
            # Get probability extraction data
            if event_ticker and event_ticker in probability_extractions:
                extraction = probability_extractions[event_ticker]
                research_summary = extraction.overall_summary
                
                # Find market-specific probability
                for market_prob in extraction.markets:
                    if market_prob.ticker == decision.ticker:
                        research_probability = market_prob.research_probability
                        research_reasoning = market_prob.reasoning
                        break
            
            # Get market odds
            if decision.ticker in market_odds:
                odds = market_odds[decision.ticker]
                
                # Calculate mid-prices from bid/ask spreads
                yes_bid = odds.get('yes_bid', 0)
                no_bid = odds.get('no_bid', 0)
                yes_ask = odds.get('yes_ask', 0)
                no_ask = odds.get('no_ask', 0)
                
                # Calculate mid-prices (same logic as used elsewhere in the bot)
                if yes_bid > 0 and yes_ask > 0:
                    market_yes_price = (yes_bid + yes_ask) / 2
                if no_bid > 0 and no_ask > 0:
                    market_no_price = (no_bid + no_ask) / 2
            
            # Skip one-sided markets in CSV output
            if (market_yes_price is None or market_yes_price == 0) or (market_no_price is None or market_no_price == 0):
                # No valid market on one side; exclude from CSV
                continue
            
            # Legacy edge calculation removed - now using R-score instead
            
            csv_row = {
                # Basic info
                'timestamp': datetime.now().isoformat(),
                'event_ticker': event_ticker or '',
                'event_title': event_title,
                'market_ticker': decision.ticker,
                'market_title': market_title,
                
                # Decision details
                'action': decision.action,
                'bet_amount': decision.amount,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                
                # Research data (original scale 0-100%)
                'research_probability': research_probability,
                'research_reasoning': research_reasoning,
                
                # Market data (original scale 0-100 cents)
                'market_yes_price': market_yes_price,
                'market_no_price': market_no_price,
                
                # Risk-adjusted metrics (hedge-fund style)
                'expected_return': decision.expected_return,
                'r_score': decision.r_score,
                'kelly_fraction': decision.kelly_fraction,
                
                # Normalized probabilities used in calculations (0-1 scale)
                'calc_market_prob': decision.market_price,  # Market probability used in R-score calc
                'calc_research_prob': decision.research_probability,  # Research probability used in R-score calc
                
                # Hedging info
                'is_hedge': decision.is_hedge,
                'hedge_for': decision.hedge_for or '',
                
                # Research context
                'research_summary': research_summary,
                'raw_research': raw_research.replace('\n', ' ').replace('\r', ' ') if raw_research else ''
            }
            csv_data.append(csv_row)
        
        # Write to CSV
        if csv_data:
            fieldnames = [
                # Basic info
                'timestamp', 'event_ticker', 'event_title', 'market_ticker', 'market_title',
                # Decision details  
                'action', 'bet_amount', 'confidence', 'reasoning',
                # Research data (original scale 0-100%)
                'research_probability', 'research_reasoning',
                # Market data (original scale 0-100 cents)
                'market_yes_price', 'market_no_price',
                # Risk-adjusted metrics (hedge-fund style)
                'expected_return', 'r_score', 'kelly_fraction',
                # Normalized probabilities used in calculations (0-1 scale)
                'calc_market_prob', 'calc_research_prob',
                # Hedging info
                'is_hedge', 'hedge_for',
                # Research context
                'research_summary', 'raw_research'
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"Saved {len(csv_data)} betting decisions to {filepath}")
            self.console.print(f"[bold green]✓[/bold green] Betting decisions saved to: [blue]{filepath}[/blue]")
        else:
            logger.warning("No betting decisions to save")
            self.console.print("[yellow]No betting decisions to save[/yellow]")
        
        return str(filepath)

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
            
            # Save betting decisions to CSV with research data
            self.save_betting_decisions_to_csv(
                analysis=analysis,
                research_results=research_results,
                probability_extractions=probability_extractions, 
                market_odds=market_odds,
                event_markets=event_markets
            )
            
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
    Z_THRESHOLD=1.5                    # Minimum R-score (z-score) for betting (default: 1.5)
    KELLY_FRACTION=0.5                 # Fraction of Kelly to use for position sizing (default: 0.5)
    BANKROLL=1000.0                    # Total bankroll for Kelly calculations (default: 1000.0)
    
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