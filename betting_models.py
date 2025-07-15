"""
Pydantic models for structured betting decisions.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class MarketProbability(BaseModel):
    """Structured probability data for a single market."""
    ticker: str = Field(..., description="The market ticker symbol")
    title: str = Field(..., description="Human-readable market title")
    research_probability: float = Field(..., ge=0, le=100, description="Research predicted probability (0-100)")
    reasoning: str = Field(..., description="Brief reasoning for the probability estimate")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the probability estimate (0-1)")


class ProbabilityExtraction(BaseModel):
    """Structured extraction of probabilities from research."""
    markets: List[MarketProbability] = Field(..., description="List of market probabilities")
    overall_summary: str = Field(..., description="Overall research summary and key insights")


class BettingDecision(BaseModel):
    """A single betting decision for a market."""
    ticker: str = Field(..., description="The market ticker symbol")
    action: Literal["buy_yes", "buy_no", "skip"] = Field(..., description="Action to take")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in decision (0-1)")
    amount: float = Field(..., ge=0, description="Amount to bet in dollars")
    reasoning: str = Field(..., description="Brief reasoning for the decision")
    
    # Human-readable names for display
    event_name: Optional[str] = Field(None, description="Human-readable event name")
    market_name: Optional[str] = Field(None, description="Human-readable market name")
    
    # Hedging fields
    is_hedge: bool = Field(False, description="Whether this is a hedge bet")
    hedge_for: Optional[str] = Field(None, description="Ticker of the main bet this hedges")
    hedge_ratio: Optional[float] = Field(None, ge=0, le=1, description="Proportion of main bet this hedges")


class MarketAnalysis(BaseModel):
    """Analysis results for all markets."""
    decisions: List[BettingDecision] = Field(..., description="List of betting decisions")
    total_recommended_bet: float = Field(..., description="Total amount recommended to bet")
    high_confidence_bets: int = Field(..., description="Number of high confidence bets (>0.7)")
    summary: str = Field(..., description="Overall market summary and strategy") 