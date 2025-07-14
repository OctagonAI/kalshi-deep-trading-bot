"""
Pydantic models for structured betting decisions.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class BettingDecision(BaseModel):
    """A single betting decision for a market."""
    ticker: str = Field(..., description="The market ticker symbol")
    action: Literal["buy_yes", "buy_no", "skip"] = Field(..., description="Action to take")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in decision (0-1)")
    amount: float = Field(..., ge=0, description="Amount to bet in dollars")
    reasoning: str = Field(..., description="Brief reasoning for the decision")


class MarketAnalysis(BaseModel):
    """Analysis results for all markets."""
    decisions: List[BettingDecision] = Field(..., description="List of betting decisions")
    total_recommended_bet: float = Field(..., description="Total amount recommended to bet")
    high_confidence_bets: int = Field(..., description="Number of high confidence bets (>0.7)")
    summary: str = Field(..., description="Overall market summary and strategy") 