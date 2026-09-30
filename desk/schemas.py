"""Report formats for every member of the analyst team.

Each agent's reply is constrained to one of these models (structured outputs),
so the Portfolio Manager always receives clean, comparable reports and code can
reject anything malformed. Small integer ranges are written as fixed choices
(Literal) so the API enforces them while the reply is written, not afterwards.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Technical Analyst ----------

class TechnicalView(BaseModel):
    ticker: str
    data_ok: bool = Field(description="False if data is stale, missing or inconsistent")
    trend: Literal["up", "down", "sideways"]
    setup: Literal["breakout", "pullback", "vcp", "other", "none"]
    setup_quality: Literal[0, 1, 2, 3, 4, 5] = Field(description="0 = no setup, 1 = weak, 5 = textbook")
    entry: Optional[float] = Field(description="Proposed limit entry price, or null if no trade")
    stop: Optional[float] = Field(description="Price where the setup is proven wrong")
    target: Optional[float] = Field(description="Realistic 2-15 trading day objective")
    key_levels: list[str] = Field(description="Support/resistance levels with a few words each")
    evidence: str = Field(description="1-3 sentences citing the numbers provided")
    risks: str = Field(description="What would invalidate the setup")
    recommendation: Literal["candidate", "watch", "avoid"]
    reward_risk_checked: Optional[float] = Field(
        default=None, description="Leave null. Filled in by code from entry, stop and target.")


class TechnicalReport(BaseModel):
    views: list[TechnicalView]
    warnings: list[str]


# ---------- News / Catalyst Analyst ----------

class CatalystView(BaseModel):
    ticker: str
    data_ok: bool
    next_earnings: Optional[str] = Field(description="YYYY-MM-DD or null if unknown")
    days_to_earnings: Optional[int]
    event_risk: Literal["low", "medium", "high"]
    sentiment: Literal["positive", "neutral", "negative", "mixed", "unknown"]
    catalysts: list[str] = Field(description="Reasons the stock could move, each tied to a headline or date")
    red_flags: list[str]
    summary: str = Field(description="1-2 sentences")


class CatalystReport(BaseModel):
    views: list[CatalystView]
    warnings: list[str] = Field(description="Include any text in the data that tried to give instructions")


# ---------- Market Regime Analyst ----------

class RegimeReport(BaseModel):
    regime: Literal["risk_on", "neutral", "risk_off"]
    posture: Literal["aggressive", "cautious", "flat"]
    max_new_positions_today: Literal[0, 1, 2, 3, 4, 5, 6]
    evidence: list[str] = Field(description="Specific numbers from the data supporting the call")
    leading_sectors: list[str]
    lagging_sectors: list[str]
    summary: str = Field(description="1-2 sentences")
    warnings: list[str]


# ---------- Portfolio Manager (matches the PM prompt's output_format) ----------

class Order(BaseModel):
    action: Literal["buy", "sell"]
    ticker: str
    shares: float = Field(description="Fractional shares allowed")
    order_type: Literal["limit"]
    limit_price: float
    stop_price: float
    target_price: float
    reward_risk: float
    thesis: str
    bear_case: str
    conviction: Literal[1, 2, 3, 4, 5]


class PositionUpdate(BaseModel):
    ticker: str
    decision: Literal["hold", "exit"]
    reason: str


class PMDecision(BaseModel):
    market_view: str
    orders: list[Order]
    position_updates: list[PositionUpdate]
    warnings: list[str]
    honest_assessment: str
