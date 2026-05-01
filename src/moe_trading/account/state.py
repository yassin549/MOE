"""Immutable account domain objects used by policy, backtesting, and live execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AccountPhase(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"
    SIM = "sim"


@dataclass(frozen=True, slots=True)
class PositionState:
    position_id: str
    asset: str
    direction: int
    entry_price: float
    stop_price: float
    risk_fraction: float
    opened_at: str
    unrealized_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class AccountState:
    phase: AccountPhase = AccountPhase.CHALLENGE
    starting_balance: float = 100_000.0
    current_equity: float = 100_000.0
    current_balance: float = 100_000.0
    open_unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    realized_pnl_overall: float = 0.0
    daily_peak_equity: float = 100_000.0
    overall_peak_equity: float = 100_000.0
    consecutive_profitable_days: int = 0
    profitable_day_count: int = 0
    days_elapsed: int = 0
    weekend_holding_permission: bool = True
    news_permission: bool = True
    max_concurrent_positions: int = 2
    current_open_risk: float = 0.0
    open_positions: tuple[PositionState, ...] = field(default_factory=tuple)
    challenge_passed: bool = False
    breached: bool = False
    breach_reason: str | None = None
    last_day: str | None = None
