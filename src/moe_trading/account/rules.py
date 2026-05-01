"""Pure prop-rule engine shared by simulation and live execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from moe_trading.account.state import AccountPhase, AccountState, PositionState
from moe_trading.config import PropConfig, PropPhaseConfig


@dataclass(frozen=True, slots=True)
class TradeIntent:
    asset: str
    direction: int
    entry_price: float
    stop_price: float
    risk_fraction: float
    opened_at: str
    hold_through_weekend: bool = False


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    allowed: bool
    capped_risk_fraction: float
    reasons: tuple[str, ...]
    must_flatten: bool = False


class PropRuleEngine:
    """Deterministic account-controller rules for challenge and funded phases."""

    def __init__(self, config: PropConfig) -> None:
        self.config = config

    def current_phase_rules(self, state: AccountState) -> PropPhaseConfig:
        if state.phase == AccountPhase.FUNDED:
            return self.config.funded
        return self.config.challenge

    def remaining_profit_target(self, state: AccountState) -> float:
        rules = self.current_phase_rules(state)
        if rules.profit_target is None:
            return 0.0
        target_amount = rules.profit_target * state.starting_balance
        progress = max(state.current_equity - state.starting_balance, 0.0)
        return max(target_amount - progress, 0.0)

    def remaining_daily_loss_buffer(self, state: AccountState) -> float:
        rules = self.current_phase_rules(state)
        used = max(state.daily_peak_equity - state.current_equity, 0.0)
        return max((rules.daily_loss_limit * state.starting_balance) - used, 0.0)

    def remaining_total_loss_buffer(self, state: AccountState) -> float:
        rules = self.current_phase_rules(state)
        used = max(state.starting_balance - state.current_equity, 0.0)
        return max((rules.overall_loss_limit * state.starting_balance) - used, 0.0)

    def required_flatten(self, state: AccountState, timestamp: str, pre_weekend: bool = False) -> RuleEvaluation:
        rules = self.current_phase_rules(state)
        reasons: list[str] = []
        must_flatten = False
        if pre_weekend and state.open_positions and not rules.allow_weekend_holding:
            reasons.append("weekend_holding_not_allowed")
            must_flatten = True
        if state.breached:
            reasons.append(state.breach_reason or "account_breached")
            must_flatten = must_flatten or bool(state.open_positions)
        return RuleEvaluation(
            allowed=not reasons,
            capped_risk_fraction=0.0,
            reasons=tuple(reasons),
            must_flatten=must_flatten,
        )

    def validate_trade(self, state: AccountState, trade: TradeIntent) -> RuleEvaluation:
        rules = self.current_phase_rules(state)
        reasons: list[str] = []
        daily_remaining_fraction = self.remaining_daily_loss_buffer(state) / state.starting_balance
        total_remaining_fraction = self.remaining_total_loss_buffer(state) / state.starting_balance
        aggregate_remaining = max(rules.max_aggregate_open_risk - state.current_open_risk, 0.0)
        capped_risk = min(
            trade.risk_fraction,
            rules.per_trade_loss_limit,
            daily_remaining_fraction,
            total_remaining_fraction,
            aggregate_remaining,
        )

        if state.breached:
            reasons.append(state.breach_reason or "account_breached")
        if state.phase == AccountPhase.CHALLENGE and state.challenge_passed:
            reasons.append("challenge_already_passed")
        if len(state.open_positions) >= rules.max_concurrent_positions:
            reasons.append("max_concurrent_positions_reached")
        if trade.risk_fraction > rules.per_trade_loss_limit:
            reasons.append("per_trade_loss_limit_exceeded")
        if trade.risk_fraction > aggregate_remaining:
            reasons.append("aggregate_open_risk_exceeded")
        if trade.risk_fraction > daily_remaining_fraction:
            reasons.append("daily_loss_buffer_exceeded")
        if trade.risk_fraction > total_remaining_fraction:
            reasons.append("overall_loss_buffer_exceeded")
        if trade.hold_through_weekend and not rules.allow_weekend_holding:
            reasons.append("weekend_holding_not_allowed")
        if trade.entry_price == trade.stop_price:
            reasons.append("invalid_stop_distance")

        return RuleEvaluation(
            allowed=not reasons,
            capped_risk_fraction=max(capped_risk, 0.0),
            reasons=tuple(reasons),
        )

    def mark_to_market(self, state: AccountState, open_unrealized_pnl: float) -> AccountState:
        equity = state.current_balance + open_unrealized_pnl
        updated = replace(
            state,
            open_unrealized_pnl=open_unrealized_pnl,
            current_equity=equity,
            daily_peak_equity=max(state.daily_peak_equity, equity),
            overall_peak_equity=max(state.overall_peak_equity, equity),
        )
        return self.evaluate_state(updated)

    def open_position(self, state: AccountState, trade: TradeIntent, position_id: str) -> AccountState:
        evaluation = self.validate_trade(state, trade)
        if not evaluation.allowed:
            raise ValueError(f"Trade not allowed: {', '.join(evaluation.reasons)}")
        position = PositionState(
            position_id=position_id,
            asset=trade.asset,
            direction=trade.direction,
            entry_price=trade.entry_price,
            stop_price=trade.stop_price,
            risk_fraction=trade.risk_fraction,
            opened_at=trade.opened_at,
        )
        updated = replace(
            state,
            current_open_risk=state.current_open_risk + trade.risk_fraction,
            open_positions=state.open_positions + (position,),
        )
        return self.evaluate_state(updated)

    def close_position(self, state: AccountState, position_id: str, realized_pnl: float) -> AccountState:
        remaining_positions = tuple(position for position in state.open_positions if position.position_id != position_id)
        closed_positions = [position for position in state.open_positions if position.position_id == position_id]
        if not closed_positions:
            raise ValueError(f"Unknown position_id: {position_id}")
        released_risk = sum(position.risk_fraction for position in closed_positions)
        new_balance = state.current_balance + realized_pnl
        new_today = state.realized_pnl_today + realized_pnl
        new_overall = state.realized_pnl_overall + realized_pnl
        new_equity = new_balance + state.open_unrealized_pnl
        updated = replace(
            state,
            current_balance=new_balance,
            current_equity=new_equity,
            realized_pnl_today=new_today,
            realized_pnl_overall=new_overall,
            current_open_risk=max(state.current_open_risk - released_risk, 0.0),
            open_positions=remaining_positions,
            daily_peak_equity=max(state.daily_peak_equity, new_equity),
            overall_peak_equity=max(state.overall_peak_equity, new_equity),
        )
        return self.evaluate_state(updated)

    def roll_day(self, state: AccountState, timestamp: str) -> AccountState:
        current_day = self._calendar_day(timestamp)
        if state.last_day is None:
            return replace(state, last_day=current_day)
        if current_day == state.last_day:
            return state

        profitable_day = state.realized_pnl_today > 0.0
        updated = replace(
            state,
            realized_pnl_today=0.0,
            daily_peak_equity=state.current_equity,
            consecutive_profitable_days=state.consecutive_profitable_days + 1 if profitable_day else 0,
            profitable_day_count=state.profitable_day_count + int(profitable_day),
            days_elapsed=state.days_elapsed + 1,
            last_day=current_day,
        )
        return self.evaluate_state(updated)

    def promote_to_funded(self, state: AccountState) -> AccountState:
        if not state.challenge_passed:
            raise ValueError("Challenge is not yet passed.")
        funded_rules = self.config.funded
        updated = replace(
            state,
            phase=AccountPhase.FUNDED,
            weekend_holding_permission=funded_rules.allow_weekend_holding,
            news_permission=funded_rules.allow_news_trading,
            max_concurrent_positions=funded_rules.max_concurrent_positions,
            challenge_passed=False,
            breached=False,
            breach_reason=None,
            daily_peak_equity=state.current_equity,
            overall_peak_equity=max(state.overall_peak_equity, state.current_equity),
        )
        return self.evaluate_state(updated)

    def evaluate_state(self, state: AccountState) -> AccountState:
        rules = self.current_phase_rules(state)
        daily_breached = self.remaining_daily_loss_buffer(state) <= 0.0
        total_breached = self.remaining_total_loss_buffer(state) <= 0.0
        challenge_passed = state.challenge_passed

        breach_reason = None
        breached = state.breached
        if daily_breached:
            breached = True
            breach_reason = "daily_loss_limit_breached"
        elif total_breached:
            breached = True
            breach_reason = "overall_loss_limit_breached"
        elif breached:
            breach_reason = state.breach_reason

        if state.phase == AccountPhase.CHALLENGE and rules.profit_target is not None:
            target_hit = (state.current_equity - state.starting_balance) >= (rules.profit_target * state.starting_balance)
            profitable_days_hit = state.profitable_day_count >= rules.minimum_profitable_days
            challenge_passed = target_hit and profitable_days_hit and not breached

        return replace(
            state,
            weekend_holding_permission=rules.allow_weekend_holding,
            news_permission=rules.allow_news_trading,
            max_concurrent_positions=rules.max_concurrent_positions,
            challenge_passed=challenge_passed,
            breached=breached,
            breach_reason=breach_reason,
        )

    @staticmethod
    def _calendar_day(timestamp: str) -> str:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()
