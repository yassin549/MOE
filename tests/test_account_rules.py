from moe_trading.account.rules import PropRuleEngine, TradeIntent
from moe_trading.account.state import AccountPhase, AccountState
from moe_trading.config import PropConfig


def _engine() -> PropRuleEngine:
    return PropRuleEngine(PropConfig())


def test_validate_trade_caps_risk_by_rule_buffers():
    engine = _engine()
    state = AccountState(
        current_equity=96_000.0,
        current_balance=96_000.0,
        daily_peak_equity=100_000.0,
        overall_peak_equity=100_000.0,
        current_open_risk=0.02,
    )
    trade = TradeIntent(
        asset="US100",
        direction=1,
        entry_price=20_000.0,
        stop_price=19_900.0,
        risk_fraction=0.03,
        opened_at="2026-04-29T10:00:00+00:00",
    )

    evaluation = engine.validate_trade(state, trade)

    assert not evaluation.allowed
    assert "daily_loss_buffer_exceeded" in evaluation.reasons
    assert evaluation.capped_risk_fraction == 0.01


def test_roll_day_tracks_profitable_days_and_resets_intraday_pnl():
    engine = _engine()
    state = AccountState(
        realized_pnl_today=1_250.0,
        last_day="2026-04-28",
        current_equity=101_250.0,
        current_balance=101_250.0,
        daily_peak_equity=101_250.0,
        overall_peak_equity=101_250.0,
    )

    rolled = engine.roll_day(state, "2026-04-29T00:01:00+00:00")

    assert rolled.realized_pnl_today == 0.0
    assert rolled.profitable_day_count == 1
    assert rolled.consecutive_profitable_days == 1
    assert rolled.days_elapsed == 1
    assert rolled.last_day == "2026-04-29"


def test_challenge_pass_requires_target_and_profitable_days():
    engine = _engine()
    state = AccountState(
        current_equity=110_500.0,
        current_balance=110_500.0,
        realized_pnl_overall=10_500.0,
        profitable_day_count=3,
        daily_peak_equity=110_500.0,
        overall_peak_equity=110_500.0,
    )

    evaluated = engine.evaluate_state(state)

    assert evaluated.phase == AccountPhase.CHALLENGE
    assert evaluated.challenge_passed
    assert not evaluated.breached


def test_daily_loss_breach_sets_account_breached():
    engine = _engine()
    state = AccountState(
        current_equity=95_000.0,
        current_balance=95_000.0,
        daily_peak_equity=100_000.0,
        overall_peak_equity=100_000.0,
    )

    evaluated = engine.evaluate_state(state)

    assert evaluated.breached
    assert evaluated.breach_reason == "daily_loss_limit_breached"


def test_funded_phase_requires_weekend_flatten():
    engine = _engine()
    challenge_state = AccountState(
        current_equity=111_000.0,
        current_balance=111_000.0,
        profitable_day_count=3,
        challenge_passed=True,
        daily_peak_equity=111_000.0,
        overall_peak_equity=111_000.0,
    )
    funded_state = engine.promote_to_funded(challenge_state)
    trade = TradeIntent(
        asset="US500",
        direction=-1,
        entry_price=5_200.0,
        stop_price=5_240.0,
        risk_fraction=0.01,
        opened_at="2026-04-29T14:00:00+00:00",
    )
    funded_state = engine.open_position(funded_state, trade, position_id="pos-1")

    evaluation = engine.required_flatten(funded_state, "2026-05-01T20:59:00+00:00", pre_weekend=True)

    assert funded_state.phase == AccountPhase.FUNDED
    assert evaluation.must_flatten
    assert "weekend_holding_not_allowed" in evaluation.reasons
