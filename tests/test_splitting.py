import pandas as pd

from moe_trading.data.splitting import make_time_split


def test_make_time_split_respects_order_and_embargo():
    frame = pd.DataFrame({"value": range(100)})
    split = make_time_split(frame, validation_ratio=0.2, test_ratio=0.2, embargo_bars=5)
    assert split.train["value"].max() == 59
    assert split.validation["value"].min() == 65
    assert split.test["value"].min() == 85
