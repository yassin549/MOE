from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.backtesting.engine import REMOVAL_MESSAGE


if __name__ == "__main__":
    raise SystemExit(
        REMOVAL_MESSAGE
        + " The old array-example script has been retired with the previous engine."
    )
