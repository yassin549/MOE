from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_trading.config import load_config
from moe_trading.training.pipeline import run_training_pipeline


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "configs" / "base.yaml"
    run_training_pipeline(load_config(config_path))
