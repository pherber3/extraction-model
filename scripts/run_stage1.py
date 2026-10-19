"""Stage 1 from a terminal or the Colab CLI.

    uv run python scripts/run_stage1.py --dry-run          # CPU smoke test
    python scripts/run_stage1.py --out /content/stage1     # on the Colab VM (A100 80 GB)
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction.stage1 import Stage1Config, run  # noqa: E402


def main() -> None:
    defaults = Stage1Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--mix-layers", type=int, nargs="+", default=list(defaults.mix_layers))
    parser.add_argument("--out", default=defaults.out_dir)
    args = parser.parse_args()
    run(
        Stage1Config(
            dry_run=args.dry_run,
            epochs=args.epochs,
            lr=args.lr,
            max_tokens=args.max_tokens,
            mix_layers=tuple(args.mix_layers),
            out_dir=args.out,
        )
    )


if __name__ == "__main__":
    main()
