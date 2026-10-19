"""Stage 0 from a terminal or the Colab CLI.

    uv run python scripts/run_stage0.py --dry-run          # CPU smoke test
    python scripts/run_stage0.py --out /content/stage0     # on the Colab VM
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extraction.stage0 import Stage0Config, run  # noqa: E402


def main() -> None:
    defaults = Stage0Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=defaults.model_id)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-synth", type=int, default=defaults.n_synth)
    parser.add_argument("--n-squad", type=int, default=defaults.n_squad)
    parser.add_argument("--n-gretel", type=int, default=defaults.n_gretel)
    parser.add_argument("--n-eval", type=int, default=defaults.n_eval)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--layers", type=int, nargs="*", help="1-based layer indices (default: all)")
    parser.add_argument("--out", default=defaults.out_dir)
    args = parser.parse_args()
    run(
        Stage0Config(
            model_id=args.model,
            dry_run=args.dry_run,
            n_synth=args.n_synth,
            n_squad=args.n_squad,
            n_gretel=args.n_gretel,
            n_eval=args.n_eval,
            epochs=args.epochs,
            layers=args.layers or None,
            out_dir=args.out,
        )
    )


if __name__ == "__main__":
    main()
