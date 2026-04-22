import argparse

from embed import run_embed
from extract import run_extract


def main():
    parser = argparse.ArgumentParser(description="Unified entry for watermark embed/extract.")
    parser.add_argument(
        "--step",
        default="all",
        choices=["embed", "extract", "all"],
        help="Run only embed, only extract, or both (default: all).",
    )
    parser.add_argument(
        "--wm_mode",
        default="240",
        choices=["64", "240"],
        help="Watermark mode: 64 or 240 (default: 64).",
    )
    args = parser.parse_args()

    if args.step in {"embed", "all"}:
        run_embed(args.wm_mode)
    if args.step in {"extract", "all"}:
        source = "embedded" if args.step == "all" else "to_extract"
        run_extract(args.wm_mode, source=source)


if __name__ == "__main__":
    main()
