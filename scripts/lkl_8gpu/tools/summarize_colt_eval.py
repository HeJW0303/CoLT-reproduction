#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    files = sorted(set(args.root.rglob("*_acc.csv")) | set(args.root.rglob("*_score.csv")))
    if not files:
        print(f"No score CSV files found under {args.root}")
        return

    print("\nCoLT evaluation score files")
    print("=" * 80)
    for path in files:
        print(f"\n{path}")
        try:
            frame = pd.read_csv(path)
            print(frame.to_string(index=False))
        except Exception as error:
            print(f"Failed to parse: {error}")


if __name__ == "__main__":
    main()
