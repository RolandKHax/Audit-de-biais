"""Télécharge et prépare le dataset COMPAS ProPublica."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import prepare_compas_data

COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


def download_compas(url: str, raw_path: Path, processed_path: Path) -> tuple[Path, Path]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"Téléchargement COMPAS: {url}")
        urllib.request.urlretrieve(url, raw_path)
    else:
        print(f"Dataset brut déjà présent: {raw_path}")

    df_raw = pd.read_csv(raw_path)
    df_processed = prepare_compas_data(df_raw)
    df_processed.to_csv(processed_path, index=False)
    print(f"Dataset traité: {processed_path} ({len(df_processed)} lignes)")
    return raw_path, processed_path


def main():
    parser = argparse.ArgumentParser(description="Télécharge le dataset COMPAS ProPublica")
    parser.add_argument("--url", default=COMPAS_URL)
    parser.add_argument("--raw-path", default="data/raw/compas-scores-two-years.csv")
    parser.add_argument("--processed-path", default="data/processed/compas_processed.csv")
    args = parser.parse_args()
    download_compas(args.url, Path(args.raw_path), Path(args.processed_path))


if __name__ == "__main__":
    main()
