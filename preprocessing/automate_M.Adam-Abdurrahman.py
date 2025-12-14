#!/usr/bin/env python3
"""
Automasi preprocessing dataset Titanic untuk Kriteria 1 (Skilled/3 pts).

Input  : ../namadataset_raw/titanic_raw.csv
Output : ../namadataset_preprocessing/titanic_preprocessing.csv

Menjalankan:
python3 automate_M.Adam-Abdurrahman.py \
  --input ../namadataset_raw/titanic_raw.csv \
  --output ../namadataset_preprocessing/titanic_preprocessing.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd


DROP_COLS_DEFAULT = ["Name", "Ticket", "Cabin"]  # bisa kamu sesuaikan


def load_raw(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame, drop_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Preprocessing yang aman dan umum untuk Titanic:
    - Drop kolom tidak relevan (Name, Ticket, Cabin) jika ada
    - Handle missing value: Age (median), Embarked (mode)
    - One-hot encoding untuk: Sex, Embarked
    - Pastikan target (Survived) tetap ada jika memang ada di input
    """
    df = df.copy()

    # --- Drop kolom jika ada ---
    drop_cols = drop_cols or []
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # --- Missing values ---
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Embarked" in df.columns:
        # mode bisa kosong kalau kolomnya semua NaN, jadi guard
        if df["Embarked"].dropna().empty:
            df["Embarked"] = df["Embarked"].fillna("Unknown")
        else:
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # --- One-hot encode untuk kolom kategori (aman jika kolomnya tidak ada) ---
    cat_cols = [c for c in ["Sex", "Embarked"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # --- Bersihkan sisa missing value numerik (opsional aman) ---
    # (Jika ada kolom numerik lain yang NaN, isi median kolom)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automasi preprocessing dataset Titanic (Kriteria 1 - Skilled)."
    )
    p.add_argument(
        "--input",
        default="../namadataset_raw/titanic_raw.csv",
        help="Path CSV input (raw). Default: ../namadataset_raw/titanic_raw.csv",
    )
    p.add_argument(
        "--output",
        default="../namadataset_preprocessing/titanic_preprocessing.csv",
        help="Path CSV output (preprocessed). Default: ../namadataset_preprocessing/titanic_preprocessing.csv",
    )
    p.add_argument(
        "--drop-cols",
        default=",".join(DROP_COLS_DEFAULT),
        help=f"Daftar kolom yang ingin di-drop (pisahkan koma). Default: {DROP_COLS_DEFAULT}",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    try:
        df_raw = load_raw(args.input)
        df_processed = preprocess(df_raw, drop_cols=drop_cols)
        save_processed(df_processed, args.output)

        print("✅ Preprocessing berhasil!")
        print(f"- Input  : {os.path.abspath(args.input)}")
        print(f"- Output : {os.path.abspath(args.output)}")
        print(f"- Shape  : {df_processed.shape}")
        return 0
    except Exception as e:
        print(f"❌ Gagal preprocessing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
