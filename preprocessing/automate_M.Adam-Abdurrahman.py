#!/usr/bin/env python3
"""
Automasi preprocessing dataset Titanic (Kriteria 1 - Skilled/3 pts)

Preprocessing yang dilakukan (sama seperti notebook):
1) Missing Values:
   - Age: fill median
   - Embarked: fill mode
   - Drop kolom Cabin (karena missing terlalu banyak)
2) Hapus duplikat
3) Encoding kategorikal:
   - Sex: LabelEncoder
   - Embarked: LabelEncoder
4) Scaling:
   - Age dan Fare: MinMaxScaler
5) Simpan hasil ke CSV

Cara menjalankan:
python automate_M.Adam-Abdurrahman.py \
  --input ../namadataset_raw/titanic_raw.csv \
  --output ../namadataset_preprocessing/titanic_preprocessing.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_raw(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {csv_path}")
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Missing values
    if "Age" in df.columns:
        df["Age"].fillna(df["Age"].median(), inplace=True)

    if "Embarked" in df.columns:
        # mode()[0] akan error kalau semua NaN, jadi guard
        if df["Embarked"].dropna().empty:
            df["Embarked"].fillna("Unknown", inplace=True)
        else:
            df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    # 2) Hapus duplikat
    dup_before = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    dup_after = int(df.duplicated().sum())

    # 3) Encoding kategorikal (LabelEncoder)
    # Catatan: LabelEncoder cocok untuk binary/ordinal sederhana. Untuk model tertentu,
    # OneHotEncoder bisa lebih baik, tapi kita samakan dengan notebook.
    if "Sex" in df.columns:
        le_sex = LabelEncoder()
        df["Sex"] = le_sex.fit_transform(df["Sex"].astype(str))

    if "Embarked" in df.columns:
        le_emb = LabelEncoder()
        df["Embarked"] = le_emb.fit_transform(df["Embarked"].astype(str))

    # 4) Scaling (MinMaxScaler) untuk Age & Fare
    cols_to_scale = [c for c in ["Age", "Fare"] if c in df.columns]
    if cols_to_scale:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Logging ringkas seperti di notebook
    print(f"Jumlah duplikat sebelum: {dup_before}")
    print(f"Jumlah duplikat sesudah : {dup_after}")
    print("\nMissing values setelah preprocessing:")
    print(df.isnull().sum())

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automasi preprocessing Titanic (Skilled).")
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
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    try:
        df_raw = load_raw(args.input)
        print("✅ Data raw berhasil dimuat")
        print(f"- Shape raw: {df_raw.shape}")

        df_processed = preprocess(df_raw)

        save_processed(df_processed, args.output)

        print("\n✅ Preprocessing selesai dan data disimpan")
        print(f"- Output: {os.path.abspath(args.output)}")
        print(f"- Shape processed: {df_processed.shape}")
        return 0
    except Exception as e:
        print(f"❌ Gagal menjalankan preprocessing: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
