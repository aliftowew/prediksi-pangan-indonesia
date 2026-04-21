"""
data_loader.py
==============
Pipeline data pangan: Excel (multi-sheet per tahun) → DataFrame long format → parquet cache.

Mendukung:
- Load dari Excel multi-sheet (tahun 2022-2026+)
- Cache parquet untuk akses cepat
- Update data: merge dataset lama + baru tanpa duplikasi
- Query per komoditas + per provinsi + nasional

Format kanonik (long):
    Tanggal | Provinsi | Komoditas | Harga
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PARQUET_PATH = DATA_DIR / "pangan_long.parquet"


def read_excel_multisheet(path: str | Path) -> pd.DataFrame:
    """Baca file Excel multi-sheet (tiap sheet = 1 tahun) → DataFrame long format."""
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=0)
        df["Tanggal"] = pd.to_datetime(df["Tanggal"]).dt.normalize()
        # Lebur wide → long (tiap kolom komoditas jadi baris)
        id_cols = ["Provinsi", "Tanggal"]
        value_cols = [c for c in df.columns if c not in id_cols]
        long = df.melt(
            id_vars=id_cols, value_vars=value_cols,
            var_name="Komoditas", value_name="Harga",
        )
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    # Bersihkan: harga 0 / NaN / negatif anggap missing
    out["Harga"] = pd.to_numeric(out["Harga"], errors="coerce")
    out.loc[out["Harga"] <= 0, "Harga"] = pd.NA
    out = out.dropna(subset=["Harga"]).reset_index(drop=True)
    return out


def save_cache(df: pd.DataFrame) -> None:
    """Simpan ke parquet untuk load cepat."""
    df.to_parquet(PARQUET_PATH, index=False)


def load_cache() -> pd.DataFrame | None:
    """Load dari parquet jika ada, else None."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    return None


def merge_new_data(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan data lama + baru. Data baru override lama jika ada konflik
    di (Tanggal, Provinsi, Komoditas).
    """
    combined = pd.concat([existing, new], ignore_index=True)
    # Drop duplikasi: keep last (artinya data baru menang)
    combined = combined.drop_duplicates(
        subset=["Tanggal", "Provinsi", "Komoditas"], keep="last"
    )
    combined = combined.sort_values(["Komoditas", "Provinsi", "Tanggal"])
    return combined.reset_index(drop=True)


def update_from_excel(path: str | Path) -> pd.DataFrame:
    """
    Update data cache dengan file Excel baru. 
    Jika cache belum ada → jadikan cache awal.
    Return dataset gabungan.
    """
    new_data = read_excel_multisheet(path)
    existing = load_cache()
    if existing is None:
        final = new_data
    else:
        final = merge_new_data(existing, new_data)
    save_cache(final)
    return final


# =========================================================
# Query helpers
# =========================================================

def list_komoditas(df: pd.DataFrame) -> list[str]:
    return sorted(df["Komoditas"].unique().tolist())


def list_provinsi(df: pd.DataFrame) -> list[str]:
    return sorted(df["Provinsi"].unique().tolist())


def get_timeseries(
    df: pd.DataFrame,
    komoditas: str,
    provinsi: str | None = None,
) -> pd.DataFrame:
    """
    Ambil time series untuk 1 komoditas.
    - provinsi=None (atau 'Nasional') → rata-rata lintas provinsi
    - provinsi=<nama> → filter provinsi tersebut

    Return: DataFrame kolom [ds, y] (format siap Prophet/LightGBM).
    """
    sub = df[df["Komoditas"] == komoditas].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ds", "y"])

    if provinsi is None or provinsi == "Nasional":
        # Rata-rata lintas provinsi per tanggal
        series = sub.groupby("Tanggal", as_index=False)["Harga"].mean()
    else:
        series = sub[sub["Provinsi"] == provinsi][["Tanggal", "Harga"]]

    series = series.rename(columns={"Tanggal": "ds", "Harga": "y"})
    series = series.sort_values("ds").reset_index(drop=True)
    # Isi tanggal yang hilang dengan interpolasi linear (libur, hari kosong)
    series = series.set_index("ds").asfreq("D").interpolate("linear").reset_index()
    return series


def data_summary(df: pd.DataFrame) -> dict:
    """Ringkasan dataset untuk ditampilkan di UI."""
    return {
        "jumlah_baris": len(df),
        "jumlah_komoditas": df["Komoditas"].nunique(),
        "jumlah_provinsi": df["Provinsi"].nunique(),
        "tanggal_mulai": df["Tanggal"].min(),
        "tanggal_akhir": df["Tanggal"].max(),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = update_from_excel(sys.argv[1])
        print("Data updated. Summary:")
        for k, v in data_summary(df).items():
            print(f"  {k}: {v}")
    else:
        df = load_cache()
        if df is None:
            print("Cache kosong. Jalankan: python data_loader.py <path_excel>")
        else:
            print("Cache summary:")
            for k, v in data_summary(df).items():
                print(f"  {k}: {v}")
