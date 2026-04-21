"""
visuals.py
==========
Komponen visual reusable untuk dashboard.

Berisi:
- sparklines_grid: mini-chart semua komoditas dalam 1 layar
- small_multiples: grid 25 komoditas dengan prediksi overlay
- peta_indonesia: choropleth map per provinsi
- seasonal_decompose: pisahkan trend/seasonal/residual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Brand colors
COLOR_BRAND = "#2563eb"
COLOR_DARK = "#111827"
COLOR_MUTED = "#9ca3af"
COLOR_SUCCESS = "#059669"
COLOR_DANGER = "#dc2626"


# ==========================================================
# 1. SPARKLINES GRID (overview cepat semua komoditas)
# ==========================================================
def sparklines_grid(
    df: pd.DataFrame,
    komoditas_list: list[str],
    provinsi: str = "Nasional",
    n_days: int = 90,
) -> list[dict]:
    """
    Return list of dict per komoditas dengan:
    - nama, harga_terakhir, perubahan_pct, sparkline_data
    Dipakai di UI untuk render kartu-kartu ringkas.
    """
    from data_loader import get_timeseries

    cards = []
    for kom in komoditas_list:
        ts = get_timeseries(df, kom, provinsi=provinsi)
        if len(ts) < n_days:
            continue
        recent = ts.tail(n_days)
        first = float(recent["y"].iloc[0])
        last = float(recent["y"].iloc[-1])
        pct = ((last - first) / first * 100) if first else 0
        cards.append({
            "komoditas": kom,
            "harga_terakhir": last,
            "perubahan_pct": pct,
            "sparkline_x": recent["ds"].tolist(),
            "sparkline_y": recent["y"].tolist(),
            "volatilitas": float(recent["y"].std() / recent["y"].mean() * 100) if recent["y"].mean() else 0,
        })
    return cards


def make_sparkline_fig(card: dict, height: int = 40) -> go.Figure:
    """Mini sparkline chart (Plotly) untuk 1 komoditas."""
    color = COLOR_SUCCESS if card["perubahan_pct"] < 0 else COLOR_DANGER
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=card["sparkline_x"], y=card["sparkline_y"],
        mode="lines",
        line=dict(color=color, width=1.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        showlegend=False, height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==========================================================
# 2. SMALL MULTIPLES (grid 25 komoditas)
# ==========================================================
def small_multiples_grid(
    df: pd.DataFrame,
    komoditas_list: list[str],
    provinsi: str = "Nasional",
    forecasts_cache: pd.DataFrame | None = None,
    n_days_hist: int = 180,
    ncols: int = 5,
) -> go.Figure:
    """
    Grid small multiples ala FT / Tufte.
    Tiap cell = 1 komoditas, dengan history + prediksi (kalau ada di cache).
    """
    from data_loader import get_timeseries

    n = len(komoditas_list)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=komoditas_list,
        vertical_spacing=0.08, horizontal_spacing=0.04,
    )

    for i, kom in enumerate(komoditas_list):
        row = i // ncols + 1
        col = i % ncols + 1
        ts = get_timeseries(df, kom, provinsi=provinsi)
        if ts.empty:
            continue
        hist = ts.tail(n_days_hist)

        # History
        fig.add_trace(
            go.Scatter(
                x=hist["ds"], y=hist["y"],
                mode="lines",
                line=dict(color=COLOR_DARK, width=1.2),
                showlegend=False, hovertemplate="Rp %{y:,.0f}<extra></extra>",
            ),
            row=row, col=col,
        )
        # Prediksi (dari cache)
        if forecasts_cache is not None:
            pred = forecasts_cache[
                (forecasts_cache["Komoditas"] == kom)
                & (forecasts_cache["Wilayah"] == provinsi)
            ]
            if not pred.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pred["ds"], y=pred["yhat"],
                        mode="lines",
                        line=dict(color=COLOR_BRAND, width=1.2, dash="dash"),
                        showlegend=False, hovertemplate="Pred: Rp %{y:,.0f}<extra></extra>",
                    ),
                    row=row, col=col,
                )

    fig.update_layout(
        height=140 * nrows,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", showticklabels=False)
    # Kecilkan font subplot title
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=10, color=COLOR_DARK)
    return fig


# ==========================================================
# 3. SEASONAL DECOMPOSE
# ==========================================================
def seasonal_decompose_chart(
    ts: pd.DataFrame,
    komoditas: str,
    period: int = 365,
) -> go.Figure:
    """
    Pisahkan series ke: Observed, Trend, Seasonal, Residual.
    Pakai classical decomposition (additive) via moving average.
    """
    y = ts["y"].values.astype(float)
    n = len(y)

    if n < period * 2:
        # Data < 2 tahun, pakai period yang lebih pendek
        period = min(period, n // 2)

    # Trend: centered moving average
    trend = pd.Series(y).rolling(window=period, center=True).mean().values

    # Detrended
    detrended = y - trend

    # Seasonal: rata-rata detrended untuk tiap index modulo period
    seasonal = np.zeros(n)
    for i in range(period):
        idx = np.arange(i, n, period)
        vals = detrended[idx]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            seasonal[idx] = np.mean(vals)
    # Normalisasi seasonal agar sum=0
    seasonal = seasonal - np.nanmean(seasonal)

    residual = y - trend - seasonal

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Observasi (Aktual)", "Trend Jangka Panjang",
                        "Seasonal (Pola Tahunan)", "Residual (Sisa)"),
    )
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=y, mode="lines", line=dict(color=COLOR_DARK, width=1.5),
        name="Observed", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=trend, mode="lines", line=dict(color=COLOR_BRAND, width=2),
        name="Trend", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=seasonal, mode="lines", line=dict(color="#059669", width=1.2),
        name="Seasonal", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=residual, mode="lines", line=dict(color=COLOR_MUTED, width=0.8),
        name="Residual", showlegend=False,
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", row=4, col=1)

    fig.update_layout(
        title=f"Seasonal Decomposition — {komoditas}",
        height=700, margin=dict(l=50, r=20, t=80, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


# ==========================================================
# 4. PETA INDONESIA (choropleth per provinsi)
# ==========================================================
# Mapping nama provinsi ke koordinat lat/lon untuk bubble map
# (GeoJSON topology Indonesia besar, kita pakai scatter geo dulu = lebih ringan)
PROVINSI_COORDS = {
    "Aceh": (4.695, 96.749),
    "Bali": (-8.340, 115.092),
    "Banten": (-6.405, 106.064),
    "Bengkulu": (-3.792, 102.261),
    "D.I Yogyakarta": (-7.875, 110.426),
    "DKI Jakarta": (-6.200, 106.845),
    "Gorontalo": (0.699, 122.446),
    "Jambi": (-1.610, 103.613),
    "Jawa Barat": (-6.915, 107.610),
    "Jawa Tengah": (-7.150, 110.140),
    "Jawa Timur": (-7.536, 112.238),
    "Kalimantan Barat": (-0.278, 111.475),
    "Kalimantan Selatan": (-3.092, 115.284),
    "Kalimantan Tengah": (-1.682, 113.382),
    "Kalimantan Timur": (1.658, 116.419),
    "Kalimantan Utara": (3.735, 117.139),
    "Kepulauan Bangka Belitung": (-2.741, 106.441),
    "Kepulauan Riau": (3.945, 108.143),
    "Lampung": (-4.559, 105.408),
    "Maluku": (-3.238, 130.145),
    "Maluku Utara": (1.571, 127.808),
    "Nusa Tenggara Barat": (-8.652, 117.361),
    "Nusa Tenggara Timur": (-8.657, 121.079),
    "Papua": (-4.270, 138.080),
    "Papua Barat": (-1.336, 133.174),
    "Papua Barat Daya": (-0.864, 132.298),
    "Papua Pegunungan": (-4.086, 138.932),
    "Papua Selatan": (-7.366, 140.708),
    "Papua Tengah": (-3.536, 136.324),
    "Riau": (0.293, 101.708),
    "Sulawesi Barat": (-2.844, 119.232),
    "Sulawesi Selatan": (-3.669, 119.974),
    "Sulawesi Tengah": (-1.430, 121.446),
    "Sulawesi Tenggara": (-4.147, 122.175),
    "Sulawesi Utara": (0.624, 123.975),
    "Sumatera Barat": (-0.740, 100.800),
    "Sumatera Selatan": (-3.319, 103.914),
    "Sumatera Utara": (2.115, 99.545),
}


def peta_harga_provinsi(
    df: pd.DataFrame,
    komoditas: str,
    tanggal: pd.Timestamp | None = None,
) -> go.Figure:
    """
    Scatter geo map Indonesia — bubble per provinsi, size + color = harga.
    """
    sub = df[df["Komoditas"] == komoditas].copy()
    if tanggal is None:
        tanggal = sub["Tanggal"].max()

    # Ambil harga terakhir <= tanggal per provinsi
    snap = (
        sub[sub["Tanggal"] <= tanggal]
        .sort_values("Tanggal")
        .groupby("Provinsi", as_index=False)
        .last()
    )
    snap = snap[snap["Provinsi"].isin(PROVINSI_COORDS)]
    snap["lat"] = snap["Provinsi"].map(lambda p: PROVINSI_COORDS[p][0])
    snap["lon"] = snap["Provinsi"].map(lambda p: PROVINSI_COORDS[p][1])

    # Untuk size marker: normalize log scale (biar selisih jadi masuk akal)
    size_val = np.sqrt(snap["Harga"].values)
    size_norm = 10 + (size_val - size_val.min()) / (size_val.max() - size_val.min() + 1e-9) * 30

    fig = go.Figure(go.Scattergeo(
        lon=snap["lon"], lat=snap["lat"],
        text=snap.apply(
            lambda r: f"<b>{r['Provinsi']}</b><br>Rp {int(r['Harga']):,}",
            axis=1,
        ),
        mode="markers",
        marker=dict(
            size=size_norm,
            color=snap["Harga"],
            colorscale="RdYlGn_r",  # Hijau = murah, merah = mahal
            colorbar=dict(title="Harga (Rp)", thickness=12, len=0.7),
            line=dict(color="white", width=1),
            sizemode="diameter",
        ),
        hoverinfo="text",
    ))
    fig.update_geos(
        scope="asia",
        center=dict(lat=-2.5, lon=118),
        projection_scale=5,
        showland=True, landcolor="#f8fafc",
        showocean=True, oceancolor="#e0f2fe",
        showcountries=True, countrycolor="#cbd5e1",
        showsubunits=True, subunitcolor="#cbd5e1",
        lataxis_range=[-12, 7], lonaxis_range=[94, 142],
    )
    fig.update_layout(
        title=f"Peta Harga {komoditas} — Per Provinsi "
              f"({tanggal.strftime('%d %b %Y')})",
        height=500, margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig
