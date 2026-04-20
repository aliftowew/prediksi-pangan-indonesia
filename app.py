"""
app.py
======
Dasbor Prediksi Pangan Indonesia v2
-----------------------------------
- Data: 2022-2026+ (Excel multi-sheet, 38 provinsi, 25 komoditas)
- Model: Prophet, LightGBM, XGBoost, Ensemble (auto-pick via backtest)
- Upload data baru via UI → cache auto-update

Run: streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    PARQUET_PATH, data_summary, get_timeseries,
    list_komoditas, list_provinsi, load_cache, update_from_excel,
)
from models import (
    AVAILABLE_MODELS, EnsembleForecaster, LightGBMForecaster,
    ProphetForecaster, XGBoostForecaster, make_forecaster,
)
from backtest import backtest_all, best_model_name

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Prediksi Pangan Indonesia",
    page_icon="🌾",
    layout="wide",
)

# ==========================================================
# SIDEBAR: Data Management
# ==========================================================
with st.sidebar:
    st.header("📁 Manajemen Data")

    uploaded = st.file_uploader(
        "Upload Excel data pangan",
        type=["xlsx"],
        help=(
            "Format: tiap sheet = 1 tahun, kolom wajib "
            "`Provinsi`, `Tanggal`, lalu kolom-kolom komoditas. "
            "Data baru akan digabung dengan data lama (duplikat di-overwrite)."
        ),
    )

    if uploaded is not None:
        if st.button("🔄 Update Data", type="primary", use_container_width=True):
            with st.spinner("Memproses Excel..."):
                tmp = Path("_uploaded_tmp.xlsx")
                tmp.write_bytes(uploaded.getbuffer())
                df_new = update_from_excel(tmp)
                tmp.unlink(missing_ok=True)
                st.cache_data.clear()
            st.success(f"✅ Data diupdate: {len(df_new):,} baris")
            st.rerun()

    st.divider()

    @st.cache_data
    def _load():
        return load_cache()

    df_all = _load()

    if df_all is None or df_all.empty:
        st.warning("Belum ada data. Upload Excel untuk mulai.")
        st.stop()

    summary = data_summary(df_all)
    st.caption("📊 Status Data")
    st.metric("Total baris", f"{summary['jumlah_baris']:,}")
    col_a, col_b = st.columns(2)
    col_a.metric("Komoditas", summary["jumlah_komoditas"])
    col_b.metric("Provinsi", summary["jumlah_provinsi"])
    st.caption(
        f"📅 {summary['tanggal_mulai'].strftime('%d %b %Y')} → "
        f"{summary['tanggal_akhir'].strftime('%d %b %Y')}"
    )

# ==========================================================
# HEADER
# ==========================================================
st.title("🌾 Dasbor Prediksi Harga Pangan Indonesia")
st.caption(
    "Multi-model forecasting • Prophet + LightGBM + XGBoost + Ensemble "
    "• Data per provinsi"
)

# ==========================================================
# CONTROLS
# ==========================================================
komoditas_list = list_komoditas(df_all)
provinsi_list = ["Nasional"] + list_provinsi(df_all)

default_kom = (
    "Cabai Rawit Merah"
    if "Cabai Rawit Merah" in komoditas_list
    else komoditas_list[0]
)

c1, c2, c3, c4 = st.columns([2, 2, 2, 1.5])
with c1:
    kom = st.selectbox(
        "Komoditas", komoditas_list,
        index=komoditas_list.index(default_kom),
    )
with c2:
    prov = st.selectbox("Wilayah", provinsi_list, index=0)
with c3:
    model_choice = st.selectbox(
        "Model",
        ["Auto (best via backtest)"] + AVAILABLE_MODELS,
        help=(
            "Auto = jalankan backtest 30 hari terakhir, pilih model dengan "
            "MAPE terendah. Butuh waktu lebih lama (~1-2 menit)."
        ),
    )
with c4:
    horizon = st.number_input(
        "Horizon (hari)", min_value=7, max_value=365, value=60, step=1,
    )

run = st.button(
    "🚀 Jalankan Prediksi", type="primary", use_container_width=True,
)

# ==========================================================
# MAIN TABS
# ==========================================================
tab_pred, tab_backtest, tab_data = st.tabs([
    "📈 Prediksi", "🧪 Backtest", "📋 Data",
])

# ----------------------------------------------------------
# Helper: plotly chart
# ----------------------------------------------------------
def plot_forecast(
    history: pd.DataFrame, pred: pd.DataFrame, title: str,
) -> go.Figure:
    fig = go.Figure()
    # Actual (zoom ke 180 hari terakhir + prediksi)
    cutoff = history["ds"].max() - pd.Timedelta(days=180)
    hist_view = history[history["ds"] >= cutoff]

    fig.add_trace(go.Scatter(
        x=hist_view["ds"], y=hist_view["y"],
        mode="lines", name="Aktual",
        line=dict(color="#111827", width=2),
    ))
    # Prediksi interval
    fig.add_trace(go.Scatter(
        x=pd.concat([pred["ds"], pred["ds"][::-1]]),
        y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="Interval 80%",
    ))
    # Prediksi median
    fig.add_trace(go.Scatter(
        x=pred["ds"], y=pred["yhat"],
        mode="lines", name="Prediksi",
        line=dict(color="#2563eb", width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ----------------------------------------------------------
# TAB: PREDIKSI
# ----------------------------------------------------------
with tab_pred:
    if not run:
        st.info(
            "👆 Pilih komoditas + wilayah + model, lalu klik "
            "**Jalankan Prediksi**."
        )
    else:
        ts = get_timeseries(df_all, kom, provinsi=prov)
        if len(ts) < 90:
            st.error(f"Data tidak cukup untuk {kom} di {prov} (hanya {len(ts)} hari).")
            st.stop()

        # Pilih model
        if model_choice == "Auto (best via backtest)":
            with st.spinner("🔬 Menjalankan backtest untuk pilih model terbaik..."):
                bt = backtest_all(ts, horizon=30)
                chosen = best_model_name(bt)
            st.success(f"🏆 Model terbaik: **{chosen}** (MAPE {bt.iloc[0]['mape']:.2f}%)")
            model = make_forecaster(chosen)
        else:
            chosen = model_choice
            model = make_forecaster(chosen)

        with st.spinner(f"🤖 Training {chosen} & prediksi {horizon} hari..."):
            model.fit(ts)
            pred = model.predict(int(horizon))

        # Kesimpulan
        harga_terakhir = ts["y"].iloc[-1]
        harga_prediksi = pred["yhat"].iloc[-1]
        delta = harga_prediksi - harga_terakhir
        pct = (delta / harga_terakhir) * 100 if harga_terakhir else 0
        tren_emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➖")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Harga terakhir",
            f"Rp {int(harga_terakhir):,}",
            help=f"{ts['ds'].iloc[-1].strftime('%d %b %Y')}",
        )
        m2.metric(
            f"Prediksi +{horizon} hari {tren_emoji}",
            f"Rp {int(harga_prediksi):,}",
            f"{pct:+.1f}%",
        )
        m3.metric(
            "Batas bawah (10%)",
            f"Rp {int(pred['yhat_lower'].iloc[-1]):,}",
        )
        m4.metric(
            "Batas atas (90%)",
            f"Rp {int(pred['yhat_upper'].iloc[-1]):,}",
        )

        st.plotly_chart(
            plot_forecast(ts, pred, f"Prediksi {kom} — {prov}"),
            use_container_width=True,
        )

        # Tabel prediksi
        with st.expander("📋 Tabel prediksi harian"):
            tbl = pred.copy()
            tbl["ds"] = tbl["ds"].dt.strftime("%d %b %Y")
            for c in ["yhat", "yhat_lower", "yhat_upper"]:
                tbl[c] = tbl[c].round(0).astype(int)
            tbl.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            csv = tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV prediksi",
                csv,
                file_name=f"prediksi_{kom}_{prov}_{horizon}hari.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------
# TAB: BACKTEST
# ----------------------------------------------------------
with tab_backtest:
    st.markdown(
        "Backtest membandingkan model pada **30 hari terakhir** data "
        "(holdout). MAPE terendah = model paling akurat untuk komoditas ini."
    )
    bt_horizon = st.slider("Horizon backtest (hari)", 7, 60, 30)
    run_bt = st.button("🧪 Jalankan Backtest", use_container_width=True)

    if run_bt:
        ts = get_timeseries(df_all, kom, provinsi=prov)
        if len(ts) < bt_horizon + 60:
            st.error(f"Data tidak cukup untuk backtest.")
        else:
            with st.spinner("Training & evaluasi 4 model..."):
                bt = backtest_all(ts, horizon=bt_horizon)

            if "mape" in bt.columns:
                def _fmt(x):
                    return f"{x:.2f}%" if pd.notna(x) else "—"

                display = bt.copy()
                for c in ["mape", "coverage"]:
                    if c in display.columns:
                        display[c] = display[c].apply(_fmt)
                for c in ["rmse", "mae"]:
                    if c in display.columns:
                        display[c] = display[c].round(0).astype("Int64", errors="ignore")
                display.columns = [
                    "Model", "MAPE", "RMSE", "MAE", "Coverage 80%"
                ][:len(display.columns)]
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.success(f"🏆 **{bt.iloc[0]['model']}** juara dengan MAPE {bt.iloc[0]['mape']:.2f}%")

                st.caption(
                    "**MAPE** (Mean Absolute Percentage Error): rata-rata error %. "
                    "**RMSE**: error absolut (Rp). "
                    "**Coverage 80%**: % hari actual jatuh di dalam interval prediksi — "
                    "idealnya mendekati 80%."
                )
            else:
                st.error(bt.to_string())

# ----------------------------------------------------------
# TAB: DATA
# ----------------------------------------------------------
with tab_data:
    ts_view = get_timeseries(df_all, kom, provinsi=prov)

    st.subheader(f"{kom} — {prov}")
    s1, s2, s3 = st.columns(3)
    s1.metric("Jumlah hari", f"{len(ts_view):,}")
    s2.metric("Harga rata-rata", f"Rp {int(ts_view['y'].mean()):,}")
    s3.metric(
        "Volatilitas (CV)",
        f"{(ts_view['y'].std() / ts_view['y'].mean() * 100):.1f}%",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_view["ds"], y=ts_view["y"],
        mode="lines", line=dict(color="#111827", width=1.5),
    ))
    fig.update_layout(
        title=f"Riwayat Harga {kom} — {prov}",
        xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
        height=400, margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Data raw (semua provinsi untuk komoditas ini)"):
        raw = df_all[df_all["Komoditas"] == kom][
            ["Tanggal", "Provinsi", "Harga"]
        ].sort_values(["Tanggal", "Provinsi"], ascending=[False, True])
        st.dataframe(raw.head(200), use_container_width=True, hide_index=True)
        st.caption(f"Menampilkan 200 baris pertama dari {len(raw):,} baris.")

st.divider()
st.caption(
    "Dibuat dengan ❤️ oleh Alif Towew • #SemuaBisaDihitung • "
    "Data: Bapanas/Pangan Nasional"
)
