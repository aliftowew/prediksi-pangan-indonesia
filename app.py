"""
app.py
======
Dasbor Prediksi Pangan Indonesia v2
-----------------------------------
Optimasi kecepatan:
- Pre-computed forecasts (load instant dari disk, bukan training live)
- On-demand training hanya untuk provinsi yang belum di-cache
- Cache session untuk ganti parameter tanpa lag

Fitur visual:
- Sparklines grid (overview cepat 25 komoditas)
- Small multiples (grid prediksi semua komoditas)
- Peta Indonesia (heatmap harga per provinsi)
- Seasonal decompose (trend + seasonal + residual)

Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    data_summary, get_timeseries,
    list_komoditas, list_provinsi, load_cache, update_from_excel,
)
from models import AVAILABLE_MODELS, make_forecaster
from backtest import backtest_all, best_model_name
from precompute import (
    load_forecasts, is_cache_valid, get_cached_forecast,
    add_forecast_to_cache, compute_all_forecasts, save_forecasts,
    DEFAULT_MODEL,
)
from visuals import (
    sparklines_grid, make_sparkline_fig,
    small_multiples_grid, seasonal_decompose_chart, peta_harga_provinsi,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Prediksi Pangan Indonesia",
    page_icon="🌾",
    layout="wide",
)

# ==========================================================
# STYLING
# ==========================================================
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
.stMetric {
    background: #f9fafb; padding: 0.75rem; border-radius: 8px;
    border: 1px solid #e5e7eb;
}
h2 { margin-top: 2.5rem !important; padding-top: 1rem; border-top: 1px solid #f1f5f9; }
h2:first-of-type { border-top: none; padding-top: 0; }
.footer {
    text-align: center; padding: 2rem 0 1rem; margin-top: 3rem;
    border-top: 2px solid #111827; color: #6b7280; font-size: 0.9rem;
}
.footer b { color: #111827; font-size: 1.05rem; }
.footer .tag { color: #2563eb; font-weight: 600; }
.spark-card {
    background: white; padding: 0.75rem; border-radius: 8px;
    border: 1px solid #e5e7eb;
}
.spark-card .nama { font-size: 0.85rem; font-weight: 600; color: #111827;
                   margin-bottom: 0.25rem; line-height: 1.2;}
.spark-card .harga { font-size: 1rem; font-weight: 700; color: #111827; }
.spark-card .delta-up { color: #dc2626; font-size: 0.8rem; }
.spark-card .delta-down { color: #059669; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR: Data Management + Upload Instructions
# ==========================================================
with st.sidebar:
    st.markdown("## 📁 Manajemen Data")

    df_preview = load_cache()
    default_expanded = df_preview is None or df_preview.empty

    with st.expander("📖 Cara Upload Data Baru", expanded=default_expanded):
        st.markdown("""
**Langkah-langkah:**
1. Siapkan file Excel (`.xlsx`)
2. Klik **Browse files** di form upload
3. Pilih file dari komputer
4. Klik **🔄 Update Data**

**Format Excel:**
- Tiap **sheet** = 1 tahun (`2022`, `2023`, dst.)
- Kolom wajib: `Provinsi` dan `Tanggal`
- Kolom lainnya = komoditas (harga Rp)

**Contoh:**

| Provinsi | Tanggal | Beras Premium | ... |
|----------|---------|---------------|-----|
| Aceh | 2022-01-01 | 12112 | ... |
| Aceh | 2022-01-02 | 12056 | ... |

**Catatan:**
- Data baru akan **digabung** dengan data lama
- Duplikasi (Tanggal + Provinsi + Komoditas) akan **ditimpa** data baru
- Harga 0/kosong diabaikan
- Sumber: [panelharga.badanpangan.go.id](https://panelharga.badanpangan.go.id)
""")

    uploaded = st.file_uploader(
        "Upload Excel data pangan",
        type=["xlsx"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        st.caption(f"📄 **{uploaded.name}** ({uploaded.size / 1024:.0f} KB)")
        if st.button("🔄 Update Data", type="primary", use_container_width=True):
            with st.spinner("Memproses Excel..."):
                tmp = Path("_uploaded_tmp.xlsx")
                tmp.write_bytes(uploaded.getbuffer())
                df_new = update_from_excel(tmp)
                tmp.unlink(missing_ok=True)
                # Invalidate forecast cache karena data berubah
                fc_path = Path("data/forecasts.parquet")
                meta_path = Path("data/forecasts_meta.json")
                fc_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                st.cache_data.clear()
                st.cache_resource.clear()
            st.success(
                f"✅ Data diperbarui: {len(df_new):,} baris. "
                "Pre-compute prediksi akan jalan otomatis."
            )
            st.rerun()

    st.divider()

    @st.cache_data
    def _load_data():
        return load_cache()

    df_all = _load_data()

    if df_all is None or df_all.empty:
        st.warning("⚠️ Belum ada data. Silakan upload Excel.")
        st.stop()

    summary = data_summary(df_all)
    st.markdown("### 📊 Status Data")
    st.metric("Total baris", f"{summary['jumlah_baris']:,}")
    col_a, col_b = st.columns(2)
    col_a.metric("Komoditas", summary["jumlah_komoditas"])
    col_b.metric("Provinsi", summary["jumlah_provinsi"])
    st.caption(
        f"📅 {summary['tanggal_mulai'].strftime('%d %b %Y')} → "
        f"{summary['tanggal_akhir'].strftime('%d %b %Y')}"
    )

# ==========================================================
# PRE-COMPUTE CHECK (pasang otomatis kalau belum ada)
# ==========================================================
@st.cache_data(show_spinner=False)
def _load_forecasts_cached():
    return load_forecasts()


fc_all, fc_meta = _load_forecasts_cached()
needs_compute = fc_all is None or not is_cache_valid(df_all)

if needs_compute:
    st.markdown("## ⚡ Setup Awal: Pre-compute Prediksi")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if fc_all is None:
            st.info(
                "📦 **Cache prediksi belum dibuat.** "
                "Dashboard akan melatih model LightGBM untuk 25 komoditas nasional sekaligus, "
                "lalu simpan hasilnya ke disk. Ini hanya sekali saja, setelah itu akses instan."
            )
        else:
            st.warning(
                "🔄 **Data baru terdeteksi — cache perlu diperbarui.** "
                "Klik tombol di sebelah untuk re-compute prediksi dengan data terbaru."
            )
        st.caption(
            "⏱️ Perkiraan waktu: ~3 menit. Kamu bisa melihat progress di bawah. "
            "Setelah selesai, halaman akan reload otomatis."
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        start = st.button("▶️ Mulai Pre-compute", type="primary", use_container_width=True)

    if start:
        progress = st.progress(0, text="Mempersiapkan model...")
        status = st.empty()

        def cb(done, total, msg):
            progress.progress(done / total, text=f"[{done}/{total}] Memproses: {msg}")

        try:
            fc_all = compute_all_forecasts(df_all, progress_callback=cb)
            save_forecasts(fc_all, df_all, DEFAULT_MODEL)
            st.cache_data.clear()
            progress.empty()
            st.success("✅ Pre-compute selesai! Halaman akan refresh dalam 2 detik...")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            progress.empty()
            st.error(f"❌ Error saat pre-compute: {e}")
            st.caption("Coba refresh halaman atau cek data input.")
    st.stop()

# ==========================================================
# HEADER
# ==========================================================
st.title("🌾 Dasbor Prediksi Harga Pangan Indonesia")
st.markdown(
    "Multi-model forecasting — **Prophet + LightGBM + XGBoost + Ensemble**. "
    "Prediksi nasional sudah dihitung sebelumnya untuk akses instan."
)
st.caption(
    f"🤖 Model dasar: **{fc_meta['model']}** • "
    f"📅 Update terakhir: {pd.Timestamp(fc_meta['generated_at']).strftime('%d %b %Y %H:%M')} • "
    f"🗃️ {fc_meta['n_rows']:,} prediksi di cache"
)

# ==========================================================
# CONTROLS
# ==========================================================
komoditas_list = list_komoditas(df_all)
provinsi_list = ["Nasional"] + list_provinsi(df_all)
default_kom = (
    "Cabai Rawit Merah" if "Cabai Rawit Merah" in komoditas_list
    else komoditas_list[0]
)

c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
with c1:
    kom = st.selectbox(
        "🌶️ Komoditas", komoditas_list,
        index=komoditas_list.index(default_kom),
    )
with c2:
    prov = st.selectbox("📍 Wilayah", provinsi_list, index=0)
with c3:
    model_choice = st.selectbox(
        "🤖 Model",
        [f"Dari cache ({fc_meta['model']})"] + AVAILABLE_MODELS + ["Auto (best via backtest)"],
        index=0,
        help=(
            "Cache = instant (pakai prediksi yang sudah dihitung). "
            "Pilih model spesifik = training live (butuh waktu)."
        ),
    )
with c4:
    horizon = st.number_input(
        "📅 Hari", min_value=7, max_value=180, value=60, step=1,
    )


# ==========================================================
# HELPERS
# ==========================================================
@st.cache_resource(show_spinner=False)
def get_trained_model(model_name: str, komoditas: str, provinsi: str):
    ts = get_timeseries(df_all, komoditas, provinsi=provinsi)
    if len(ts) < 90:
        return None, ts
    model = make_forecaster(model_name)
    model.fit(ts)
    return model, ts


@st.cache_data(show_spinner=False)
def run_backtest_cached(komoditas: str, provinsi: str, horizon: int):
    ts = get_timeseries(df_all, komoditas, provinsi=provinsi)
    return backtest_all(ts, horizon=horizon)


# ==========================================================
# SECTION 1: PREDIKSI UTAMA
# ==========================================================
st.markdown("## 📈 Hasil Prediksi")

ts_hist = get_timeseries(df_all, kom, provinsi=prov)

use_cache = model_choice.startswith("Dari cache")
pred = None
chosen_name = fc_meta["model"]

if use_cache:
    # Coba ambil dari cache
    pred_cached = get_cached_forecast(fc_all, kom, prov, int(horizon))

    if pred_cached.empty:
        # Provinsi ini belum di-cache, train on-demand
        with st.spinner(
            f"🤖 {prov} belum di cache — training {chosen_name} "
            f"({kom})... (~7 detik, lalu di-cache)"
        ):
            model, ts = get_trained_model(chosen_name, kom, prov)
            if model is None:
                st.error(f"Data tidak cukup untuk {kom} di {prov}.")
                st.stop()
            pred_raw = model.predict(180)  # Full cache
            fc_all = add_forecast_to_cache(
                pred_raw, kom, prov,
                harga_terakhir=float(ts["y"].iloc[-1]),
                tanggal_terakhir=ts["ds"].iloc[-1],
            )
            st.cache_data.clear()  # invalidate streamlit cache
            pred = pred_raw.head(int(horizon)).copy()
        st.caption(f"✅ {prov} di-cache — akses berikutnya instant.")
    else:
        pred = pred_cached
elif model_choice == "Auto (best via backtest)":
    with st.spinner("🔬 Backtest 4 model untuk pilih terbaik... (1-2 menit)"):
        bt = run_backtest_cached(kom, prov, 30)
        chosen_name = best_model_name(bt)
    st.info(
        f"🏆 Model terpilih: **{chosen_name}** "
        f"(MAPE {bt.iloc[0]['mape']:.2f}%)"
    )
    with st.spinner(f"🤖 Training {chosen_name}..."):
        model, _ = get_trained_model(chosen_name, kom, prov)
    pred = model.predict(int(horizon))
else:
    chosen_name = model_choice
    with st.spinner(f"🤖 Training {chosen_name}..."):
        model, _ = get_trained_model(chosen_name, kom, prov)
    if model is None:
        st.error(f"Data tidak cukup untuk {kom} di {prov}.")
        st.stop()
    pred = model.predict(int(horizon))

# --- Metrik ---
if pred is not None and not pred.empty:
    harga_terakhir = float(ts_hist["y"].iloc[-1])
    harga_prediksi = float(pred["yhat"].iloc[-1])
    delta = harga_prediksi - harga_terakhir
    pct = (delta / harga_terakhir) * 100 if harga_terakhir else 0
    tren_emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➖")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Harga terakhir", f"Rp {int(harga_terakhir):,}",
        help=f"Per {ts_hist['ds'].iloc[-1].strftime('%d %b %Y')}",
    )
    m2.metric(
        f"Prediksi +{horizon} hari {tren_emoji}",
        f"Rp {int(harga_prediksi):,}", f"{pct:+.1f}%",
    )
    m3.metric("Batas bawah (P10)", f"Rp {int(pred['yhat_lower'].iloc[-1]):,}")
    m4.metric("Batas atas (P90)", f"Rp {int(pred['yhat_upper'].iloc[-1]):,}")

    # --- Chart ---
    fig = go.Figure()
    cutoff = ts_hist["ds"].max() - pd.Timedelta(days=180)
    hist_view = ts_hist[ts_hist["ds"] >= cutoff]

    fig.add_trace(go.Scatter(
        x=hist_view["ds"], y=hist_view["y"],
        mode="lines", name="Aktual (180 hari terakhir)",
        line=dict(color="#111827", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([pred["ds"], pred["ds"][::-1]]),
        y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="Interval 80%",
    ))
    fig.add_trace(go.Scatter(
        x=pred["ds"], y=pred["yhat"],
        mode="lines", name=f"Prediksi ({chosen_name})",
        line=dict(color="#2563eb", width=2.5, dash="dash"),
    ))
    fig.update_layout(
        title=f"Prediksi {kom} — {prov}",
        xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460, margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(gridcolor="#f1f5f9")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabel prediksi harian + download CSV"):
        tbl = pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        tbl["ds"] = pd.to_datetime(tbl["ds"]).dt.strftime("%d %b %Y")
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            tbl[c] = tbl[c].round(0).astype(int)
        tbl.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", csv,
            file_name=f"prediksi_{kom}_{prov}_{horizon}hari.csv",
            mime="text/csv",
        )

# ==========================================================
# SECTION 2: SPARKLINES - OVERVIEW 25 KOMODITAS
# ==========================================================
st.markdown("## ✨ Overview 25 Komoditas — Sparklines")
st.caption(
    "Mini-chart 90 hari terakhir untuk semua komoditas. "
    "Hijau = turun, merah = naik. Klik nama untuk detail."
)

@st.cache_data(show_spinner=False)
def _spark_cards(provinsi: str):
    return sparklines_grid(df_all, komoditas_list, provinsi, n_days=90)


cards = _spark_cards(prov)

# Grid 5 kolom
for i in range(0, len(cards), 5):
    row_cards = cards[i:i+5]
    cols = st.columns(5)
    for col, card in zip(cols, row_cards):
        with col:
            color_class = "delta-down" if card["perubahan_pct"] < 0 else "delta-up"
            arrow = "▼" if card["perubahan_pct"] < 0 else "▲"
            st.markdown(
                f"""<div class='spark-card'>
                <div class='nama'>{card['komoditas']}</div>
                <div class='harga'>Rp {int(card['harga_terakhir']):,}</div>
                <div class='{color_class}'>{arrow} {abs(card['perubahan_pct']):.1f}% 90d</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_sparkline_fig(card),
                use_container_width=True,
                config={"displayModeBar": False},
            )

# ==========================================================
# SECTION 3: SMALL MULTIPLES — GRID 25 KOMODITAS + PREDIKSI
# ==========================================================
st.markdown("## 📊 Small Multiples — 25 Komoditas + Prediksi")
st.caption(
    "Garis hitam = aktual 180 hari terakhir. "
    "Garis biru putus-putus = prediksi nasional dari cache. "
    "Scan cepat pola makro."
)

@st.cache_data(show_spinner=False)
def _small_mult_fig(provinsi: str):
    fc, _ = load_forecasts()
    return small_multiples_grid(df_all, komoditas_list, provinsi, forecasts_cache=fc)


with st.spinner("Menyiapkan 25 chart..."):
    fig_sm = _small_mult_fig(prov)
st.plotly_chart(fig_sm, use_container_width=True)

# ==========================================================
# SECTION 4: PETA INDONESIA
# ==========================================================
st.markdown("## 🗺️ Peta Harga per Provinsi")
st.caption(
    f"Snapshot harga {kom} per provinsi. "
    "Bubble besar & merah = harga tinggi, bubble kecil & hijau = harga murah."
)

@st.cache_data(show_spinner=False)
def _peta_fig(komoditas: str):
    return peta_harga_provinsi(df_all, komoditas)


fig_peta = _peta_fig(kom)
st.plotly_chart(fig_peta, use_container_width=True)

# ==========================================================
# SECTION 5: SEASONAL DECOMPOSE
# ==========================================================
st.markdown("## 📐 Seasonal Decomposition")
st.caption(
    "Pisahkan harga menjadi komponen: **Trend** (arah jangka panjang), "
    "**Seasonal** (pola tahunan, mis. efek Lebaran), dan **Residual** (sisa yang tidak dijelaskan). "
    "Berguna untuk memahami apa yang menggerakkan harga."
)

@st.cache_data(show_spinner=False)
def _decompose_fig(komoditas: str, provinsi: str):
    ts = get_timeseries(df_all, komoditas, provinsi=provinsi)
    return seasonal_decompose_chart(ts, f"{komoditas} — {provinsi}")


fig_dec = _decompose_fig(kom, prov)
st.plotly_chart(fig_dec, use_container_width=True)

# ==========================================================
# SECTION 6: BACKTEST
# ==========================================================
st.markdown("## 🧪 Backtest — Perbandingan Akurasi Model")
st.caption(
    "Bandingkan 4 model pada holdout 30 hari terakhir. "
    "MAPE terendah = model paling akurat."
)

bt_col1, bt_col2 = st.columns([3, 1])
with bt_col1:
    bt_horizon = st.slider(
        "Horizon backtest (hari)", 7, 60, 30,
    )
with bt_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_bt = st.button("🧪 Jalankan Backtest", use_container_width=True)

if run_bt:
    with st.spinner("Training & evaluasi 4 model..."):
        bt = run_backtest_cached(kom, prov, bt_horizon)

    if "mape" in bt.columns and bt["mape"].notna().any():
        display = bt.copy()
        display["mape"] = display["mape"].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
        )
        display["coverage"] = display["coverage"].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
        )
        for c in ["rmse", "mae"]:
            display[c] = display[c].round(0).astype("Int64", errors="ignore")
        display.columns = ["Model", "MAPE", "RMSE", "MAE", "Coverage 80%"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        best = bt.iloc[0]
        st.success(
            f"🏆 **{best['model']}** juara dengan MAPE {best['mape']:.2f}% "
            f"— gunakan untuk {kom} di {prov}"
        )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    """
    <div class="footer">
        Dibuat oleh <b>Alif Towew</b> • <span class="tag">#SemuaBisaDihitung</span>
        <br>
        <span style="font-size:0.8rem; color:#9ca3af;">
            Dashboard prediksi harga pangan Indonesia •
            Multi-model forecasting (Prophet, LightGBM, XGBoost, Ensemble)
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
