"""
app.py
======
Dasbor Prediksi Pangan Indonesia v2
-----------------------------------
- Data: 2022-2026+ (Excel multi-sheet, 38 provinsi, 25 komoditas)
- Model: Prophet, LightGBM, XGBoost, Ensemble (auto-pick via backtest)
- Upload data baru via UI → cache auto-update
- UI: single-page scroll, auto-run prediksi

Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    data_summary, get_timeseries,
    list_komoditas, list_provinsi, load_cache, update_from_excel,
)
from models import AVAILABLE_MODELS, make_forecaster
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
# STYLING
# ==========================================================
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
.stMetric {
    background: #f9fafb; padding: 0.75rem; border-radius: 8px;
    border: 1px solid #e5e7eb;
}
.section-gap { margin-top: 2.5rem; margin-bottom: 1rem; }
h2 { margin-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR: Data Management + Upload Instructions
# ==========================================================
with st.sidebar:
    st.markdown("## 📁 Manajemen Data")

    # --- Cek data ada belum ---
    df_preview = load_cache()
    default_expanded = df_preview is None or df_preview.empty

    # --- Instruksi Upload ---
    with st.expander("📖 Cara Upload Data Baru", expanded=default_expanded):
        st.markdown("""
**Langkah-langkah:**

1. Siapkan file Excel (`.xlsx`) dengan format yang benar (lihat di bawah)
2. Klik **Browse files** di form upload
3. Pilih file `.xlsx` dari komputermu
4. Klik tombol **🔄 Update Data** yang muncul

**Format Excel yang diterima:**

- Tiap **sheet** Excel = 1 tahun (nama sheet: `2022`, `2023`, `2024`, dst.)
- Kolom wajib: **`Provinsi`** dan **`Tanggal`**
- Kolom lainnya = komoditas (harga dalam Rupiah)

**Contoh isi 1 sheet:**

| Provinsi | Tanggal | Beras Premium | Cabai Rawit Merah | ... |
|----------|---------|---------------|-------------------|-----|
| Aceh | 2022-01-01 | 12112 | 23000 | ... |
| Aceh | 2022-01-02 | 12056 | 24056 | ... |
| Bali | 2022-01-01 | 11800 | 45000 | ... |

**Yang perlu diperhatikan:**

- Data baru akan **digabung** dengan data lama secara otomatis
- Jika ada baris yang sama (Tanggal + Provinsi + Komoditas), data baru **menimpa** data lama
- Harga bernilai 0 atau kosong akan diabaikan
- Sumber data rekomendasi: [panelharga.badanpangan.go.id](https://panelharga.badanpangan.go.id)
""")

    # --- File uploader ---
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
                st.cache_data.clear()
                st.cache_resource.clear()
            st.success(f"✅ Data diperbarui: {len(df_new):,} baris")
            st.rerun()

    st.divider()

    # --- Status Data ---
    @st.cache_data
    def _load():
        return load_cache()

    df_all = _load()

    if df_all is None or df_all.empty:
        st.warning("⚠️ Belum ada data. Silakan upload Excel terlebih dulu.")
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

    st.divider()
    st.caption("Dibuat oleh **Alif Towew** • #SemuaBisaDihitung")

# ==========================================================
# HEADER
# ==========================================================
st.title("🌾 Dasbor Prediksi Harga Pangan Indonesia")
st.markdown(
    "Multi-model forecasting — **Prophet + LightGBM + XGBoost + Ensemble**. "
    "Pilih komoditas dan wilayah, prediksi muncul otomatis."
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
        AVAILABLE_MODELS + ["Auto (best via backtest)"],
        index=0,
        help=(
            "Auto = jalankan backtest 30 hari, pilih model dengan MAPE terendah. "
            "Lebih akurat tapi butuh waktu lebih lama."
        ),
    )
with c4:
    horizon = st.number_input(
        "📅 Hari", min_value=7, max_value=365, value=60, step=1,
        help="Berapa hari ke depan yang ingin diprediksi",
    )

# ==========================================================
# CACHED HELPERS (supaya ganti parameter tidak lambat)
# ==========================================================
@st.cache_resource(show_spinner=False)
def get_trained_model(model_name: str, komoditas: str, provinsi: str):
    """Cache model terlatih per (model, komoditas, provinsi).
    Ganti horizon tidak perlu retrain."""
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
# SECTION 1: PREDIKSI (auto-run)
# ==========================================================
st.markdown("## 📈 Hasil Prediksi")

# Resolve model name (auto vs manual)
if model_choice == "Auto (best via backtest)":
    with st.spinner("🔬 Backtest 4 model untuk pilih yang terbaik... (1-2 menit pertama kali)"):
        bt_for_auto = run_backtest_cached(kom, prov, 30)
        chosen_name = best_model_name(bt_for_auto)
    st.info(
        f"🏆 Model terpilih berdasarkan backtest 30 hari: **{chosen_name}** "
        f"(MAPE {bt_for_auto.iloc[0]['mape']:.2f}%)"
    )
else:
    chosen_name = model_choice

# Training + prediksi (auto-run)
with st.spinner(f"🤖 Training {chosen_name} & prediksi {horizon} hari..."):
    model, ts = get_trained_model(chosen_name, kom, prov)

if model is None:
    st.error(
        f"Data tidak cukup untuk {kom} di {prov} "
        f"(hanya {len(ts)} hari, minimal 90)."
    )
else:
    pred = model.predict(int(horizon))

    # --- Metrik ---
    harga_terakhir = float(ts["y"].iloc[-1])
    harga_prediksi = float(pred["yhat"].iloc[-1])
    delta = harga_prediksi - harga_terakhir
    pct = (delta / harga_terakhir) * 100 if harga_terakhir else 0
    tren_emoji = "📈" if delta > 0 else ("📉" if delta < 0 else "➖")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Harga terakhir", f"Rp {int(harga_terakhir):,}",
        help=f"Per {ts['ds'].iloc[-1].strftime('%d %b %Y')}",
    )
    m2.metric(
        f"Prediksi +{horizon} hari {tren_emoji}",
        f"Rp {int(harga_prediksi):,}", f"{pct:+.1f}%",
    )
    m3.metric(
        "Batas bawah (P10)",
        f"Rp {int(pred['yhat_lower'].iloc[-1]):,}",
    )
    m4.metric(
        "Batas atas (P90)",
        f"Rp {int(pred['yhat_upper'].iloc[-1]):,}",
    )

    # --- Chart ---
    fig = go.Figure()
    cutoff = ts["ds"].max() - pd.Timedelta(days=180)
    hist_view = ts[ts["ds"] >= cutoff]

    fig.add_trace(go.Scatter(
        x=hist_view["ds"], y=hist_view["y"],
        mode="lines", name="Aktual (180 hari terakhir)",
        line=dict(color="#111827", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([pred["ds"], pred["ds"][::-1]]),
        y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.15)",
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
        height=460,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Tabel prediksi (expander) ---
    with st.expander("📋 Lihat tabel prediksi harian + download CSV"):
        tbl = pred.copy()
        tbl["ds"] = tbl["ds"].dt.strftime("%d %b %Y")
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            tbl[c] = tbl[c].round(0).astype(int)
        tbl.columns = ["Tanggal", "Prediksi", "Batas Bawah", "Batas Atas"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            file_name=f"prediksi_{kom}_{prov}_{horizon}hari.csv",
            mime="text/csv",
        )

# ==========================================================
# SECTION 2: BACKTEST
# ==========================================================
st.markdown("## 🧪 Backtest — Perbandingan Akurasi Model")
st.caption(
    "Membandingkan 4 model pada 30 hari terakhir data (holdout). "
    "MAPE terendah = paling akurat."
)

bt_col1, bt_col2 = st.columns([3, 1])
with bt_col1:
    bt_horizon = st.slider(
        "Horizon backtest (hari)", 7, 60, 30,
        help="Berapa hari terakhir yang dijadikan test set",
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
            f"— gunakan model ini untuk {kom} di {prov}"
        )

        with st.expander("ℹ️ Cara baca metrik"):
            st.markdown("""
- **MAPE** (*Mean Absolute Percentage Error*): rata-rata error dalam persen. Semakin kecil semakin bagus.
- **RMSE**: akar rata-rata error kuadrat (Rp). Memberi penalti lebih untuk error besar.
- **MAE**: rata-rata error absolut (Rp). Lebih robust ke outlier.
- **Coverage 80%**: persentase hari aktual jatuh di dalam interval prediksi. Idealnya ~80%.
""")
    else:
        st.error("Backtest gagal. Data mungkin tidak cukup.")

# ==========================================================
# SECTION 3: DATA OVERVIEW (always visible)
# ==========================================================
st.markdown("## 📋 Riwayat Data")

ts_view = get_timeseries(df_all, kom, provinsi=prov)

s1, s2, s3 = st.columns(3)
s1.metric("Jumlah hari", f"{len(ts_view):,}")
s2.metric("Harga rata-rata", f"Rp {int(ts_view['y'].mean()):,}")
cv = (ts_view["y"].std() / ts_view["y"].mean() * 100) if ts_view["y"].mean() else 0
s3.metric(
    "Volatilitas (CV)", f"{cv:.1f}%",
    help="Coefficient of Variation — semakin tinggi, harga semakin fluktuatif",
)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=ts_view["ds"], y=ts_view["y"],
    mode="lines", line=dict(color="#111827", width=1.5),
    name="Harga",
))
fig_hist.update_layout(
    title=f"Riwayat Harga {kom} — {prov} (Seluruh Periode)",
    xaxis_title="Tanggal", yaxis_title="Harga (Rp)",
    height=380, margin=dict(l=40, r=20, t=50, b=40),
)
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("🔍 Lihat data mentah (raw data semua provinsi)"):
    raw = df_all[df_all["Komoditas"] == kom][
        ["Tanggal", "Provinsi", "Harga"]
    ].sort_values(["Tanggal", "Provinsi"], ascending=[False, True])
    st.dataframe(raw.head(500), use_container_width=True, hide_index=True)
    st.caption(f"Menampilkan 500 baris pertama dari {len(raw):,} total.")
