# 🌾 Prediksi Pangan Indonesia v2.1

Dasbor forecasting harga pangan Indonesia dengan multi-model + data per provinsi + visual lengkap.

Dibuat oleh **Alif Towew** • #SemuaBisaDihitung

## ⚡ Apa yang Baru di v2.1

- **Pre-compute forecasts** — prediksi nasional sudah dihitung di disk, load **instant** (0.1 detik, dari 7+ detik)
- **On-demand caching per provinsi** — provinsi yang diklik akan di-cache, akses berikutnya instant
- **Sparklines grid** — overview 25 komoditas dalam 1 layar (mini chart dengan indikator naik/turun)
- **Small multiples** — grid 5×5 semua komoditas + prediksi overlay (ala FT/Tufte)
- **Peta Indonesia** — choropleth bubble map per provinsi (size + color = harga)
- **Seasonal decompose** — pisahkan harga jadi Trend / Seasonal / Residual (statistik klasik)
- **Footer branded** — #SemuaBisaDihitung di setiap halaman

## 🏗️ Arsitektur

```
prediksi-pangan/
├── app.py                  # Streamlit UI (single-page scroll)
├── data_loader.py          # Excel → parquet + merge update
├── models.py               # Prophet / LightGBM / XGBoost / Ensemble
├── backtest.py             # Walk-forward CV + auto-pick best
├── precompute.py           # Pre-compute nasional, cache on-demand per provinsi
├── visuals.py              # Sparklines, small multiples, peta, decompose
├── requirements.txt
├── README.md
└── data/
    ├── pangan_long.parquet    # Data mentah long-format
    ├── forecasts.parquet      # Cache prediksi (instant access)
    └── forecasts_meta.json    # Metadata: model, fingerprint, timestamp
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inisialisasi data (sekali saja)
python data_loader.py Data_Konsumen_Harian_Per_Provinsi_2022-2026.xlsx

# 3. Pre-compute prediksi (3 menit, sekali saja)
python precompute.py

# 4. Jalankan dasbor
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

> **Catatan:** Kalau `python precompute.py` dilewati, dashboard akan meminta kamu untuk menjalankannya via tombol saat pertama kali dibuka.

## 🔄 Update Data (3 Cara)

### Cara 1: Via UI (paling mudah)
1. Buka dasbor
2. Sidebar kiri → **📁 Upload Excel data pangan**
3. Pilih file `.xlsx`
4. Klik **🔄 Update Data**
5. Cache prediksi lama otomatis di-invalidate → tombol **▶️ Mulai Pre-compute** akan muncul

### Cara 2: CLI
```bash
python data_loader.py path/ke/file_baru.xlsx
python precompute.py  # re-compute setelah data baru
```

### Cara 3: Programmatic
```python
from data_loader import update_from_excel
from precompute import compute_all_forecasts, save_forecasts

df = update_from_excel("data_baru.xlsx")
fc = compute_all_forecasts(df)
save_forecasts(fc, df, "LightGBM")
```

### Format Excel
- Tiap sheet = 1 tahun (`2022`, `2023`, dst.)
- Kolom wajib: `Provinsi`, `Tanggal`
- Kolom lain = komoditas (harga Rupiah)

## 🧠 Model yang Digunakan

### 1. Prophet (Meta)
- Kekuatan: seasonality tahunan, efek Lebaran (calendar), hari libur Indonesia
- Config: Lebaran 2022-2027, `holidays_prior_scale=0.5`, `changepoint_prior_scale=0.05`
- Lebih cocok untuk komoditas stabil (beras, tepung, gula)

### 2. LightGBM (Microsoft)
- Kekuatan: non-linear, lag features, rolling statistics, quantile regression
- Features: lag {1,2,3,7,14,30}, rolling mean/std 7 & 30 hari, calendar, days_to_lebaran
- Uncertainty: quantile regression α=0.1 & α=0.9
- **Default model untuk pre-compute** (paling akurat di komoditas volatile)

### 3. XGBoost
- Alternatif LightGBM. Kadang menang, kadang kalah — backtest yang menentukan.

### 4. Ensemble
- Rata-rata Prophet + LightGBM + XGBoost. Paling robust.

### Mode Auto
Pilih **"Auto (best via backtest)"**:
1. Split: 30 hari terakhir = test, sisanya = train
2. Fit 4 model, prediksi 30 hari
3. Pilih MAPE terendah
4. Training ulang di full data dengan model pemenang

## 📊 Contoh Hasil Backtest

**Cabai Rawit Merah** (30-day holdout, data 2022-2026):

| Model | MAPE | RMSE | Coverage 80% |
|---|---|---|---|
| **LightGBM** 🏆 | **4.45%** | 4.347 | 20% |
| XGBoost | 6.10% | 6.597 | 33% |
| Ensemble | 9.13% | 7.677 | 27% |
| Prophet | 19.59% | 14.889 | 0% |

→ Untuk komoditas volatile, LightGBM ~4x lebih akurat dari Prophet.

## 🎨 Komponen Visual

**1. Sparklines Grid** (25 mini-chart)
- Overview 90 hari terakhir untuk semua komoditas
- Hijau ▼ = turun, merah ▲ = naik
- Format: kartu kompak dengan harga + delta %

**2. Small Multiples** (5×5 grid)
- Tiap sel: 1 komoditas × 180 hari history + prediksi overlay
- Scan cepat pola makro lintas komoditas

**3. Peta Indonesia**
- Scatter geo bubble per provinsi
- Size + color = harga (hijau murah, merah mahal)
- Ideal untuk spasial: provinsi mana paling mahal/murah

**4. Seasonal Decomposition**
- Pisahkan observasi menjadi: Trend + Seasonal + Residual
- Pakai classical decomposition (centered moving average, period=365)
- Berguna untuk edukatif: "kenaikan ini karena trend atau Lebaran?"

## ⚡ Speed Benchmarks

| Aksi | Sebelum | Sesudah | Speedup |
|---|---|---|---|
| Load prediksi nasional | 7-10 detik | **0.1 detik** | 70-100x |
| Ganti horizon | 5 detik | **Instant** | - |
| Provinsi pertama kali | 7 detik | **7 detik** (lalu cached) | - |
| Provinsi kedua kali | 5 detik | **Instant** | - |

Pre-compute cost: **~3 menit sekali** di startup, setelah itu dashboard feel snappy.

## 🎯 Tips

- **Komoditas stabil** (Beras, Tepung, Gula) → Prophet OK
- **Komoditas volatile** (Cabai, Bawang) → LightGBM/XGBoost jauh lebih baik
- **Horizon panjang** (>90 hari): Akurasi menurun karena compound error pada recursive forecast
- **Per provinsi**: Lebih noisy dari nasional, MAPE biasanya lebih tinggi
- **Coverage 80% rendah** = interval prediksi terlalu sempit (trade-off akurasi vs cakupan)

## 🔧 Pengembangan Lanjutan

- [ ] Conformal prediction untuk interval yang coverage terjamin
- [ ] External regressors: kurs USD/IDR (komoditas impor), curah hujan (komoditas sentra)
- [ ] Direct multi-step forecasting (per-horizon model)
- [ ] Hierarchical forecasting: rekonsiliasi nasional vs provinsi
- [ ] NeuralProphet / TimesFM / Chronos (foundation models)
- [ ] Deploy ke Vercel (frontend Next.js + pre-computed JSON)

## 📝 License

MIT

---

**Dibuat oleh Alif Towew** • #SemuaBisaDihitung
