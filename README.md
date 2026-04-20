# 🌾 Prediksi Pangan Indonesia v2

Dasbor forecasting harga pangan Indonesia dengan multi-model + data per provinsi.

## 🆕 Apa yang Baru (v2)

| Aspek | v1 (lama) | v2 (sekarang) |
|---|---|---|
| **Periode data** | 2025-01 → 2026-03 (282 hari) | 2022-01 → 2026-03 (**1.536 hari**, 4x lebih banyak) |
| **Granularitas** | Rata-rata nasional | **38 provinsi** + agregat nasional |
| **Model** | Prophet saja | **Prophet + LightGBM + XGBoost + Ensemble** |
| **Auto-pick** | Tidak ada | **Backtest otomatis pilih model terbaik per komoditas** |
| **Update data** | Edit CSV manual | **Upload Excel via UI** |
| **Siklus Lebaran** | 1 siklus (2025) | **5 siklus** (2022-2026), Prophet belajar pola Ramadhan lebih baik |

## 📦 Struktur Project

```
prediksi-pangan-v2/
├── app.py                 # Streamlit UI
├── data_loader.py         # Excel → parquet + merge update
├── models.py              # Prophet / LightGBM / XGBoost / Ensemble
├── backtest.py            # Walk-forward CV, auto-pick best
├── requirements.txt
├── README.md
└── data/
    └── pangan_long.parquet   # Cache (otomatis dibuat)
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inisialisasi data (sekali saja di awal)
python data_loader.py Data_Konsumen_Harian_Per_Provinsi_2022-2026.xlsx

# 3. Jalankan dasbor
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

## 🔄 Update Data (3 Cara)

### Cara 1: Via UI (paling mudah)
1. Buka dasbor
2. Sidebar kiri → **📁 Upload Excel data pangan**
3. Pilih file `.xlsx` baru
4. Klik **🔄 Update Data**
5. Data baru otomatis digabung dengan data lama. Duplikasi (Tanggal + Provinsi + Komoditas yang sama) akan di-overwrite dengan data baru.

### Cara 2: CLI
```bash
python data_loader.py path/ke/file_baru.xlsx
```

### Cara 3: Programmatic
```python
from data_loader import update_from_excel
df = update_from_excel("data_baru.xlsx")
```

### Format Excel yang Diterima
- **Tiap sheet** = 1 tahun (nama sheet bebas, contoh `2022`, `2023`, dll.)
- **Kolom wajib**: `Provinsi`, `Tanggal`
- **Kolom lain** = komoditas (harga dalam Rupiah)
- Contoh struktur:

| Provinsi | Tanggal | Beras Premium | Beras Medium | ... |
|---|---|---|---|---|
| Aceh | 2022-01-01 | 12112 | 11044 | ... |
| Aceh | 2022-01-02 | 12056 | 10976 | ... |

## 🧠 Model yang Digunakan

### 1. Prophet (Meta)
- **Kekuatan**: Seasonality tahunan, efek Lebaran (prior_scale 0.5), hari libur Indonesia
- **Lemah**: Kalah pada komoditas volatile (Cabai, Bawang) karena additive model
- **Config**: Lebaran 2022-2027, changepoint_prior_scale 0.05

### 2. LightGBM (Microsoft)
- **Kekuatan**: Non-linear, lag features, rolling statistics, quantile regression untuk interval
- **Features**: lag_{1,2,3,7,14,30}, rollmean/std 7 & 30 hari, calendar features, days_to_lebaran
- **Uncertainty**: Quantile regression α=0.1 (lower) & α=0.9 (upper)
- **Juara** M5 Forecasting Competition

### 3. XGBoost
- Alternatif LightGBM dengan feature set yang sama. Kadang lebih akurat, kadang kalah — karena itu ada backtest.

### 4. Ensemble
- Rata-rata prediksi Prophet + LightGBM + XGBoost (weighted equal)
- Paling robust jika tidak mau pilih satu model

### Mode Auto (rekomendasi)
Pilih **"Auto (best via backtest)"** di dropdown model:
1. Split data: 30 hari terakhir = test, sisanya = train
2. Fit 4 model, prediksi 30 hari
3. Hitung MAPE, pilih model dengan MAPE terendah
4. Train ulang di seluruh data dengan model pemenang, prediksi ke depan

## 📊 Contoh Hasil Backtest

Cabai Rawit Merah (30 hari holdout):

| Model | MAPE | RMSE | Coverage 80% |
|---|---|---|---|
| **LightGBM** 🏆 | **4.45%** | 4.347 | 20% |
| XGBoost | 6.10% | 6.597 | 33% |
| Ensemble | 9.13% | 7.677 | 27% |
| Prophet | 19.59% | 14.889 | 0% |

→ Untuk komoditas volatile, LightGBM ~4x lebih akurat dari Prophet.
→ Coverage 20% berarti interval terlalu sempit; bisa diperbaiki dengan tuning quantile atau conformal prediction.

## 🧪 Evaluasi Model Sendiri

```bash
python backtest.py
```

Atau dari Python:

```python
from data_loader import load_cache, get_timeseries
from backtest import backtest_all

df = load_cache()
ts = get_timeseries(df, "Beras Premium", provinsi="Nasional")
result = backtest_all(ts, horizon=30)
print(result)
```

## 🎯 Tips & Catatan

- **Komoditas stabil** (Beras, Tepung, Gula) → Prophet sering oke, hampir sama dengan LightGBM
- **Komoditas volatile** (Cabai, Bawang, Sayuran) → LightGBM/XGBoost jauh lebih baik
- **Interval prediksi**: Saat ini pakai quantile regression 10-90%. Coverage aktual sering <80% untuk komoditas volatile — ini trade-off akurasi vs cakupan interval.
- **Horizon panjang** (>90 hari): Akurasi menurun signifikan karena recursive forecast compound error. Gunakan dengan kehati-hatian.
- **Per provinsi**: Data per provinsi lebih noisy dari nasional (tidak ada averaging). MAPE biasanya lebih tinggi.

## 🔧 Pengembangan Lanjutan

Ide peningkatan (belum diimplementasi):

- [ ] External regressors: Kurs USD/IDR untuk komoditas impor, curah hujan untuk komoditas sentra
- [ ] Direct multi-step forecasting (per-horizon model) untuk kurangi compound error
- [ ] Conformal prediction untuk interval yang coverage-nya terjamin
- [ ] Hierarchical forecasting: reconcile nasional vs provinsi (aggregate consistency)
- [ ] NeuralProphet atau TimesFM (foundation model) sebagai tambahan
- [ ] Cache model per-komoditas (joblib) agar tidak retrain setiap klik

## 📝 License

MIT
