# 🚀 Panduan Deploy Dashboard Pangan

Panduan step-by-step untuk deploy dashboard ke Streamlit Cloud (gratis, terkoneksi GitHub).

## Prasyarat

- Akun GitHub (kamu sudah punya: `aliftowew`)
- Akun [share.streamlit.io](https://share.streamlit.io) (gratis, login pakai GitHub)
- Git di komputer lokal

## Step 1: Persiapkan Repo GitHub

### Opsi A: Update Repo Lama (rekomendasi)

Kalau kamu sudah punya repo `prediksi-pangan-indonesia`:

```bash
# 1. Clone repo lama
git clone https://github.com/aliftowew/prediksi-pangan-indonesia.git
cd prediksi-pangan-indonesia

# 2. Hapus semua file lama (kecuali .git dan LICENSE)
find . -maxdepth 1 ! -name '.git' ! -name 'LICENSE' ! -name '.' -exec rm -rf {} +

# 3. Extract v2.1 dan copy isinya ke sini
unzip ~/Downloads/prediksi-pangan-v2.zip
cp -r prediksi-pangan-v2/* prediksi-pangan-v2/.[!.]* .
rm -rf prediksi-pangan-v2

# 4. Commit & push
git add -A
git commit -m "feat: v2.1 - multi-model forecasting + 4 visual baru + speed optimization

- Data 2022-2026 per provinsi (38 provinsi, 25 komoditas, 1M+ rows)
- 4 model: Prophet, LightGBM, XGBoost, Ensemble (auto-pick via backtest)
- Pre-compute cache: load prediksi 0.003s (dari 7-10s)
- 4 visual baru: sparklines, small multiples, peta Indonesia, seasonal decompose
- Upload Excel via UI untuk update data
- Footer branded #SemuaBisaDihitung"
git push
```

### Opsi B: Repo Baru

```bash
cd ~/Downloads
unzip prediksi-pangan-v2.zip
cd prediksi-pangan-v2
git init
git add .
git commit -m "feat: dashboard prediksi pangan Indonesia v2.1"
git branch -M main
git remote add origin https://github.com/aliftowew/prediksi-pangan-v2.git
git push -u origin main
```

## Step 2: Deploy ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **New app** (pojok kanan atas)
4. Isi form:
   - **Repository**: `aliftowew/prediksi-pangan-indonesia`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (opsional): `prediksi-pangan-alif`
5. Klik **Advanced settings...**
   - **Python version**: `3.11` (paling stabil untuk Prophet)
6. Klik **Deploy**

Tunggu 2-5 menit untuk build. Setelah selesai, app akan tersedia di:
`https://prediksi-pangan-alif.streamlit.app`

## Step 3: Custom Domain (Opsional)

Kalau mau URL custom seperti `pangan.aliftowew.com`:

1. Beli domain (Namecheap, Cloudflare Registrar, dll)
2. Di DNS provider, tambahkan CNAME record:
   ```
   Type: CNAME
   Name: pangan
   Value: prediksi-pangan-alif.streamlit.app
   ```
3. Di Streamlit Cloud: Settings → Custom domains → Tambahkan `pangan.aliftowew.com`
4. Tunggu propagasi DNS (5-60 menit)

## 🔧 Troubleshooting

### "Error installing Prophet"
Prophet butuh compiler C++. Streamlit Cloud seharusnya sudah handle, tapi kalau error, tambahkan file `packages.txt`:
```
build-essential
```

### "App terlalu besar / memory limit"
Streamlit Cloud free tier: 1 GB RAM. Data parquet kamu 2.6 MB, aman. Tapi kalau ada isu:
- Reduce pre-computed forecast horizon dari 180 → 90 hari
- Uncomment `data/*.parquet` di `.gitignore`, user-upload Excel saat pertama buka

### "Prediksi instant tapi pertama kali terlalu lama"
Di Streamlit Cloud, container "tidur" setelah tidak dipakai → cold start ~30-60 detik.
Untuk mengatasi ini:
- Upgrade ke paid tier ($5/bulan tidak pernah tidur), atau
- Pakai [kaffeine.herokuapp.com](https://kaffeine.herokuapp.com) (gratis, ping setiap 30 menit)

## 📊 Update Data Rutin

Kalau mau otomatis update data bulanan:

### Manual (sederhana)
1. Download data terbaru dari [panelharga.badanpangan.go.id](https://panelharga.badanpangan.go.id)
2. Buka dashboard live
3. Upload via UI → klik Update Data → pre-compute
4. Selesai — data baru terupload ke Streamlit Cloud

### Otomatis (advanced) dengan GitHub Actions

Buat file `.github/workflows/update-data.yml`:

```yaml
name: Update Data Bulanan

on:
  schedule:
    - cron: '0 1 1 * *'  # Tanggal 1 tiap bulan, jam 01:00 UTC
  workflow_dispatch:  # Bisa run manual

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - name: Download data terbaru
        run: |
          # Custom script untuk scrape panelharga.badanpangan.go.id
          python scripts/download_panel_harga.py --output data_latest.xlsx
      - name: Update cache
        run: |
          python data_loader.py data_latest.xlsx
          python precompute.py
      - name: Commit
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add data/
          git commit -m "chore: auto-update data $(date +%Y-%m)" || true
          git push
```

(Script scraper `download_panel_harga.py` perlu dibuat terpisah sesuai struktur API Bapanas.)

## 🎯 Next Steps Setelah Deploy

1. ✅ **Test deploy**: pastikan semua visual load dan upload works
2. 📱 **Test mobile**: buka di HP, cek responsiveness
3. 🔗 **Share link** di bio Instagram / Twitter / LinkedIn
4. 📝 **Konten edukasi**: buat thread Twitter tentang insight dari dashboard
5. 📈 **Analytics**: tambahkan Google Analytics atau Plausible.io

## 💡 Tips untuk Konten #SemuaBisaDihitung

Dashboard ini bisa jadi basis banyak konten:
- **Thread mingguan**: "Top 5 komoditas paling volatile minggu ini"
- **Monthly report**: Screenshot peta Indonesia + analisis regional
- **Explainer video**: "Kenapa LightGBM 4x lebih akurat dari Prophet untuk cabai?"
- **Educational carousel**: "Cara baca seasonal decomposition"
- **Live dashboard di video**: Tunjukkan update data Ramadhan berikutnya

---

Dibuat oleh **Alif Towew** • #SemuaBisaDihitung
