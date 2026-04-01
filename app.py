import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta
import yfinance as yf
import requests
import warnings
warnings.filterwarnings('ignore')

# --- PENGATURAN HALAMAN STREAMLIT ---
st.set_page_config(page_title="Dasbor Prediksi Pangan", page_icon="📈", layout="wide")

# --- 1. MEMUAT DATA DENGAN CACHE (AGAR SUPER CEPAT) ---
@st.cache_data
def load_data():
    df = pd.read_csv('data_pangan_bersih.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

df_pangan = load_data()
daftar_komoditas = df_pangan.columns.drop('Tanggal').tolist()

# KELOMPOK BARANG
komoditas_impor = [
    'Bawang Putih Honan', 'Bawang Putih Kating', 'Kedelai Impor', 
    'Daging Kerbau Impor Beku', 'Daging Sapi Impor Beku', 'Tepung Terigu'
]

sentra_pangan = {
    'Bawang Merah': ('Brebes', -6.8694, 109.0436),
    'Cabai Rawit Merah': ('Garut', -7.2278, 107.9087),
    'Cabai Rawit Hijau': ('Garut', -7.2278, 107.9087),
    'Cabai Merah Keriting': ('Garut', -7.2278, 107.9087),
    'Cabai Merah Besar': ('Kediri', -7.8202, 112.0117),
    'Beras Medium': ('Indramayu', -6.3275, 108.3249),
    'Beras Premium': ('Indramayu', -6.3275, 108.3249),
    'Beras Khusus': ('Indramayu', -6.3275, 108.3249),
    'Tomat': ('Lembang', -6.8148, 107.6186),
    'Sawi Hijau': ('Lembang', -6.8148, 107.6186),
    'Kangkung': ('Lembang', -6.8148, 107.6186),
    'Ketimun Sedang': ('Lembang', -6.8148, 107.6186),
    'Kacang Panjang': ('Lembang', -6.8148, 107.6186),
    'Kentang Sedang': ('Wonosobo', -7.3604, 109.9019),
    'Pisang Lokal': ('Malang', -7.9797, 112.6304),
    'Jeruk Lokal': ('Malang', -7.9797, 112.6304),
    'Ketela Pohon': ('Malang', -7.9797, 112.6304),
    'Jagung Lokal Pipilan': ('Tuban', -6.8976, 112.0649),
    'Kedelai Lokal': ('Grobogan', -7.0287, 110.9136),
    'Kacang Tanah': ('Grobogan', -7.0287, 110.9136),
    'Kacang Hijau': ('Grobogan', -7.0287, 110.9136),
    'Daging Ayam Ras': ('Blitar', -8.0983, 112.1681),
    'Telur Ayam Ras': ('Blitar', -8.0983, 112.1681),
    'Daging Ayam Kampung': ('Blitar', -8.0983, 112.1681),
    'Telur Ayam Kampung': ('Blitar', -8.0983, 112.1681),
    'Daging Sapi Paha Belakang': ('Boyolali', -7.5330, 110.5939),
    'Daging Sapi Paha Depan': ('Boyolali', -7.5330, 110.5939),
    'Daging Sapi Sandung Lamur': ('Boyolali', -7.5330, 110.5939),
    'Daging Sapi Tetelan': ('Boyolali', -7.5330, 110.5939),
    'Ikan Bandeng': ('Semarang', -6.9666, 110.4166),
    'Ikan Kembung': ('Semarang', -6.9666, 110.4166),
    'Ikan Tongkol': ('Semarang', -6.9666, 110.4166),
    'Ikan Teri': ('Semarang', -6.9666, 110.4166),
    'Udang Basah': ('Semarang', -6.9666, 110.4166),
    'Garam Halus': ('Semarang', -6.9666, 110.4166),
}

# --- 2. FUNGSI ROBOT API ---
@st.cache_data(ttl=3600) # Cache kedaluwarsa tiap 1 jam agar data cuaca selalu segar
def get_cuaca_historis(lat, lon, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=Asia%2FJakarta"
    try:
        respon = requests.get(url).json()
        df = pd.DataFrame({'Tanggal': pd.to_datetime(respon['daily']['time']), 'Curah_Hujan': respon['daily']['precipitation_sum']})
        df['Curah_Hujan'] = df['Curah_Hujan'].fillna(0)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_prakiraan_cuaca(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=Asia%2FJakarta&forecast_days=14"
    try:
        respon = requests.get(url).json()
        df = pd.DataFrame({'ds': pd.to_datetime(respon['daily']['time']), 'Curah_Hujan_Forecast': respon['daily']['precipitation_sum']})
        return df
    except: return None

# --- 3. MESIN PREDIKSI UTAMA ---
def prediksi_harga(komoditas, hari_kedepan):
    tanggal_mulai = df_pangan['Tanggal'].min().strftime('%Y-%m-%d')
    tanggal_akhir_data = df_pangan['Tanggal'].max().strftime('%Y-%m-%d')
    
    df_prophet = df_pangan[['Tanggal', komoditas]].copy()
    
    is_impor = komoditas in komoditas_impor
    is_sensitif_cuaca = komoditas in sentra_pangan
    
    if is_impor:
        df_kurs_live = yf.download("USDIDR=X", start=tanggal_mulai, progress=False).reset_index()
        if isinstance(df_kurs_live.columns, pd.MultiIndex):
            df_kurs_live.columns = df_kurs_live.columns.get_level_values(0)
        df_kurs_live = df_kurs_live[['Date', 'Close']]
        df_kurs_live.columns = ['Tanggal', 'Kurs']
        df_kurs_live['Tanggal'] = pd.to_datetime(df_kurs_live['Tanggal']).dt.tz_localize(None)
        df_prophet = pd.merge(df_prophet, df_kurs_live, on='Tanggal', how='left')
        df_prophet['Kurs'] = df_prophet['Kurs'].ffill().bfill()
        
    elif is_sensitif_cuaca:
        kota, lat, lon = sentra_pangan[komoditas]
        df_cuaca = get_cuaca_historis(lat, lon, tanggal_mulai, tanggal_akhir_data)
        if df_cuaca is not None:
            df_prophet = pd.merge(df_prophet, df_cuaca, on='Tanggal', how='left')
            df_prophet['Curah_Hujan'] = df_prophet['Curah_Hujan'].fillna(0)
        else: df_prophet['Curah_Hujan'] = 0

    df_prophet = df_prophet.rename(columns={'Tanggal': 'ds', komoditas: 'y'})
    
    # Peredam Sensitivitas Liburan (Holiday Prior Scale diset ke 0.1 agar tidak over-fitting)
    kalender_lebaran = pd.DataFrame({
      'holiday': 'Musim_Lebaran',
      'ds': pd.to_datetime(['2024-04-10', '2025-03-31', '2026-03-20']),
      'lower_window': -30, 'upper_window': 7,
    })
    
    if is_impor:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor('Kurs')
    elif is_sensitif_cuaca:
        model = Prophet(holidays=kalender_lebaran, holidays_prior_scale=0.1, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor('Curah_Hujan')
    else:
        model = Prophet(holidays=kalender_lebaran, holidays_prior_scale=0.1, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        
    model.add_country_holidays(country_name='ID')
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=int(hari_kedepan))
    
    if is_impor:
        mapping_kurs = dict(zip(df_prophet['ds'], df_prophet['Kurs']))
        future['Kurs'] = future['ds'].map(mapping_kurs)
        mapping_live = dict(zip(df_kurs_live['Tanggal'], df_kurs_live['Kurs']))
        future['Kurs'] = future['Kurs'].fillna(future['ds'].map(mapping_live)).ffill() 
        
    elif is_sensitif_cuaca:
        mapping_cuaca = dict(zip(df_prophet['ds'], df_prophet['Curah_Hujan']))
        future['Curah_Hujan'] = future['ds'].map(mapping_cuaca)
        
        df_forecast = get_prakiraan_cuaca(lat, lon)
        if df_forecast is not None:
            mapping_forecast = dict(zip(df_forecast['ds'], df_forecast['Curah_Hujan_Forecast']))
            future['Curah_Hujan'] = future['Curah_Hujan'].fillna(future['ds'].map(mapping_forecast))
            
        rata_rata_hujan = df_prophet['Curah_Hujan'].mean()
        future['Curah_Hujan'] = future['Curah_Hujan'].fillna(rata_rata_hujan)
        
    forecast = model.predict(future)
    
    # --- PEMBUATAN GRAFIK ---
    fig_zoom = plt.figure(figsize=(10, 5))
    batas_zoom_mundur = df_prophet['ds'].max() - timedelta(days=30)
    df_zoom = df_prophet[df_prophet['ds'] >= batas_zoom_mundur]
    forecast_zoom = forecast[forecast['ds'] >= batas_zoom_mundur]
    plt.plot(df_zoom['ds'], df_zoom['y'], 'k.', markersize=10, label='Aktual (30 Hari Terakhir)')
    plt.plot(forecast_zoom['ds'], forecast_zoom['yhat'], 'b-', linewidth=2, label='Prediksi Tren')
    plt.fill_between(forecast_zoom['ds'], forecast_zoom['yhat_lower'], forecast_zoom['yhat_upper'], color='blue', alpha=0.2)
    plt.title(f'Fokus Jangka Pendek: Harga {komoditas}', fontsize=14, fontweight='bold')
    plt.xlabel('Tanggal'); plt.ylabel('Harga (Rp)'); plt.xticks(rotation=45)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()

    tabel = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(int(hari_kedepan)).copy()
    harga_akhir_prediksi = tabel['yhat'].iloc[-1]
    harga_terakhir_aktual = df_prophet['y'].iloc[-1]
    selisih = harga_akhir_prediksi - harga_terakhir_aktual
    tren = "naik 📈" if selisih > 0 else "turun 📉" if selisih < 0 else "stabil ➖"
    
    if is_impor: info_tambahan = f"\n\n*(💡 **AI Info:** Terkalibrasi dengan Kurs Dolar Live).* "
    elif is_sensitif_cuaca: info_tambahan = f"\n\n*(⛈️ **AI Info:** Terhubung ke Prakiraan Cuaca 14 Hari satelit **{kota}**).* "
    else: info_tambahan = f"\n\n*(🏭 **AI Info:** Komoditas olahan. Prediksi berdasarkan tren musiman).* "
        
    deskripsi = f"### 📝 Kesimpulan AI:\nData terakhir mencatat harga **{komoditas}** di angka **Rp {int(harga_terakhir_aktual):,}**.\n\nDalam **{hari_kedepan} hari ke depan**, diproyeksikan tren harga akan **{tren}** menuju rata-rata **Rp {int(harga_akhir_prediksi):,}**.{info_tambahan}"
    
    tabel['ds'] = tabel['ds'].dt.strftime('%d-%m-%Y')
    tabel['yhat'] = tabel['yhat'].round(0).astype(int)
    tabel['yhat_lower'] = tabel['yhat_lower'].round(0).astype(int)
    tabel['yhat_upper'] = tabel['yhat_upper'].round(0).astype(int)
    tabel.columns = ['Tanggal', 'Prediksi Harga', 'Batas Bawah', 'Batas Atas']
    
    fig_full = plt.figure(figsize=(10, 5))
    plt.plot(df_prophet['ds'], df_prophet['y'], 'k.', label='Aktual')
    plt.plot(forecast['ds'], forecast['yhat'], 'b-', label='Prediksi Tren')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2, label='Rentang Prediksi')
    plt.title(f'Tren Jangka Panjang: Harga {komoditas}')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
    
    fig_komponen = model.plot_components(forecast)
    return fig_zoom, deskripsi, tabel, fig_full, fig_komponen

# --- 4. ANTARMUKA USER (UI) STREAMLIT ---
st.title("📈 Dasbor Cerdas Prediksi Harga Pangan")
st.markdown("Integrasi Live API: Saham Global & Satelit Cuaca Open-Meteo dengan Predictive AI")
st.divider()

# Membagi layar: Sidebar untuk input, area utama untuk output
with st.sidebar:
    st.header("⚙️ Pengaturan Prediksi")
    pilihan_komoditas = st.selectbox("1. Pilih Komoditas", daftar_komoditas, index=daftar_komoditas.index('Cabai Rawit Merah'))
    jumlah_hari = st.slider("2. Prediksi Berapa Hari ke Depan?", min_value=7, max_value=180, value=60, step=1)
    tombol_jalankan = st.button("Mulai Prediksi 🚀", use_container_width=True, type="primary")

# Jika tombol ditekan
if tombol_jalankan:
    with st.spinner(f"🤖 AI sedang menganalisis data {pilihan_komoditas}..."):
        # Panggil fungsi AI
        g_zoom, teks_desc, df_tabel, g_full, g_komponen = prediksi_harga(pilihan_komoditas, jumlah_hari)
        
        # Tampilkan hasil di layar utama
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(g_zoom)
        with col2:
            st.markdown(teks_desc)
            
        st.divider()
        st.subheader("📋 Tabel Prediksi Harga Harian")
        st.dataframe(df_tabel, use_container_width=True)
        
        st.divider()
        st.subheader("📅 Riwayat Keseluruhan")
        st.pyplot(g_full)
        
        st.divider()
        st.subheader("🧠 Analisis Komponen AI")
        st.pyplot(g_komponen)
