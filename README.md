# 🎵 NadaFana — AI-Powered Indie & Mellow Music Discovery

> **Sistem Kurasi dan Rekomendasi Musik Indie & Mellow Berbasis Machine Learning dan NLP Semantik.**

NadaFana lahir dari kebutuhan untuk mengkurasi lagu-lagu berirama tenang, balada, dan *mellow* berbahasa Indonesia secara akurat dan bermakna—melampaui batasan rekomendasi berbasis popularitas standar.

Proyek ini menggabungkan riset *data science* skala besar (pemrosesan 950K+ lagu Spotify) dengan antarmuka web interaktif yang intuitif dan elegan.

---

## 📌 Ringkasan Riset & Machine Learning Pipeline (`NadaFana.ipynb`)

Riset data science dan pemodelan inti pada proyek ini dikerjakan secara komprehensif di dalam notebook [`NadaFana.ipynb`](./NadaFana.ipynb), dengan tahapan arsitektur sebagai berikut:

1. **Pemrosesan Data Skala Penuh (Full Scale ~950K Trek):**
   - Mengolah dataset Spotify berskala masif yang mencakup atribut audio dan lirik lagu.
2. **Clustering Efisien & Deteksi Klaster Otomatis:**
   - Menggunakan `MiniBatchKMeans` untuk mengelompokkan karakteristik audio secara efisien tanpa kendala memori.
   - Klaster *mellow* dideteksi secara dinamis berdasarkan kombinasi nilai *valence* (suasana positif/sedih) dan *energy* terendah dari *centroid* klaster.
3. **Deteksi Bahasa Berbasis AI (fastText):**
   - Menggunakan model `fastText` teroptimasi untuk mengklasifikasikan lirik berbahasa Indonesia secara akurat dan menepis *false positives* dari metode heuristik biasa.
4. **NLP Semantik (Sentence Embeddings):**
   - Merepresentasikan makna lirik ke dalam ruang vektor berdimensi padat menggunakan model transformer `paraphrase-multilingual-MiniLM-L12-v2`.
5. **Kurasi & Ekspor Artefak:**
   - Menghasilkan dataset terfilter yang ringkas dan siap pakai: `df_lagu_mellow_indo.parquet` dan `embeddings_lagu_mellow.npy`.

---

## 🌟 Fitur Aplikasi Web (`app.py`)

Aplikasi interaktif dibangun dengan **Streamlit** mengusung desain UI/UX modern, hangat (*warm orange & psychological comfort theme*), dan bebas dari istilah teknis yang membingungkan:

- **Pencarian Berdasarkan Suasana Hati & Curahan Hati:**
  Ketik apa yang sedang dirasakan (contoh: *"lagu tentang rindu yang tak tersampaikan di malam sunyi"*). Sistem mencocokkan semantik lirik untuk menemukan lagu yang paling relevan.
- **Eksplorasi Lagu Serupa:**
  Pilih lagu favorit untuk menemukan kurasi lagu lain dengan spektrum audio dan kedalaman emosi serupa.
- **Kurasi Cepat (Quick Preset):**
  Pilihan tombol suasana hati instan: *Larut Malam*, *Hening & Sunyi*, *Patah Hati*, *Rindu & Kenangan*, dan *Tenang & Damai*.
- **Pemutar Musik & Tautan Eksternal:**
  Dilengkapi Spotify Embed Player untuk mendengarkan langsung di web, serta tautan cepat ke Spotify dan YouTube Music.
- **Filter Fleksibel:**
  Opsi slider tempo, keheningan audio, serta filter album untuk personalisasi pencarian.

---

## 📂 Struktur Repositori

```text
project_1/
├── .streamlit/
│   └── config.toml               # Konfigurasi tema warna & server Streamlit
├── .gitignore                    # Berkas yang diabaikan oleh Git
├── NadaFana.ipynb                # Notebook riset inti (Data Science, NLP & ML)
├── app.py                        # Aplikasi antarmuka web Streamlit
├── df_lagu_mellow_indo.parquet   # Dataset terkurasi lagu mellow Indonesia (~640 KB)
├── embeddings_lagu_mellow.npy    # Vektor embedding lirik hasil Sentence-Transformers (~820 KB)
├── requirements.txt              # Daftar pustaka dependensi Python
└── README.md                     # Dokumentasi proyek
```

---

## 🚀 Panduan Menjalankan Secara Lokal

### 1. Kloning Repositori
```bash
git clone https://github.com/noorafaelhakimfaudka-droid/NadaFana.git
cd NadaFana
```

### 2. Buat Lingkungan Virtual (Opsional tapi Disarankan)
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi otomatis terbuka di peramban Anda pada alamat `http://localhost:8501`.

---

## ☁️ Deployment ke Streamlit Community Cloud

Aplikasi ini telah dioptimasi agar dapat langsung dideploy ke **Streamlit Community Cloud** secara gratis:
1. *Push* repositori ini ke akun GitHub Anda.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io).
3. Klik **"New app"**, pilih repositori ini, tentukan branch `main`, dan set file utama ke `app.py`.
4. Klik **"Deploy"**!

---

## 🛠️ Tumpukan Teknologi (Tech Stack)

- **Machine Learning & NLP:** `sentence-transformers`, `scikit-learn`, `fastText`, `numpy`
- **Data Engineering:** `pandas`, `pyarrow`
- **Web Application & UI/UX:** `streamlit`, HTML5, CSS3 Glassmorphism

---

## 👤 Penulis & Kontribusi

- **Riset & Pemodelan Machine Learning (`NadaFana.ipynb`):** [Rafael Hakim Souissa](https://github.com/noorafaelhakimfaudka-droid)
- **Pengembangan Web & Optimasi Antarmuka (`app.py`):** NadaFana Project
