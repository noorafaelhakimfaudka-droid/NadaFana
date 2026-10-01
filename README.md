<div align="center">

# 🎵 NadaFana — Nada yang Tak Pernah Fana
### *AI-Powered Indonesian Music Discovery & Multimodal Recommendation Engine*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Sentence--Transformers-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![FAISS](https://img.shields.io/badge/FAISS-Sub--millisecond%20ANN-green.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

*Temukan karya musik Indonesia yang benar-benar berbicara pada jiwamu — melalui kedalaman rasa dalam lirik dan getaran psikoakustik melodi.*

---

</div>

## 📌 Ringkasan Eksekutif & Latar Belakang

Sistem rekomendasi musik konvensional umumnya mengandalkan penyaringan kolaboratif (*collaborative filtering*) atau metadata leksikal (nama artis, genre, dan statistik jumlah pemutaran). Pendekatan ini rentan terhadap:
- **Popularity Bias**: Lagu-lagu baru atau artis non-mainstream terabaikan.
- **Lexical Gap**: Kegagalan menangkap majas, metafora, dan kedalaman narasi puitis lirik lagu Indonesia jika hanya mencocokkan kata kunci harfiah.

**NadaFana** memecahkan masalah ini dengan merekayasa sistem rekomendasi berbasis **pemahaman semantik mendalam (*Transformer Sentence Embeddings*)** dan **penyelarasan psikoakustik audio kontinu (*Multimodal Hybrid Re-ranking*)**, dipadukan dengan pengindeksan berkecepatan sub-milidetik menggunakan **FAISS**.

---

## ✨ Fitur Utama

- 📖 **Curahan Peras (Pencarian Narasi & Suasana Hati):**
  Tuliskan isi hati, kenangan, atau perasaanmu dalam bahasa sehari-hari. Model NLP memetakan narasi pengguna ke dalam ruang vektor lirik untuk menemukan lagu dengan resonansi emosi paling tepat.
- 🎶 **Lagu Serupa (Rekomendasi Berbasis Lagu Acuan):**
  Pilih lagu favoritmu dari katalog, dan temukan rekomendasi lagu lain yang memiliki koherensi cerita dan atmosfer musik yang selaras.
- 🎛️ **Kendali Bobot Multimodal Interaktif:**
  Pengguna dapat mengatur proporsi antara **Kesamaan Makna Lirik** vs. **Karakter Audio** secara dinamis melalui antarmuka web.
- 📊 **Radar Psikoakustik Interaktif:**
  Visualisasi radar multi-dimensi untuk metrik audio Spotify (*Valence, Energy, Acousticness, Danceability*).
- 🎧 **Pemutar Musik Terintegrasi:**
  Dengarkan langsung sampel audio via YouTube atau Spotify Web Player langsung di halaman web tanpa membuka tab baru.
- 💎 **Antarmuka Modern Spotify Glassmorphism:**
  Desain bertema gelap (*dark mode*) yang elegan dengan kartu musik sinematik, micro-animations, dan tata letak responsif.

---

## 🏛️ Arsitektur Sistem & Alur Pipeline

```
[Raw Spotify Dataset (~955k Trek Global)]
                   │
                   ▼
  1. Language Identification (fastText lid.176.bin)
     └─ Isolasi korpus trek berbahasa Indonesia
                   │
                   ▼
  2. Data Cleansing & Normalization
     ├─ Pembersihan tag aransemen ([Chorus], [Verse], metadata)
     ├─ Normalisasi karakter & tanda baca
     └─ Deduplikasi katalog (Spotify ID & Pasangan Judul+Artis)
                   │
                   ▼
  3. Feature Engineering & Embedding
     ├─ Psikoakustik Audio: Valence, Energy, Acousticness, Danceability
     ├─ K-Means Clustering (k=5 Klaster Vibes Audio)
     └─ Transformer Sentence Embeddings: paraphrase-multilingual-mpnet-base-v2 (768 Dimensi)
                   │
                   ▼
  4. High-Speed Indexing (FAISS IndexFlatIP)
     └─ L2 Normalized Dense Vectors -> Cosine Similarity dalam < 2 ms
                   │
                   ▼
  5. Multimodal Hybrid Re-ranking Engine
     ├─ Min-Max Similarity Calibration (Eliminasi Skor Inflasi)
     ├─ Multimodal Fusion: (α × Lirik) + (β × Audio)
     ├─ Artist Affinity Boost (+8% Musisi Sama / Kolaborasi)
     ├─ Cluster Coherence Bonus (+3% Klaster Identik)
     └─ Catalog Diversity Filter (Batas maks 2 trek per artis)
                   │
                   ▼
  6. Production Serving (Streamlit Glassmorphism UI)
```

---

## 🔬 Metodologi Machine Learning

### 1. Deteksi Bahasa Skala Besar (*fastText*)
- Memproses ~955.000 trek mentah Spotify.
- Model probabilistik terawasi `lid.176.bin` mengevaluasi n-gram karakter secara efisien dengan throughput ribuan baris per detik, jauh lebih cepat daripada model klasifikasi berbasis BERT tanpa mengorbankan akurasi klasifikasi bahasa lokal (>98%).

### 2. Dense Semantic Embeddings (*mpnet-base-v2*)
- Menggunakan arsitektur transformator `paraphrase-multilingual-mpnet-base-v2` ($d = 768$).
- Menghasilkan vektor padat yang mampu mengenali ekuivalensi semantik antara kalimat puitis yang berbeda kata leksikalnya (misal: *"hujan membasahi kenangan"* vs *"gerimis mengiringi rindu"*).

### 3. Pengindeksan Sub-Milidetik (*FAISS*)
- Vektor embedding dinormalisasi secara $L_2$, sehingga operasi *Inner Product* ($IP$) identik secara matematis dengan *Cosine Similarity*.
- Struktur indeks `faiss.IndexFlatIP` memberikan hasil eksak 100% (*zero recall loss*) dalam waktu **< 2 milidetik**.

### 4. Optimalisasi Kualitas Rekomendasi (5 Pilar)

| Masalah Baseline | Solusi NadaFana | Dampak |
| :--- | :--- | :--- |
| **Skor Inflasi** (Cosine berkisar 0.50–0.90) | **Min-Max Calibration** | Rentang skor tersebar merata (0.0–1.0), perbedaan relevansi terlihat jelas |
| **Dominasi Audio** (Lagu asing audio mirip) | **Multimodal Fusion Berimbang** | Memadukan bobot semantik lirik dan nada audio secara harmonis |
| **Lagu Duplikat** di Rekomendasi | **Deduplikasi 2-Faktor** (ID & Judul+Artis) | Menghilangkan hasil ganda yang mengotori katalog |
| **Relevansi Musisi Lemah** | **Artist Affinity Boost** (+8% & +4%) | Mengutamakan diskografi musisi referensi secara wajar |
| **Dominasi Satu Artis** | **Diversity Constraint** (Maks 2 trek/artis) | Menjamin keberagaman pilihan lagu bagi pendengar |

---

## 📊 Evaluasi Komparatif (Sebelum vs Sesudah)

Uji coba lagu referensi: **"Apakah Kamu" — Yura Yunita**

| Peringkat | Baseline (Pure FAISS Semantic) | NadaFana Enhanced Hybrid Multimodal |
| :---: | :--- | :--- |
| **#1** | Lemas — *Ruffedge* (0.79) | **Buktikan — Yura Yunita** (Skor: 0.935) ★ |
| **#2** | Bagaimana Kalau Aku Tidak Baik-Baik Saja — *Judika* (0.79) | **Sebelah Mata — D'MASIV** (Skor: 0.879) |
| **#3** | Apa Salahku — *D'MASIV* (0.78) | **Maybe For Me — Suga Free** (Skor: 0.870) |

> **Analisis:** Pada metode baseline lama, lagu Yura Yunita lainnya tidak muncul sama sekali. Pada sistem terpadu NadaFana, lagu "Buktikan" dari Yura Yunita langsung menempati posisi puncak dengan skor tinggi (0.935), diikuti lagu pop Indonesia dengan tempo dan nuansa aransemen yang selaras.

---

## 📁 Struktur Repositori

```
NadaFana/
├── .streamlit/
│   └── config.toml             # Tema gelap Spotify & konfigurasi server Streamlit
├── assets/                     # Aset visual & latar belakang web (opsional)
├── df_lagu_indo.parquet        # Metadata terkurasi & fitur psikoakustik audio
├── embeddings_lagu_indo.npy    # Matriks biner Sentence Embeddings (1508 x 768)
├── index_lagu_indo.faiss       # Indeks FAISS biner untuk inferensi pencarian cepat
├── NadaFana.ipynb              # Notebook penelitian, kurasi korpus, & benchmark ML
├── app.py                      # Aplikasi web produksi berbasis Streamlit
├── requirements.txt            # Daftar pustaka Python yang dibutuhkan
├── .gitignore                  # Aturan pengabaian berkas Git
└── README.md                   # Dokumentasi teknis proyek
```

---

## 🚀 Panduan Menjalankan Proyek Secara Lokal

### Prasyarat
- Python 3.10 atau lebih baru
- Git

### Langkah Instalasi

1. **Clone Repositori:**
   ```bash
   git clone https://github.com/noorafaelhakimfaudka-droid/NadaFana.git
   cd NadaFana
   ```

2. **Buat & Aktifkan Virtual Environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Pasang Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Web:**
   ```bash
   streamlit run app.py
   ```
   Buka peramban di `http://localhost:8501`.

---

## ☁️ Penerapan ke Cloud (Streamlit Community Cloud)

1. Fork atau push repositori ini ke akun GitHub Anda.
2. Buka [share.streamlit.io](https://share.streamlit.io/).
3. Hubungkan akun GitHub Anda dan pilih repositori `NadaFana`.
4. Atur:
   - **Main file path:** `app.py`
5. Klik **Deploy**! Berkas artefak biner (`df_lagu_indo.parquet`, `embeddings_lagu_indo.npy`, `index_lagu_indo.faiss`) berukuran ringan (< 5MB masing-masing) sehingga aplikasi dapat langsung berjalan tanpa perlu mengunduh data mentah.

---

## 🛠️ Tumpukan Teknologi

- **Model Bahasa & NLP:** [Hugging Face Transformers](https://huggingface.co/), [Sentence-Transformers](https://sbert.net/), [fastText](https://fasttext.cc/)
- **Pencarian Vektor:** [Facebook AI Similarity Search (FAISS)](https://github.com/facebookresearch/faiss)
- **Machine Learning & Statistik:** [scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Frontend / Aplikasi Web:** [Streamlit](https://streamlit.io/)
- **Penyimpanan Kolom:** Apache Parquet via [PyArrow](https://arrow.apache.org/)

---

## 👤 Pengembang

**Noorafael Hakim Faudka**
- GitHub: [@noorafaelhakimfaudka-droid](https://github.com/noorafaelhakimfaudka-droid)

---

## 📄 Lisensi

Didistribusikan di bawah lisensi MIT. Lihat berkas [LICENSE](LICENSE) untuk informasi lebih lanjut.

---

<div align="center">
  <b>NadaFana</b> — Dibuat dengan dedikasi untuk khazanah musik Indonesia.
</div>
