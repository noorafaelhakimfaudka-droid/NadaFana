"""
NadaFana — AI-Powered Indonesian Music Recommendation Engine
Ultra-polished Modern UI/UX Streamlit Interface
"""

import html
import os

# Batasi thread linear algebra untuk mencegah OpenBLAS / Windows memory paging exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Konfigurasi Halaman & Metadata
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NadaFana — AI Indonesian Music Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prioritaskan berkas artefak terbaru (*_indo.*), dengan fallback ke berkas lama jika ada
PATH_PARQUET = os.path.join(ARTIFACT_DIR, "df_lagu_indo.parquet")
if not os.path.exists(PATH_PARQUET):
    PATH_PARQUET = os.path.join(ARTIFACT_DIR, "df_lagu_mellow_indo.parquet")

PATH_EMBED = os.path.join(ARTIFACT_DIR, "embeddings_lagu_indo.npy")
if not os.path.exists(PATH_EMBED):
    PATH_EMBED = os.path.join(ARTIFACT_DIR, "embeddings_lagu_mellow.npy")

PATH_FAISS = os.path.join(ARTIFACT_DIR, "index_lagu_indo.faiss")
if not os.path.exists(PATH_FAISS):
    PATH_FAISS = os.path.join(ARTIFACT_DIR, "index_lagu_mellow.faiss")

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ----------------------------------------------------------------------------
# 2. Custom CSS: Designer Glassmorphism & Modern Typography
# ----------------------------------------------------------------------------
# PERHATIAN: Hindari `span { font-family: ... !important; }` karena merusak ikon Material Streamlit (_arrow_right)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Lora:ital,wght@1,400;1,600&display=swap');

/* ═══════════════════════════════════════════════════
   WARM ORANGE & WHITE — Psychological Comfort Theme
   Warna hangat (oranye/amber) meningkatkan rasa percaya diri,
   kenyamanan, dan kreativitas menurut psikologi warna.
   ═══════════════════════════════════════════════════ */

:root {
    --orange-50: #fff7ed;
    --orange-100: #ffedd5;
    --orange-200: #fed7aa;
    --orange-300: #fdba74;
    --orange-400: #fb923c;
    --orange-500: #f97316;
    --orange-600: #ea580c;
    --orange-700: #c2410c;
    --amber-400: #fbbf24;
    --amber-500: #f59e0b;
    --white: #ffffff;
    --warm-50: #fffbf5;
    --warm-100: #fef3e2;
    --warm-200: #fde6c4;
    --warm-gray-50: #fafaf9;
    --warm-gray-100: #f5f5f4;
    --warm-gray-200: #e7e5e4;
    --warm-gray-300: #d6d3d1;
    --warm-gray-400: #a8a29e;
    --warm-gray-500: #78716c;
    --warm-gray-600: #57534e;
    --warm-gray-700: #44403c;
    --warm-gray-800: #292524;
    --warm-gray-900: #1c1917;
    --text-primary: #1c1917;
    --text-secondary: #57534e;
    --text-muted: #78716c;
}

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(165deg, #fffbf5 0%, #ffffff 30%, #fff7ed 60%, #fef3e2 100%) !important;
    color: var(--text-primary);
}

h1, h2, h3, h4, p, label, .stMarkdown p {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

/* ─────── Floating Warm Orbs (Psychological Ambient Comfort) ─────── */
.stApp::before, .stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(80px);
    opacity: 0.35;
}
.stApp::before {
    width: 500px; height: 500px;
    top: -100px; right: -80px;
    background: radial-gradient(circle, rgba(251, 146, 60, 0.3) 0%, transparent 70%);
    animation: floatOrb1 18s ease-in-out infinite;
}
.stApp::after {
    width: 400px; height: 400px;
    bottom: -50px; left: -60px;
    background: radial-gradient(circle, rgba(245, 158, 11, 0.2) 0%, transparent 70%);
    animation: floatOrb2 22s ease-in-out infinite;
}
@keyframes floatOrb1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(-30px, 40px) scale(1.1); }
}
@keyframes floatOrb2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(25px, -30px) scale(1.08); }
}

/* ─────── Hero Badge & Header ─────── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 9999px;
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(245, 158, 11, 0.08) 100%);
    border: 1px solid rgba(249, 115, 22, 0.25);
    color: var(--orange-600);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
    animation: fadeSlideIn 0.6s ease-out;
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--orange-600) 0%, var(--orange-500) 40%, var(--amber-500) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    line-height: 1.12;
    animation: fadeSlideIn 0.8s ease-out;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--warm-gray-500);
    max-width: 780px;
    line-height: 1.7;
    margin-bottom: 24px;
    animation: fadeSlideIn 1s ease-out;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ─────── Glassmorphic Cards (Warm) ─────── */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(20px) saturate(1.3);
    -webkit-backdrop-filter: blur(20px) saturate(1.3);
    border: 1px solid rgba(249, 115, 22, 0.12);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 24px -4px rgba(249, 115, 22, 0.08),
                0 1px 3px rgba(0, 0, 0, 0.04);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--orange-400), var(--amber-400), var(--orange-500));
    opacity: 0;
    transition: opacity 0.35s ease;
}
.glass-card:hover {
    border-color: rgba(249, 115, 22, 0.3);
    box-shadow: 0 16px 48px -8px rgba(249, 115, 22, 0.12),
                0 4px 12px rgba(0, 0, 0, 0.06);
    transform: translateY(-3px);
}
.glass-card:hover::before {
    opacity: 1;
}

/* ─────── Reference DNA Card ─────── */
.reference-card {
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.9) 0%, rgba(255, 255, 255, 0.85) 100%);
    border: 1.5px solid rgba(249, 115, 22, 0.2);
    border-radius: 18px;
    padding: 22px;
    margin-top: 14px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -4px rgba(249, 115, 22, 0.08);
}


/* ─────── Match Score Pill ─────── */
.match-badge {
    display: inline-flex;
    align-items: center;
    padding: 7px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
.match-high {
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.12) 0%, rgba(245, 158, 11, 0.1) 100%);
    border: 1.5px solid rgba(249, 115, 22, 0.35);
    color: var(--orange-600);
}
.match-mid {
    background: rgba(245, 158, 11, 0.1);
    border: 1.5px solid rgba(245, 158, 11, 0.3);
    color: var(--amber-500);
}

/* ─────── Rank Tag ─────── */
.rank-badge {
    background: linear-gradient(135deg, var(--orange-500) 0%, var(--amber-500) 100%);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.73rem;
    padding: 4px 10px;
    border-radius: 8px;
    letter-spacing: 0.06em;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(249, 115, 22, 0.25);
}

/* ─────── Track Title & Artist ─────── */
.track-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--warm-gray-900);
    margin-top: 5px;
    margin-bottom: 2px;
}
.track-artist {
    font-size: 0.95rem;
    color: var(--orange-600);
    font-weight: 600;
}

/* ─────── Spotify Button ─────── */
.spotify-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, var(--orange-500) 0%, var(--orange-600) 100%);
    color: #ffffff !important;
    text-decoration: none !important;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 9px 18px;
    border-radius: 9999px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 2px 8px rgba(249, 115, 22, 0.2);
}
.spotify-btn:hover {
    background: linear-gradient(135deg, var(--orange-600) 0%, var(--orange-700) 100%);
    transform: scale(1.04) translateY(-1px);
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.3);
}

/* ─────── Lyric Quote Box ─────── */
.lyric-box {
    font-family: 'Lora', serif;
    font-style: italic;
    background: linear-gradient(135deg, rgba(255, 247, 237, 0.6) 0%, rgba(255, 255, 255, 0.4) 100%);
    border-left: 3px solid var(--orange-400);
    padding: 14px 20px;
    border-radius: 0 12px 12px 0;
    color: var(--warm-gray-600);
    font-size: 0.92rem;
    line-height: 1.7;
    margin-top: 14px;
    white-space: pre-wrap;
    word-break: break-word;
}

.lyric-full-container {
    background: rgba(255, 247, 237, 0.5);
    border: 1px solid rgba(249, 115, 22, 0.1);
    border-radius: 14px;
    padding: 18px 22px;
    font-family: 'Lora', serif;
    font-size: 0.92rem;
    line-height: 1.9;
    color: var(--warm-gray-600);
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-line;
    margin-top: 8px;
}

/* ─────── Stat Grid ─────── */
.stat-card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(249, 115, 22, 0.1);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    transition: all 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.08);
}
.stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--orange-600);
}
.stat-sub {
    font-size: 0.73rem;
    color: var(--warm-gray-500);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* ─────── Streamlit Tabs Customization ─────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid rgba(249, 115, 22, 0.1);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 12px 12px 0 0;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--warm-gray-500);
    background-color: transparent;
    border: none;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--orange-600);
    background: rgba(249, 115, 22, 0.05);
}
.stTabs [aria-selected="true"] {
    color: var(--orange-600) !important;
    background: rgba(249, 115, 22, 0.08) !important;
    border-bottom: 3px solid var(--orange-500) !important;
    font-weight: 700 !important;
}

/* ─────── Streamlit Button Restyle ─────── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1.5px solid rgba(249, 115, 22, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--orange-500) 0%, var(--orange-600) 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--orange-600) 0%, var(--orange-700) 100%) !important;
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.3) !important;
}

/* ─────── Sidebar (Warm Dark) ─────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1917 0%, #292524 100%) !important;
    border-right: 1px solid rgba(249, 115, 22, 0.15);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] li {
    color: #fafaf9 !important;
}
section[data-testid="stSidebar"] .stMarkdown strong {
    color: var(--orange-300) !important;
}

/* ─────── Streamlit Selectbox, Slider & Input ─────── */
.stSelectbox label, .stSlider label, .stTextArea label {
    color: var(--warm-gray-700) !important;
    font-weight: 600 !important;
}

/* ─────── Expander ─────── */
.streamlit-expanderHeader {
    color: var(--warm-gray-700) !important;
    font-weight: 600 !important;
}

/* ─────── Scrollbar ─────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--orange-300); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--orange-400); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. Data & Artifact Loader (Cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan koleksi musik...")
def _load_artifacts_cached(parquet_mtime, embed_mtime):
    missing = [p for p in (PATH_PARQUET, PATH_EMBED) if not os.path.exists(p)]
    if missing:
        return None, None, missing

    df_indo = pd.read_parquet(PATH_PARQUET)
    df_indo["artists"] = df_indo["artists"].fillna("Artis Tidak Diketahui")
    embeddings = np.load(PATH_EMBED)
    return df_indo, embeddings, []


def load_artifacts():
    p_mtime = os.path.getmtime(PATH_PARQUET) if os.path.exists(PATH_PARQUET) else 0
    e_mtime = os.path.getmtime(PATH_EMBED) if os.path.exists(PATH_EMBED) else 0
    return _load_artifacts_cached(p_mtime, e_mtime)


@st.cache_resource(show_spinner="Menyiapkan pencarian teks...")
def load_tfidf_matcher(df_indo):
    """Fallback semantic matcher berkecepatan tinggi & ramah memori jika PyTorch/Transformer overload."""
    corpus = df_indo["lirik_bersih"].fillna(df_indo["lyrics"].astype(str)).tolist()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


@st.cache_resource(show_spinner="Menyiapkan kecerdasan bahasa...")
def load_embed_model():
    """Memuat transformer dengan penanganan memori aman untuk Windows."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            EMBED_MODEL_NAME,
            model_kwargs={"low_cpu_mem_usage": False}
        )
        return model
    except Exception:
        return None


def format_score(skor):
    pct = max(0.0, min(100.0, skor * 100))
    return f"{pct:.1f}%"


def rekomendasikan_dari_index(df_indo, embeddings, idx_lagu, jumlah=5, bobot_audio=0.6):
    """
    Rekomendasi hybrid cerdas:
    Memadukan kecocokan irama & nuansa audio Spotify (valence, acousticness, energy, danceability)
    dengan resonansi semantik lirik (MiniLM embeddings).
    """
    audio_cols = ["valence", "acousticness", "energy", "danceability"]
    audio_matrix = df_indo[audio_cols].values.astype(np.float32)
    target_audio = audio_matrix[idx_lagu]

    # Kemiripan Audio DNA: Jarak Euclidean dinormalisasi (jarak maksimum pada kubus 4D adalah sqrt(4) = 2.0)
    dist_audio = np.linalg.norm(audio_matrix - target_audio, axis=1)
    sim_audio = np.clip(1.0 - (dist_audio / 2.0), 0.0, 1.0)

    # Kemiripan Lirik Semantik (Cosine Similarity)
    q_vec = embeddings[idx_lagu]
    sim_lirik = np.clip(np.dot(embeddings, q_vec), 0.0, 1.0)

    # Perpaduan Berbobot
    bobot_lirik = 1.0 - bobot_audio
    skor_hybrid = (bobot_audio * sim_audio) + (bobot_lirik * sim_lirik)

    # Kecualikan lagu acuan itu sendiri
    skor_hybrid[idx_lagu] = -1.0

    top_indices = np.argsort(skor_hybrid)[::-1][:jumlah]
    hasil = [
        (int(idx), float(skor_hybrid[idx]), float(sim_audio[idx]), float(sim_lirik[idx]))
        for idx in top_indices
    ]
    return hasil


def rekomendasikan_dari_teks(df_indo, embeddings, model_embed, teks_query, jumlah=5, vibe_filter="Semua Nuansa & Genre"):
    """
    Rekomendasi dari teks suasana hati dengan opsi penyelarasan preferensi nuansa audio.
    """
    vibe_profiles = {
        "Paling Melankolis & Sedih": np.array([0.18, 0.70, 0.25, 0.35], dtype=np.float32),
        "Sangat Hening & Akustik": np.array([0.28, 0.90, 0.18, 0.30], dtype=np.float32),
        "Santai & Hangat (Mid-Tempo)": np.array([0.45, 0.50, 0.40, 0.55], dtype=np.float32),
        "Energik & Penuh Semangat": np.array([0.70, 0.20, 0.80, 0.65], dtype=np.float32),
        "Irama Dansa & Ceria": np.array([0.75, 0.15, 0.70, 0.85], dtype=np.float32),
    }
    target_audio = vibe_profiles.get(vibe_filter)
    audio_cols = ["valence", "acousticness", "energy", "danceability"]
    audio_matrix = df_indo[audio_cols].values.astype(np.float32)

    sim_lirik = None
    engine_type = "tfidf"

    # Coba gunakan Transformer Embedding jika model tersedia
    if model_embed is not None:
        try:
            query_vec = model_embed.encode([teks_query], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            sim_lirik = np.clip(np.dot(embeddings, query_vec.T).flatten(), 0.0, 1.0)
            engine_type = "neural"
        except Exception:
            pass

    # Fallback aman dan cepat menggunakan TF-IDF semantic matching
    if sim_lirik is None:
        vectorizer, tfidf_matrix = load_tfidf_matcher(df_indo)
        q_vec = vectorizer.transform([teks_query])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        sim_lirik = np.clip(sims, 0.0, 1.0)
        engine_type = "tfidf"

    if target_audio is not None:
        dist_audio = np.linalg.norm(audio_matrix - target_audio, axis=1)
        sim_audio = np.clip(1.0 - (dist_audio / 2.0), 0.0, 1.0)
        # Padukan 50% makna lirik + 50% karakter audio yang dipilih
        skor_total = (0.5 * sim_lirik) + (0.5 * sim_audio)
        top_indices = np.argsort(skor_total)[::-1][:jumlah]
        hasil = [
            (int(idx), float(skor_total[idx]), float(sim_audio[idx]), float(sim_lirik[idx]))
            for idx in top_indices
        ]
    else:
        top_indices = np.argsort(sim_lirik)[::-1][:jumlah]
        hasil = [(int(idx), float(sim_lirik[idx])) for idx in top_indices if sim_lirik[idx] > 0.0]
        if not hasil:
            top_fallback = df_indo.sort_values(by="valence", ascending=True).head(jumlah).index
            hasil = [(df_indo.index.get_loc(idx_val), 0.65) for idx_val in top_fallback]

    return hasil, engine_type


# ----------------------------------------------------------------------------
# 4. UI Component Helpers
# ----------------------------------------------------------------------------


def render_spotify_embed(track_id):
    if not track_id or pd.isna(track_id):
        return
    embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
    iframe_code = f"""
    <iframe style="border-radius:12px; margin-top: 10px;" 
            src="{embed_url}" 
            width="100%" 
            height="80" 
            frameBorder="0" 
            allowfullscreen="" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy">
    </iframe>
    """
    components.html(iframe_code, height=96)


def format_lyric_preview(lirik_raw, max_chars=220):
    """Membersihkan teks lirik agar terhindar dari error parsing dan enak dibaca."""
    if not lirik_raw or pd.isna(lirik_raw):
        return "Lirik belum tersedia untuk lagu ini."

    clean = str(lirik_raw).replace("\r", "").strip()
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if not lines:
        return "Lirik belum tersedia untuk lagu ini."

    # Ambil baris-baris pertama sebagai cuplikan
    preview_snippet = " / ".join(lines[:4])
    if len(preview_snippet) > max_chars:
        preview_snippet = preview_snippet[:max_chars] + "..."
    return preview_snippet


def render_hasil_cards(df_indo, hasil, show_player=True):
    if not hasil:
        st.info("Belum ada rekomendasi yang sesuai.")
        return

    for peringkat, item in enumerate(hasil, start=1):
        idx, skor = item[0], item[1]

        row = df_indo.iloc[idx]
        pct_str = format_score(skor)
        badge_class = "match-high" if skor >= 0.70 else "match-mid"
        track_id = row.get("id", "")
        spotify_url = f"https://open.spotify.com/track/{track_id}" if track_id else "#"

        clean_title = html.escape(str(row.get("name", "Unknown Title")))
        clean_artists = html.escape(str(row.get("artists", "Unknown Artist")))

        lirik_raw = str(row.get("lyrics", "")).strip()
        lirik_preview = html.escape(format_lyric_preview(lirik_raw))

        # PENTING: Jangan gunakan indentasi 4+ spasi di dalam string HTML markdown
        # agar parser markdown CommonMark tidak mengubahnya menjadi tag <pre><code>!
        card_html = (
            '<div class="glass-card">\n'
            '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">\n'
            '<div>\n'
            f'<span class="rank-badge">#{peringkat}</span>\n'
            f'<div class="track-title">{clean_title}</div>\n'
            f'<div class="track-artist">oleh {clean_artists}</div>\n'
            '</div>\n'
            '<div style="display: flex; align-items: center; gap: 12px;">\n'
            f'<div class="match-badge {badge_class}">{pct_str} Cocok</div>\n'
            f'<a href="{spotify_url}" target="_blank" class="spotify-btn"><span>▶</span> Buka Spotify</a>\n'
            '</div>\n'
            '</div>\n'
            f'<div class="lyric-box">"{lirik_preview}"</div>\n'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Expander untuk membaca lirik lengkap tanpa resiko formatting error
        with st.expander(f"Baca Lirik Lengkap: {clean_title}"):
            if lirik_raw:
                st.markdown(
                    f'<div class="lyric-full-container">{html.escape(lirik_raw)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Lirik lengkap belum tercatat di dalam basis data.")

        if show_player and track_id:
            with st.expander(f"Putar Cuplikan Player: {clean_title}"):
                render_spotify_embed(track_id)


# ----------------------------------------------------------------------------
# 5. Header / Hero Section
# ----------------------------------------------------------------------------
col_hero_1, col_hero_2 = st.columns([3.5, 1.5])
with col_hero_1:
    st.markdown('<div class="hero-badge">Temukan Musik yang Memahami Perasaanmu</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">NadaFana</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Jelajahi ribuan lagu Indonesia dari berbagai genre — '
        'cukup pilih lagu favorit atau ceritakan perasaanmu, '
        'dan kami akan mencarikan lagu-lagu yang paling cocok dengan selera musikmu.'
        '</p>',
        unsafe_allow_html=True,
    )

df_indo, embeddings, missing = load_artifacts()

if missing:
    st.error(
        "Data musik belum tersedia. "
        "Pastikan semua file data sudah berada di folder yang benar:"
    )
    for m in missing:
        st.code(os.path.basename(m))
    st.stop()

with col_hero_2:
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
            <div class="stat-card">
                <div class="stat-val">{len(df_indo):,}</div>
                <div class="stat-sub">Lagu Indonesia</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">AI</div>
                <div class="stat-sub">Cerdas</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">Instan</div>
                <div class="stat-sub">Pencarian</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">Lokal</div>
                <div class="stat-sub">Semua Genre</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. Tab Navigation: 3 Seamless User Journeys
# ----------------------------------------------------------------------------
tab_pilih, tab_teks, tab_katalog = st.tabs([
    "Berdasarkan Lagu Favorit",
    "Berdasarkan Mood & Cerita",
    "Jelajahi Koleksi Musik",
])

# ============================================================================
# TAB 1: Cari dari Lagu Referensi
# ============================================================================
with tab_pilih:
    st.markdown(
        "<p style='color:#78716c; margin-top: 6px;'>Pilih salah satu lagu favoritmu, lalu kami carikan lagu-lagu lain yang punya nuansa dan rasa serupa.</p>",
        unsafe_allow_html=True,
    )

    opsi_lagu = (df_indo["name"] + " — " + df_indo["artists"].astype(str)).tolist()

    col_select, col_slider = st.columns([3, 1])
    with col_select:
        pilihan = st.selectbox(
            "Pilih lagu favoritmu:",
            opsi_lagu,
            index=0,
            help="Ketik judul lagu atau nama musisi untuk memfilter daftar.",
        )
    with col_slider:
        jumlah_rekomendasi = st.slider("Jumlah lagu:", 1, 10, 5, key="slider_pilih")

    # Slider Kontrol Keseimbangan Hybrid (Nuansa Irama vs Makna Lirik)
    col_bobot, col_info = st.columns([2.2, 1.8])
    with col_bobot:
        bobot_audio_pct = st.slider(
            "Atur prioritas pencarian:",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Geser ke kiri untuk mencari lagu dengan cerita/lirik serupa. Geser ke kanan untuk mencari lagu dengan irama & nuansa musik serupa.",
            key="slider_bobot_audio",
        )
    with col_info:
        bobot_audio = bobot_audio_pct / 100.0
        bobot_lirik_pct = 100 - bobot_audio_pct
        st.markdown(
            f"<div style='background: #fff7ed; border: 1px solid #fed7aa; border-radius: 12px; padding: 10px 14px; margin-top: 18px; font-size: 0.82rem; color: #78716c;'>"
            f"<b>Irama Serupa:</b> <span style='color:#ea580c; font-weight:700;'>{bobot_audio_pct}%</span> &nbsp;|&nbsp; "
            f"<b>Cerita Serupa:</b> <span style='color:#f97316; font-weight:700;'>{bobot_lirik_pct}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    idx_lagu = opsi_lagu.index(pilihan)
    row_ref = df_indo.iloc[idx_lagu]

    # Reference Song DNA Preview Card
    with st.container():
        clean_ref_name = html.escape(str(row_ref['name']))
        clean_ref_art = html.escape(str(row_ref['artists']))
        ref_track_id = row_ref.get("id", "")

        ref_card_html = (
            '<div class="reference-card">\n'
            '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">\n'
            '<div>\n'
            '<span style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #ea580c; font-weight: 700;">Lagu Pilihanmu</span>\n'
            f'<div style="font-size: 1.35rem; font-weight: 700; color: #1c1917; margin-top: 2px;">{clean_ref_name}</div>\n'
            f'<div style="color: #78716c; font-size: 0.95rem;">{clean_ref_art}</div>\n'
            '</div>\n'
            '<div>\n'
            f'<a href="https://open.spotify.com/track/{ref_track_id}" target="_blank" class="spotify-btn">Dengarkan di Spotify ↗</a>\n'
            '</div>\n'
            '</div>\n'
            '</div>'
        )
        st.markdown(ref_card_html, unsafe_allow_html=True)

    col_btn, col_chk = st.columns([1, 2])
    with col_btn:
        btn_rekomendasi = st.button("Temukan Lagu Serupa", key="btn_pilih", type="primary", use_container_width=True)
    with col_chk:
        show_player_pilih = st.toggle("Tampilkan pemutar musik", value=True, key="toggle_player_pilih")

    if btn_rekomendasi:
        with st.spinner("Mencari lagu yang cocok untukmu..."):
            hasil = rekomendasikan_dari_index(
                df_indo, embeddings, idx_lagu, jumlah_rekomendasi, bobot_audio=bobot_audio
            )
            st.session_state["hasil_tab1"] = hasil
            st.session_state["ref_name_tab1"] = row_ref['name']

    if "hasil_tab1" in st.session_state:
        st.markdown(f"#### Lagu yang cocok dengan *{st.session_state['ref_name_tab1']}*:")
        render_hasil_cards(df_indo, st.session_state["hasil_tab1"], show_player=show_player_pilih)

# ============================================================================
# TAB 2: Cari dari Mood / Lirik Bebas
# ============================================================================
with tab_teks:
    st.markdown(
        "<p style='color:#78716c; margin-top: 6px;'>Tuliskan perasaanmu, suasana hatimu, atau cerita yang sedang kamu rasakan — kami akan mencarikan lagu yang paling cocok.</p>",
        unsafe_allow_html=True,
    )

    # Inisialisasi state untuk input teks mood
    if "teks_query_val" not in st.session_state:
        st.session_state.teks_query_val = ""

    def apply_preset(teks):
        st.session_state.teks_query_val = teks

    # Preset Quick Mood Chips
    st.markdown("<span style='font-size: 0.8rem; font-weight: 700; color: #78716c; text-transform: uppercase; letter-spacing: 0.05em;'>Inspirasi Mood Cepat:</span>", unsafe_allow_html=True)
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    with preset_col1:
        st.button("Hujan & Kenangan Masa Lalu", use_container_width=True, on_click=apply_preset, args=("Hujan sore yang dingin mengingatkanku pada kenangan masa lalu yang tak kembali",))
        st.button("Kopi Senja Menatap Jendela", use_container_width=True, on_click=apply_preset, args=("Duduk sendiri di beranda menikmati kopi saat senja sambil merenungi jalan hidup",))

    with preset_col2:
        st.button("Patah Hati yang Sunyi", use_container_width=True, on_click=apply_preset, args=("Rasa sakit karena harus berpisah dalam diam tanpa sempat berpamitan",))
        st.button("Melamun Larut Malam", use_container_width=True, on_click=apply_preset, args=("Insomnia tengah malam menatap langit-langit kamar memikirkan masa depan",))

    with preset_col3:
        st.button("Berdamai dengan Kehilangan", use_container_width=True, on_click=apply_preset, args=("Mencoba ikhlas melepaskan orang yang sangat kucintai agar ia bahagia",))
        st.button("Rindu Rumah & Dekapan", use_container_width=True, on_click=apply_preset, args=("Rindu suasana hangat rumah dan pelukan orang-orang terkasih di perantauan",))

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col_input, col_ctrl = st.columns([2.5, 1.5])
    with col_input:
        teks_query = st.text_area(
            "Ceritakan perasaanmu:",
            value=st.session_state.teks_query_val,
            placeholder="Contoh: 'rindu yang tak pernah usai' atau 'hujan sore yang membuatku teringat masa lalu'...",
            height=130,
            key="input_area_mood",
        )
    with col_ctrl:
        vibe_filter = st.selectbox(
            "Karakter musik:",
            [
                "Semua Nuansa & Genre",
                "Paling Melankolis & Sedih",
                "Sangat Hening & Akustik",
                "Santai & Hangat (Mid-Tempo)",
                "Energik & Penuh Semangat",
                "Irama Dansa & Ceria",
            ],
            key="select_vibe_teks",
            help="Pilih karakter musik yang kamu inginkan untuk hasil pencarian.",
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            jumlah_rekomendasi_teks = st.slider("Jumlah lagu:", 1, 10, 5, key="slider_teks")
        with col_c2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            show_player_teks = st.toggle("Pemutar musik", value=True, key="toggle_player_teks")

    if st.button("Carikan Lagu Untukku", key="btn_teks", type="primary", use_container_width=True):
        if not teks_query.strip():
            st.warning("Silakan ketikkan deskripsi suasana hati atau pilih salah satu inspirasi mood di atas.")
        else:
            with st.spinner("Mencari lagu yang cocok..."):
                model_embed = load_embed_model()
                hasil, engine_type = rekomendasikan_dari_teks(
                    df_indo, embeddings, model_embed, teks_query, jumlah_rekomendasi_teks, vibe_filter=vibe_filter
                )
                st.session_state["hasil_tab2"] = hasil
                st.session_state["query_tab2"] = teks_query

    if "hasil_tab2" in st.session_state:
        st.markdown(f"#### Lagu untuk ceritamu: *\"{st.session_state['query_tab2']}\"*")
        render_hasil_cards(df_indo, st.session_state["hasil_tab2"], show_player=show_player_teks)

# ============================================================================
# TAB 3: Jelajahi Koleksi Musik Indonesia
# ============================================================================
with tab_katalog:
    st.markdown(
        "<p style='color:#78716c; margin-top: 6px;'>Telusuri koleksi lagu Indonesia pilihan berdasarkan suasana — dari yang paling akustik hingga yang paling energik.</p>",
        unsafe_allow_html=True,
    )

    kategori = st.radio(
        "Pilih suasana:",
        [
            "Paling Akustik",
            "Paling Melankolis",
            "Paling Tenang",
            "Paling Energik",
            "Paling Dansa / Ceria",
        ],
        horizontal=True,
    )

    if "Paling Akustik" in kategori:
        df_curated = df_indo.sort_values(by="acousticness", ascending=False).head(6)
        desc = "Lagu-lagu dengan dominasi instrumen akustik alami — gitar akustik, piano lembut, cello."
    elif "Paling Melankolis" in kategori:
        df_curated = df_indo.sort_values(by="valence", ascending=True).head(6)
        desc = "Lagu-lagu dengan nuansa sendu dan melankolis yang paling menyentuh hati."
    elif "Paling Tenang" in kategori:
        df_curated = df_indo.sort_values(by="energy", ascending=True).head(6)
        desc = "Lagu-lagu bertempo hening dan lembut, menenangkan suasana hati."
    elif "Paling Energik" in kategori:
        df_curated = df_indo.sort_values(by="energy", ascending=False).head(6)
        desc = "Lagu-lagu dengan ketukan dinamis, bertenaga, dan bersemangat tinggi."
    else:
        df_curated = df_indo.sort_values(by=["danceability", "valence"], ascending=[False, False]).head(6)
        desc = "Lagu-lagu dengan irama dansa yang asik, ceria, dan penuh getaran positif."

    st.markdown(f"<div style='color: #ea580c; font-weight: 600; margin-bottom: 16px;'>{desc}</div>", unsafe_allow_html=True)

    curated_hasil = [(df_indo.index.get_loc(idx_val), 0.90 - i * 0.02) for i, idx_val in enumerate(df_curated.index)]
    render_hasil_cards(df_indo, curated_hasil, show_player=True)

# ----------------------------------------------------------------------------
# 7. Sidebar: Arsitektur & Kurasi
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0 20px 0;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 800; color: #ea580c; letter-spacing: 2px; margin-bottom: 4px;">NADAFANA</div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 600; color: #fafaf9;">Rekomendasi Musik Cerdas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Tentang NadaFana")
    st.markdown(
        """
        NadaFana menganalisis **ribuan lagu Indonesia dari berbagai genre** untuk mencarikan musik yang paling cocok dengan selera dan perasaanmu.

        Kami memadukan kemiripan **irama & nuansa musik** dengan **makna cerita di balik lirik** — sehingga rekomendasi yang kamu dapatkan benar-benar terasa personal.
        """
    )

    st.markdown("---")
    st.markdown("### Cara Menggunakan")
    st.markdown(
        """
        - **Berdasarkan Lagu** — Pilih lagu favorit, temukan lagu-lagu serupa.
        - **Berdasarkan Perasaan** — Ceritakan suasana hatimu dengan kata-kata.
        - **Jelajahi Koleksi** — Telusuri lagu Indonesia berdasarkan suasana tertentu.
        """
    )

    st.markdown("---")
    st.caption("Dibuat untuk pecinta musik Indonesia.")
