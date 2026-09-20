"""
NadaFana — AI-Powered Indonesian Music Recommendation Engine
Ultra-polished Modern UI/UX Streamlit Interface
"""

import base64
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
# 1. Konfigurasi Halaman & Metadata (Full-Canvas Netflix Style)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NadaFana — AI Indonesian Music Discovery",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
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

# Muat gambar latar belakang musik sinematik (Spotify Green Vinyl & Soundwaves)
bg_base64 = ""
path_bg_asset = os.path.join(ARTIFACT_DIR, "assets", "spotify_bg.jpg")
if os.path.exists(path_bg_asset):
    try:
        with open(path_bg_asset, "rb") as f_img:
            bg_base64 = base64.b64encode(f_img.read()).decode()
    except Exception:
        bg_base64 = ""

EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# ----------------------------------------------------------------------------
# 2. Custom CSS: Designer Glassmorphism & Modern Typography
# ----------------------------------------------------------------------------
# PERHATIAN: Hindari `span { font-family: ... !important; }` karena merusak ikon Material Streamlit (_arrow_right)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=Lora:ital,wght@1,400;1,600&display=swap');

/* ═══════════════════════════════════════════════════
   SPOTIFY x NETFLIX DESIGN SYSTEM
   Warna: 100% Autentik Spotify (#1DB954 Green, #121212 Dark)
   Bentuk: Netflix Card Layout (Poster Banner, Dynamic Tags, Hover Zoom)
   ═══════════════════════════════════════════════════ */

:root {
    --sp-green: #1DB954;
    --sp-green-bright: #1ED760;
    --sp-green-dark: #169c46;
    --sp-green-glow: rgba(29, 185, 84, 0.3);
    --sp-green-subtle: rgba(29, 185, 84, 0.12);
    --sp-black: #000000;
    --sp-bg: #121212;
    --sp-surface: #181818;
    --sp-surface-elevated: #242424;
    --sp-surface-hover: #282828;
    --sp-surface-active: #333333;
    --sp-white: #ffffff;
    --sp-gray-100: #e0e0e0;
    --sp-gray-200: #b3b3b3;
    --sp-gray-300: #a0a0a0;
    --sp-gray-400: #727272;
    --sp-gray-500: #535353;
    --sp-gray-600: #333333;
    --text-primary: #ffffff;
    --text-secondary: #b3b3b3;
    --text-muted: #727272;
}

html, body, .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #121212 !important;
    background-image: 
        radial-gradient(ellipse at 50% 0%, rgba(29, 185, 84, 0.22) 0%, transparent 60%),
        linear-gradient(180deg, rgba(14, 14, 14, 0.65) 0%, rgba(18, 18, 18, 0.88) 360px, rgba(18, 18, 18, 0.98) 720px, #121212 100%),
        url('__BG_IMAGE_URL__') !important;
    background-attachment: fixed !important;
    background-position: top center !important;
    background-size: cover !important;
    background-repeat: no-repeat !important;
    color: var(--text-primary);
}

h1, h2, h3, h4, p, label, .stMarkdown p {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
}

/* ─────── Floating Ambient Spotify Orbs ─────── */
.stApp::before, .stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(120px);
    opacity: 0.18;
}
.stApp::before {
    width: 550px; height: 550px;
    top: -120px; right: -100px;
    background: radial-gradient(circle, #1DB954 0%, transparent 70%);
    animation: floatOrb1 18s ease-in-out infinite;
}
.stApp::after {
    width: 450px; height: 450px;
    bottom: -80px; left: -80px;
    background: radial-gradient(circle, rgba(29, 185, 84, 0.4) 0%, transparent 70%);
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
    padding: 7px 18px;
    border-radius: 9999px;
    background: rgba(29, 185, 84, 0.1);
    border: 1px solid rgba(29, 185, 84, 0.35);
    color: #1DB954;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 12px;
    animation: fadeSlideIn 0.6s ease-out;
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 60%, #1DB954 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    line-height: 1.1;
    letter-spacing: -0.02em;
    animation: fadeSlideIn 0.8s ease-out;
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #b3b3b3;
    max-width: 600px;
    line-height: 1.5;
    margin-bottom: 12px;
    animation: fadeSlideIn 1s ease-out;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ─────── Reference DNA Card (Spotlight Featured) ─────── */
.reference-card {
    background: linear-gradient(135deg, #181818 0%, #121212 100%);
    border: 1.5px solid rgba(29, 185, 84, 0.35);
    border-radius: 18px;
    padding: 22px;
    margin-top: 14px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.6), 0 0 20px rgba(29, 185, 84, 0.1);
    position: relative;
    overflow: hidden;
}
.reference-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    background: #1DB954;
}

/* ─────── NETFLIX-STYLE MUSIC CARD ─────── */
.netflix-card {
    background: #181818;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s ease, border-color 0.3s ease, background 0.3s ease;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    position: relative;
}

.netflix-card:hover {
    transform: translateY(-6px) scale(1.015);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.8), 0 0 26px rgba(29, 185, 84, 0.3);
    border-color: rgba(29, 185, 84, 0.55);
    background: #202020;
}

/* Netflix Poster Banner Header */
.netflix-card-banner {
    height: 125px;
    position: relative;
    padding: 14px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    overflow: hidden;
}

.netflix-card-banner::after {
    content: '';
    position: absolute;
    right: -25px;
    bottom: -25px;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    border: 8px solid rgba(255, 255, 255, 0.04);
    box-shadow: 0 0 0 16px rgba(255, 255, 255, 0.02), 0 0 0 32px rgba(255, 255, 255, 0.01);
    pointer-events: none;
}

/* Rank Badge (Netflix TOP 10 style, Spotify Green) */
.netflix-rank-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(0, 0, 0, 0.85);
    border: 1.5px solid #1DB954;
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 4px 9px;
    border-radius: 6px;
    letter-spacing: 0.06em;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    z-index: 2;
}
.netflix-rank-badge span {
    color: #1DB954;
    font-weight: 900;
}

/* Match Percentage (Netflix Match style) */
.netflix-match-badge {
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(29, 185, 84, 0.45);
    color: #1DB954;
    font-weight: 800;
    font-size: 0.82rem;
    padding: 4px 12px;
    border-radius: 9999px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    letter-spacing: 0.02em;
    z-index: 2;
}

/* Circular Spotify Play Button inside Banner */
.netflix-play-btn {
    position: absolute;
    bottom: 12px;
    right: 14px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: #1DB954;
    color: #000000 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    text-decoration: none !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.6);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    z-index: 3;
}
.netflix-card:hover .netflix-play-btn {
    transform: scale(1.15);
    background: #1ED760;
    box-shadow: 0 0 22px rgba(30, 215, 96, 0.6);
}

/* Netflix Card Body */
.netflix-card-body {
    padding: 16px 18px 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.netflix-card-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.22rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.netflix-card-artist {
    font-size: 0.92rem;
    color: #b3b3b3;
    font-weight: 500;
    margin-top: -4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Audio Trait Chips (like Netflix tags: HD, 5.1, Sci-Fi) */
.netflix-tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 4px 0 6px 0;
}
.netflix-tag {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e0e0e0;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 0.02em;
}
.netflix-tag-spotify {
    background: rgba(29, 185, 84, 0.12);
    border: 1px solid rgba(29, 185, 84, 0.35);
    color: #1DB954;
    font-weight: 700;
}

/* Lyric Synopsis Quote (Netflix Overview Style) */
.netflix-lyric-box {
    font-family: 'Lora', serif;
    font-style: italic;
    background: rgba(0, 0, 0, 0.35);
    border-left: 3px solid #1DB954;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    color: #b3b3b3;
    font-size: 0.86rem;
    line-height: 1.6;
    margin-top: 4px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* Action Link Row */
.netflix-action-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.spotify-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1DB954;
    color: #000000 !important;
    text-decoration: none !important;
    font-weight: 700;
    font-size: 0.82rem;
    padding: 8px 18px;
    border-radius: 9999px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
.spotify-btn:hover {
    background: #1ED760;
    transform: scale(1.04) translateY(-1px);
    box-shadow: 0 4px 18px rgba(30, 215, 96, 0.5);
}

/* Full Lyrics Container inside expander */
.lyric-full-container {
    background: #141414;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'Lora', serif;
    font-size: 0.9rem;
    line-height: 1.85;
    color: #d4d4d4;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-line;
}

/* ─────── Stat Grid ─────── */
.stat-card {
    background: #181818;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    transition: all 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    background: #222222;
    border-color: rgba(29, 185, 84, 0.3);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 16px rgba(29, 185, 84, 0.15);
}
.stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1DB954;
}
.stat-sub {
    font-size: 0.73rem;
    color: #b3b3b3;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* ─────── Streamlit Tabs Customization ─────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 12px 12px 0 0;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #b3b3b3;
    background-color: transparent;
    border: none;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.05);
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: rgba(29, 185, 84, 0.08) !important;
    border-bottom: 3px solid #1DB954 !important;
    font-weight: 700 !important;
}

/* ─────── Streamlit Button Restyle (Authentic Spotify Pills) ─────── */
.stButton > button {
    border-radius: 9999px !important;
    font-weight: 700 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
    background: #181818 !important;
    color: #ffffff !important;
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.02) !important;
    border-color: #1DB954 !important;
    color: #1DB954 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), 0 0 12px rgba(29, 185, 84, 0.2) !important;
    background: #242424 !important;
}
.stButton > button[kind="primary"] {
    background: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(29, 185, 84, 0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1ED760 !important;
    color: #000000 !important;
    box-shadow: 0 8px 24px rgba(30, 215, 96, 0.55) !important;
    transform: scale(1.03) !important;
}

/* ─────── Sembunyikan Sidebar Sepenuhnya (Netflix Full-Width Canvas) ─────── */
section[data-testid="stSidebar"],
div[data-testid="collapsedControl"],
button[data-testid="baseButton-header"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Maksimalkan lebar kontainer utama */
.main .block-container {
    max-width: 1400px !important;
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ─────── Sleek Compact Spotify Topbar ─────── */
.spotify-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(24, 24, 24, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 14px;
    transition: border-color 0.25s ease;
}
.spotify-topbar:hover {
    border-color: rgba(29, 185, 84, 0.3);
}
.spotify-topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    line-height: 1;
}
.brand-name-green { color: #1DB954; }
.brand-name-white { color: #ffffff; }
.brand-sub {
    font-size: 0.84rem;
    color: #a0a0a0;
    font-weight: 500;
    border-left: 1px solid rgba(255, 255, 255, 0.14);
    padding-left: 12px;
}
.song-count-pill {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #b3b3b3;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 5px 13px;
    border-radius: 9999px;
}

/* ─────── Clean Audio Preference Box ─────── */
.pref-panel {
    background: #181818;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 16px 10px 16px;
    margin-bottom: 10px;
}
.pref-header {
    font-family: 'Outfit', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #ffffff;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.pref-label {
    font-size: 0.78rem;
    color: #b3b3b3;
    font-weight: 600;
    margin-bottom: 4px;
}


/* ─────── Streamlit Selectbox, Slider & Input ─────── */
.stSelectbox label, .stSlider label, .stTextArea label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ─────── Expander ─────── */
.streamlit-expanderHeader {
    background: #181818 !important;
    border-radius: 10px !important;
    color: #b3b3b3 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}
.streamlit-expanderHeader:hover {
    color: #1DB954 !important;
    border-color: rgba(29, 185, 84, 0.3) !important;
}
details[data-testid="stExpander"] {
    border: none !important;
    margin-bottom: 6px !important;
}

/* ─────── Animated Audio Equalizer Bars ─────── */
.eq-container {
    display: inline-flex;
    align-items: flex-end;
    gap: 3px;
    height: 15px;
    padding: 0 2px;
}
.eq-bar {
    width: 3px;
    background: #1DB954;
    border-radius: 2px;
    animation: eqBounce 1.2s ease-in-out infinite alternate;
}
.eq-bar:nth-child(1) { height: 45%; animation-delay: 0.1s; }
.eq-bar:nth-child(2) { height: 85%; animation-delay: 0.35s; }
.eq-bar:nth-child(3) { height: 60%; animation-delay: 0.2s; }
.eq-bar:nth-child(4) { height: 100%; animation-delay: 0.45s; }
@keyframes eqBounce {
    0% { height: 25%; }
    100% { height: 100%; }
}

/* ─────── Vertical Studio Rack / Tuning Console ─────── */
.vertical-console {
    background: linear-gradient(180deg, rgba(26, 26, 26, 0.94) 0%, rgba(16, 16, 16, 0.98) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 2.5px solid #1DB954;
    border-radius: 16px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.7), 0 0 20px rgba(29, 185, 84, 0.1);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 12px;
}
.vertical-console:hover {
    border-color: rgba(29, 185, 84, 0.45);
    box-shadow: 0 20px 42px rgba(0, 0, 0, 0.8), 0 0 28px rgba(29, 185, 84, 0.22);
}
.console-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.console-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #ffffff;
    text-transform: uppercase;
}

/* Live Dynamic Level Meter */
.meter-panel {
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 6px 0 10px 0;
}
.meter-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.meter-label { color: #b3b3b3; }
.meter-value { color: #1DB954; }
.meter-track {
    width: 100%;
    height: 5px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    overflow: hidden;
}
.meter-fill-lyrics {
    height: 100%;
    background: linear-gradient(90deg, #1DB954 0%, #1ED760 100%);
    border-radius: 9999px;
    transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 8px rgba(29, 185, 84, 0.5);
}
.meter-fill-audio {
    height: 100%;
    background: linear-gradient(90deg, #169c46 0%, #1DB954 100%);
    border-radius: 9999px;
    transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 8px rgba(29, 185, 84, 0.5);
}
.output-counter-badge {
    display: inline-block;
    background: rgba(29, 185, 84, 0.14);
    border: 1px solid rgba(29, 185, 84, 0.4);
    color: #1DB954;
    font-size: 0.74rem;
    font-weight: 800;
    padding: 3px 10px;
    border-radius: 9999px;
    letter-spacing: 0.04em;
    margin-top: 4px;
}

/* Netflix Card Entrance Animation */
.netflix-card {
    animation: netflixFadeIn 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes netflixFadeIn {
    from {
        opacity: 0;
        transform: translateY(14px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ─────── Scrollbar ─────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333333; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #1DB954; }
</style>
"""
bg_url = f"data:image/jpeg;base64,{bg_base64}" if bg_base64 else ""
st.markdown(CUSTOM_CSS.replace("__BG_IMAGE_URL__", bg_url), unsafe_allow_html=True)


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


def rekomendasikan_terpadu(
    df_indo,
    embeddings,
    idx_lagu,
    teks_query="",
    model_embed=None,
    jumlah=6,
    bobot_audio=0.6,
):
    """
    Fitur Utama NadaFana:
    Memadukan DNA audio lagu kesukaan dengan resonansi semantik lirik dan curahan perasaan pengguna.
    """
    audio_cols = ["valence", "acousticness", "energy", "danceability"]
    audio_matrix = df_indo[audio_cols].values.astype(np.float32)
    target_audio = audio_matrix[idx_lagu]

    dist_audio = np.linalg.norm(audio_matrix - target_audio, axis=1)
    sim_audio = np.clip(1.0 - (dist_audio / 2.0), 0.0, 1.0)

    q_vec_lagu = embeddings[idx_lagu]
    sim_lirik_lagu = np.clip(np.dot(embeddings, q_vec_lagu), 0.0, 1.0)

    teks_bersih = teks_query.strip() if teks_query else ""
    if teks_bersih:
        sim_cerita = None
        if model_embed is not None:
            try:
                query_vec = model_embed.encode([teks_bersih], convert_to_numpy=True)
                query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
                sim_cerita = np.clip(np.dot(embeddings, query_vec.T).flatten(), 0.0, 1.0)
            except Exception:
                pass

        if sim_cerita is None:
            vectorizer, tfidf_matrix = load_tfidf_matcher(df_indo)
            q_vec_t = vectorizer.transform([teks_bersih])
            sims = cosine_similarity(q_vec_t, tfidf_matrix)[0]
            sim_cerita = np.clip(sims, 0.0, 1.0)

        sim_semantik = 0.5 * sim_lirik_lagu + 0.5 * sim_cerita
    else:
        sim_semantik = sim_lirik_lagu

    bobot_lirik = 1.0 - bobot_audio
    skor_total = (bobot_audio * sim_audio) + (bobot_lirik * sim_semantik)
    skor_total[idx_lagu] = -1.0

    top_indices = np.argsort(skor_total)[::-1][:jumlah]
    hasil = [
        (int(idx), float(skor_total[idx]), float(sim_audio[idx]), float(sim_semantik[idx]))
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


def render_hasil_cards(df_indo, hasil, show_player=True):
    if not hasil:
        st.info("Belum ada rekomendasi yang sesuai.")
        return

    # Palet gradien gelap ambient khas Spotify & Netflix poster
    BANNER_GRADIENTS = [
        "linear-gradient(135deg, #0d381e 0%, #151515 100%)",
        "linear-gradient(135deg, #134e2c 0%, #0e2016 100%)",
        "linear-gradient(135deg, #0a331f 0%, #181818 100%)",
        "linear-gradient(135deg, #165b32 0%, #102619 100%)",
        "linear-gradient(135deg, #0e3d23 0%, #161616 100%)",
        "linear-gradient(135deg, #1a4d2e 0%, #0e1d15 100%)",
    ]

    # Grid 2 Kolom Netflix Card - Bersih, Visual, Minim Teks
    for chunk_start in range(0, len(hasil), 2):
        chunk = hasil[chunk_start : chunk_start + 2]
        cols = st.columns(2)
        for col_idx, item in enumerate(chunk):
            peringkat = chunk_start + col_idx + 1
            idx, skor = item[0], item[1]

            row = df_indo.iloc[idx]
            pct_str = format_score(skor)
            track_id = row.get("id", "")
            spotify_url = f"https://open.spotify.com/track/{track_id}" if track_id else "#"

            clean_title = html.escape(str(row.get("name", "Unknown Title")))
            clean_artists = html.escape(str(row.get("artists", "Unknown Artist")))
            lirik_raw = str(row.get("lyrics", "")).strip()

            # Maksimal 2 tag audio ringkas ala Netflix
            tags_html_parts = []
            try:
                ac = float(row.get("acousticness", 0.5))
                va = float(row.get("valence", 0.5))
                en = float(row.get("energy", 0.5))

                if ac >= 0.55:
                    tags_html_parts.append('<span class="netflix-tag">Akustik</span>')
                elif en >= 0.65:
                    tags_html_parts.append('<span class="netflix-tag">Energik</span>')
                elif en <= 0.35:
                    tags_html_parts.append('<span class="netflix-tag">Mellow</span>')

                if va >= 0.60:
                    tags_html_parts.append('<span class="netflix-tag">Ceria</span>')
                elif va <= 0.35:
                    tags_html_parts.append('<span class="netflix-tag">Melankolis</span>')
                else:
                    tags_html_parts.append('<span class="netflix-tag">Syahdu</span>')
            except Exception:
                pass

            tags_rendered = "".join(tags_html_parts[:2])
            banner_bg = BANNER_GRADIENTS[(peringkat - 1) % len(BANNER_GRADIENTS)]

            card_html = (
                f'<div class="netflix-card">\n'
                f'<div class="netflix-card-banner" style="background: {banner_bg};">\n'
                f'<div class="netflix-rank-badge">#{peringkat}</div>\n'
                f'<div class="netflix-match-badge">{pct_str} Cocok</div>\n'
                '<svg style="position: absolute; right: 18px; top: 16px; opacity: 0.12; pointer-events: none;" width="76" height="76" viewBox="0 0 24 24" fill="#1DB954">\n'
                '<path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z"/>\n'
                '</svg>\n'
                f'<a href="{spotify_url}" target="_blank" class="netflix-play-btn" title="Dengarkan di Spotify">\n'
                '<svg width="14" height="14" viewBox="0 0 24 24" fill="#000000" style="margin-left: 2px;"><polygon points="6,4 20,12 6,20"/></svg>\n'
                '</a>\n'
                '</div>\n'
                '<div class="netflix-card-body">\n'
                f'<div class="netflix-card-title" title="{clean_title}">{clean_title}</div>\n'
                f'<div class="netflix-card-artist" title="{clean_artists}">{clean_artists}</div>\n'
                f'<div class="netflix-tags-row">{tags_rendered}</div>\n'
                '<div class="netflix-action-row">\n'
                f'<a href="{spotify_url}" target="_blank" class="spotify-btn"><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: -1px; margin-right: 5px;"><polygon points="6,4 20,12 6,20"/></svg>Buka di Spotify</a>\n'
                '</div>\n'
                '</div>\n'
                '</div>'
            )

            with cols[col_idx]:
                st.markdown(card_html, unsafe_allow_html=True)
                with st.expander(f"Lirik & Audio: {clean_title}"):
                    if show_player and track_id:
                        render_spotify_embed(track_id)
                    if lirik_raw:
                        st.markdown(
                            f'<div class="lyric-full-container">{html.escape(lirik_raw)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Lirik lengkap belum tercatat di dalam basis data.")


# ----------------------------------------------------------------------------
# 5. Netflix x Spotify Cinematic Hero Billboard
# ----------------------------------------------------------------------------
df_indo, embeddings, missing = load_artifacts()

if missing:
    st.error(
        "Data musik belum tersedia. "
        "Pastikan semua file data sudah berada di folder yang benar:"
    )
    for m in missing:
        st.code(os.path.basename(m))
    st.stop()

topbar_html = f"""<div class="spotify-topbar">
<div class="spotify-topbar-brand">
<span class="brand-name"><span class="brand-name-green">NADA</span><span class="brand-name-white">FANA</span></span>
<span class="brand-sub">Rekomendasi Musik Indonesia</span>
</div>
<div class="spotify-topbar-meta">
<span class="song-count-pill">{len(df_indo):,} lagu</span>
</div>
</div>"""
st.markdown(topbar_html, unsafe_allow_html=True)

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6. Flagship Discovery Engine: Lagu Kesukaan & Curahan Hati (1 Fitur Utama)
# ----------------------------------------------------------------------------
opsi_lagu = [
    f"{name} — {artist}"
    for name, artist in zip(df_indo["name"], df_indo["artists"].astype(str))
]

if "input_cerita_utama" not in st.session_state:
    st.session_state["input_cerita_utama"] = ""

def apply_preset(teks):
    st.session_state["input_cerita_utama"] = teks

col_search_main, col_console = st.columns([2.1, 1.0], gap="large")

with col_console:
    st.markdown(
        """<div class="pref-panel">
<div class="pref-header">Pengaturan Pencarian</div>
<div class="pref-label">Keseimbangan Musik</div>
</div>""",
        unsafe_allow_html=True,
    )

    bobot_audio_pct = st.slider(
        "Keseimbangan Lirik vs Irama",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        key="slider_bobot_audio",
        label_visibility="collapsed",
    )
    bobot_audio = bobot_audio_pct / 100.0
    pct_lirik = 100 - bobot_audio_pct

    st.caption(f"Fokus: {pct_lirik}% Lirik · {bobot_audio_pct}% Irama")

    st.markdown(
        """<div class="pref-label" style="margin-top: 14px;">Jumlah Rekomendasi</div>""",
        unsafe_allow_html=True,
    )

    jumlah_rekomendasi = st.slider(
        "Jumlah Lagu",
        min_value=2,
        max_value=10,
        value=6,
        step=1,
        key="slider_jumlah",
        label_visibility="collapsed",
    )
    st.caption(f"{jumlah_rekomendasi} lagu rekomendasi")

with col_search_main:
    # 1. Pilih Lagu Kesayangan
    idx_lagu = st.selectbox(
        "Pilih lagu kesukaanmu:",
        options=list(range(len(df_indo))),
        format_func=lambda i: opsi_lagu[i],
        index=0,
        help="Ketik judul lagu atau nama musisi untuk memfilter daftar.",
    )

    row_ref = df_indo.iloc[idx_lagu]

    # Reference Song DNA Preview Card
    clean_ref_name = html.escape(str(row_ref['name']))
    clean_ref_art = html.escape(str(row_ref['artists']))
    ref_track_id = row_ref.get("id", "")

    ref_card_html = (
        '<div class="reference-card" style="padding: 12px 18px; margin: 8px 0 14px 0;">\n'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">\n'
        '<div>\n'
        f'<div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; font-family: \'Outfit\', sans-serif;">{clean_ref_name}</div>\n'
        f'<div style="color: #b3b3b3; font-size: 0.88rem; font-weight: 500;">{clean_ref_art}</div>\n'
        '</div>\n'
        '<div>\n'
        f'<a href="https://open.spotify.com/track/{ref_track_id}" target="_blank" class="spotify-btn"><svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: -1px; margin-right: 5px;"><polygon points="6,4 20,12 6,20"/></svg>Buka Spotify</a>\n'
        '</div>\n'
        '</div>\n'
        '</div>'
    )
    st.markdown(ref_card_html, unsafe_allow_html=True)

    # 2. Curahan Perasaan / Suasana Hati (Opsional - Bisa Dimatikan/Dinyalakan)
    sertakan_cerita = st.toggle(
        "Tambah curahan suasana hati / perasaan",
        value=False,
        key="toggle_sertakan_cerita",
        help="Nyalakan jika ingin menyelaraskan musik dengan cerita atau perasaan yang sedang kamu rasakan.",
    )

    if sertakan_cerita:
        # Preset Quick Mood Chips (100% Emoji Free)
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            st.button("Hujan Sore", use_container_width=True, on_click=apply_preset, args=("Hujan sore yang dingin mengingatkanku pada kenangan masa lalu",))
            st.button("Kopi Senja", use_container_width=True, on_click=apply_preset, args=("Duduk sendiri di beranda menikmati kopi saat senja merenungi hidup",))
        with preset_col2:
            st.button("Patah Hati", use_container_width=True, on_click=apply_preset, args=("Rasa sakit karena harus berpisah dalam diam tanpa pamit",))
            st.button("Larut Malam", use_container_width=True, on_click=apply_preset, args=("Insomnia tengah malam menatap langit kamar memikirkan masa depan",))
        with preset_col3:
            st.button("Ikhlas Melepas", use_container_width=True, on_click=apply_preset, args=("Mencoba ikhlas melepaskan orang yang kucintai agar ia bahagia",))
            st.button("Rindu Rumah", use_container_width=True, on_click=apply_preset, args=("Rindu suasana hangat rumah dan pelukan orang-orang terkasih",))

        teks_cerita = st.text_area(
            "Tulis perasaanmu:",
            placeholder="Contoh: Lagi rindu masa lalu, pengen yang tenang dan syahdu...",
            height=85,
            key="input_cerita_utama",
            label_visibility="collapsed",
        )
    else:
        teks_cerita = ""

    col_btn, col_chk = st.columns([1.2, 1.8])
    with col_btn:
        btn_cari = st.button("Temukan Lagu", key="btn_cari_utama", type="primary", use_container_width=True)
    with col_chk:
        show_player = st.toggle("Tampilkan pemutar musik", value=True, key="toggle_player_utama")

if btn_cari:
    with st.spinner("Mencari lagu yang seirama..."):
        teks_final = teks_cerita.strip() if sertakan_cerita else ""
        model_embed = None
        if teks_final:
            model_embed = load_embed_model()
        hasil = rekomendasikan_terpadu(
            df_indo,
            embeddings,
            idx_lagu,
            teks_query=teks_final,
            model_embed=model_embed,
            jumlah=jumlah_rekomendasi,
            bobot_audio=bobot_audio,
        )
        st.session_state["hasil_utama"] = hasil
        st.session_state["ref_name_utama"] = row_ref['name']
        st.session_state["query_utama"] = teks_final

if "hasil_utama" in st.session_state:
    if st.session_state.get("query_utama"):
        st.markdown(f"#### Rekomendasi Berdasarkan *{st.session_state['ref_name_utama']}* & Curahan Hatimu:")
    else:
        st.markdown(f"#### Rekomendasi Serupa untuk *{st.session_state['ref_name_utama']}*:")
    render_hasil_cards(df_indo, st.session_state["hasil_utama"], show_player=show_player)



