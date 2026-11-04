# AgroScan AI — Precision Crop Pathology & Management Platform
# Professional Production UI
import os
import sys
from pathlib import Path
import json
import datetime
from typing import Tuple, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

from src.config import (
    MODEL_PATH, CLASS_INDICES_PATH, IMAGE_SIZE, SAMPLES_DIR,
    REPORTS_DIR, TRAINING_CURVES_PATH, CONFUSION_MATRIX_PATH,
    EVALUATION_REPORT_PATH
)
from src.remedies import get_remedy, load_disease_database
from src.ai_engine import PrecisionAIEngine
from src.epidemiology import fetch_live_weather, analyze_climate_fusion
from src.continual_learning import (
    save_sample_feedback,
    get_continual_learning_stats,
    trigger_incremental_fine_tuning
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AgroScan AI — Precision Crop Pathology Suite",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Premium Design System CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1a12 0%, #122318 40%, #0f2b1a 100%);
        border-right: 1px solid rgba(16, 185, 129, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.6rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(16, 185, 129, 0.08) !important;
    }
    [data-testid="stSidebar"] .stCheckbox label span {
        font-size: 0.82rem !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 30%, #047857 60%, #059669 100%);
        color: white;
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(52,211,153,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.5rem 0;
        color: #fff !important;
        line-height: 1.2;
    }
    .hero-banner p {
        font-size: 0.92rem;
        color: #a7f3d0;
        margin: 0;
        line-height: 1.6;
        max-width: 720px;
        font-weight: 400;
    }
    .hero-version {
        position: absolute;
        top: 1.2rem;
        right: 1.5rem;
        background: rgba(255,255,255,0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #6ee7b7;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 0.6rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
        letter-spacing: -0.01em;
    }
    .section-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin: -0.3rem 0 1rem 0;
        font-weight: 400;
    }

    /* ── Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        transition: all 0.15s ease;
    }
    .metric-card:hover {
        border-color: #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
    }
    .metric-card .mc-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .metric-card .mc-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .metric-card .mc-hint {
        font-size: 0.7rem;
        color: #10b981;
        margin-top: 0.15rem;
        font-weight: 500;
    }

    /* ── Weather Pill ── */
    .wx-pill {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .wx-pill .wx-icon { font-size: 1.5rem; margin-bottom: 0.2rem; }
    .wx-pill .wx-label { font-size: 0.7rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
    .wx-pill .wx-value { font-size: 1.2rem; font-weight: 700; margin: 0.15rem 0; }
    .wx-pill .wx-status { font-size: 0.7rem; font-weight: 600; }

    /* ── Severity Badges ── */
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-green { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-yellow { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .badge-red { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-purple { background: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }

    /* ── Risk Gauge ── */
    .risk-gauge {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .risk-gauge .rg-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .risk-gauge .rg-value {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .risk-gauge .rg-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* ── Tank Card ── */
    .tank-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.3rem;
        border: 1px solid #e2e8f0;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
    }
    .tank-card h4 {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin: 0 0 0.4rem 0;
    }
    .tank-card .tc-val {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .tank-card .tc-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin: 0.2rem 0 0 0;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.75rem;
        color: #94a3b8;
    }

    /* ── Streamlit overrides ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border: none;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.65rem 1.5rem;
        border-radius: 10px;
        letter-spacing: -0.01em;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3);
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #64748b !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_cached_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

@st.cache_data
def load_class_labels() -> List[str]:
    if CLASS_INDICES_PATH.exists():
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [data[str(i)] for i in range(len(data))]
    from src.config import DEFAULT_CLASSES
    return DEFAULT_CLASSES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 2.2rem; margin-bottom: 0.2rem;">🌱</div>
        <div style="font-size: 1.2rem; font-weight: 800; letter-spacing: -0.02em; color: #10b981 !important;">AgroScan AI</div>
        <div style="font-size: 0.7rem; font-weight: 500; color: #6ee7b7 !important; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.15rem;">Precision Crop Pathology</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🔬 Diagnosis Engine",
            "🔄 Continual Learning & Retrain",
            "💊 Spray Calculator",
            "🌦️ Weather Advisory",
            "📊 Model Analytics",
            "📖 Disease Library"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#6ee7b7 !important; margin-bottom:0.5rem;">AI Settings</p>', unsafe_allow_html=True)
    use_tta = st.checkbox("Test-Time Augmentation", value=True, help="Multi-angle inference for robustness")
    use_foliar_gate = st.checkbox("Foliar Quality Gate", value=True, help="Rejects non-plant images")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #94a3b8 !important; line-height: 1.7;">
        <div style="font-weight:600; color:#6ee7b7 !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">Specifications</div>
        <div>Model: MobileNetV2</div>
        <div>Classes: 38 Pathologies</div>
        <div>Val Accuracy: 89.2%</div>
        <div>Top-3: 98.3%</div>
        <div style="margin-top:0.8rem; font-size:0.65rem; color:#475569 !important;">CSE276 Project &middot; v2.0</div>
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hero Banner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="hero-banner">
    <div class="hero-version">v2.0 &middot; MobileNetV2</div>
    <h1>AgroScan AI</h1>
    <p>Upload a leaf photo or capture one with your camera. Our deep learning engine identifies diseases across 38 crop pathologies, quantifies foliar damage, and generates climate-aware treatment protocols — all in seconds.</p>
</div>
""", unsafe_allow_html=True)

model = load_cached_model()
class_labels = load_class_labels()
ai_engine = PrecisionAIEngine(model, class_labels) if model else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: Diagnosis Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if menu == "🔬 Diagnosis Engine":

    # ── Step 1: Image Input ──
    st.markdown('<div class="section-header">Step 1 &mdash; Leaf Specimen</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Provide a clear, close-up photograph of the affected leaf surface.</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["Upload from device", "Camera capture", "Benchmark samples"],
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_image = None
    specimen_name = "Leaf Specimen"

    col_in, col_pre = st.columns([1, 1], gap="large")

    with col_in:
        if input_mode == "Upload from device":
            uploaded = st.file_uploader("Upload leaf image (JPG / PNG)", type=["jpg", "jpeg", "png"])
            if uploaded:
                selected_image = Image.open(uploaded)
                specimen_name = uploaded.name

        elif input_mode == "Camera capture":
            cam = st.camera_input("Capture leaf photo")
            if cam:
                selected_image = Image.open(cam)
                specimen_name = "Camera Capture"

        elif input_mode == "Benchmark samples":
            samples = list(SAMPLES_DIR.glob("*_sample.jpg"))
            if samples:
                names = [s.stem.replace("_sample", "").replace("___", " — ").replace("_", " ") for s in samples]
                choice = st.selectbox("Select benchmark image", names)
                idx = names.index(choice)
                selected_image = Image.open(samples[idx])
                specimen_name = samples[idx].name

    with col_pre:
        if selected_image:
            st.image(selected_image, caption=specimen_name, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#f8fafc; border:2px dashed #cbd5e1; border-radius:12px; padding:3rem 1.5rem; text-align:center; color:#94a3b8;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">📷</div>
                <div style="font-size:0.88rem; font-weight:600; color:#64748b;">No image selected</div>
                <div style="font-size:0.78rem; margin-top:0.3rem;">Upload, capture, or select a sample from the left panel</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Step 2: Location & Weather ──
    if selected_image:
        st.markdown("---")
        st.markdown('<div class="section-header">Step 2 &mdash; Farm Location & Live Weather</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Optional. Entering your city enables climate-fused epidemiological risk analysis.</div>', unsafe_allow_html=True)

        lc1, lc2 = st.columns([1, 2], gap="large")
        with lc1:
            city_input = st.text_input(
                "Farm city",
                placeholder="e.g. Amritsar, Pune, Delhi",
                key="city_weather_input",
                label_visibility="collapsed"
            )

        weather_data = None
        if city_input and city_input.strip():
            with st.spinner(f"Fetching weather for {city_input.strip()}..."):
                weather_data = fetch_live_weather(city_input.strip())

            if weather_data:
                with lc2:
                    st.markdown(f'<div style="font-size:0.8rem; color:#059669; font-weight:600;">📍 {weather_data["city"]} &middot; {weather_data["latitude"]:.2f}°N, {weather_data["longitude"]:.2f}°E</div>', unsafe_allow_html=True)

                wc1, wc2, wc3, wc4 = st.columns(4)
                with wc1:
                    st.markdown(f'<div class="wx-pill"><div class="wx-icon">🌡️</div><div class="wx-label">Temperature</div><div class="wx-value" style="color:#0f172a;">{weather_data["temperature"]}°C</div></div>', unsafe_allow_html=True)
                with wc2:
                    st.markdown(f'<div class="wx-pill"><div class="wx-icon">💧</div><div class="wx-label">Humidity</div><div class="wx-value" style="color:#0f172a;">{weather_data["humidity"]}%</div></div>', unsafe_allow_html=True)
                with wc3:
                    st.markdown(f'<div class="wx-pill"><div class="wx-icon">💨</div><div class="wx-label">Wind</div><div class="wx-value" style="color:#0f172a;">{weather_data["wind_speed"]} km/h</div></div>', unsafe_allow_html=True)
                with wc4:
                    st.markdown(f'<div class="wx-pill"><div class="wx-icon">🌧️</div><div class="wx-label">Precipitation</div><div class="wx-value" style="color:#0f172a;">{weather_data["precipitation"]} mm</div></div>', unsafe_allow_html=True)
            else:
                st.warning(f"Could not find weather data for '{city_input}'. Check spelling.")

        # ── Step 3: Run Analysis ──
        st.markdown("---")
        st.markdown('<div class="section-header">Step 3 &mdash; Run Analysis</div>', unsafe_allow_html=True)
        btn = st.button("Run Pathology Analysis", type="primary", use_container_width=True)

        if btn:
            if not ai_engine:
                st.error("Model not found. Train the model first with `python src/train.py`.")
            else:
                with st.spinner("Running MobileNetV2 inference, saliency mapping, and damage segmentation..."):
                    result = ai_engine.diagnose(selected_image, run_validation=use_foliar_gate)

                if not result.get("success"):
                    st.error(f"**Quality Gate:** {result.get('message')}")
                    st.info("Please provide a clear, focused photo of a plant leaf.")
                else:
                    top = result["top_prediction"]
                    pred_class = top["class_name"]
                    confidence = top["confidence"]
                    remedy = get_remedy(pred_class)
                    damage = result["damage"]
                    status = result["status"]

                    crop_name = remedy["crop"]
                    disease_name = remedy["disease_name"]
                    severity = remedy["severity"]

                    # ── Results Header ──
                    st.markdown("---")
                    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

                    r1, r2, r3, r4 = st.columns(4)
                    with r1:
                        st.markdown(f'<div class="metric-card"><div class="mc-label">Crop</div><div class="mc-value">{crop_name}</div></div>', unsafe_allow_html=True)
                    with r2:
                        st.markdown(f'<div class="metric-card"><div class="mc-label">Pathology</div><div class="mc-value" style="font-size:1.1rem;">{disease_name}</div></div>', unsafe_allow_html=True)
                    with r3:
                        conf_color = "#059669" if confidence > 0.8 else ("#d97706" if confidence > 0.5 else "#dc2626")
                        st.markdown(f'<div class="metric-card"><div class="mc-label">Confidence</div><div class="mc-value" style="color:{conf_color};">{confidence*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with r4:
                        dmg_color = "#059669" if damage["percentage"] < 10 else ("#d97706" if damage["percentage"] < 30 else "#dc2626")
                        st.markdown(f'<div class="metric-card"><div class="mc-label">Foliar Damage</div><div class="mc-value" style="color:{dmg_color};">{damage["percentage"]}%</div></div>', unsafe_allow_html=True)

                    # Severity badge
                    st.markdown("<div style='margin:0.8rem 0;'>", unsafe_allow_html=True)
                    if status == "NOVEL_OR_MIXED_PATHOLOGY":
                        st.markdown('<span class="badge badge-purple">⚠ NOVEL / ATYPICAL PATHOLOGY</span>', unsafe_allow_html=True)
                        st.warning(result["status_note"])
                    elif "healthy" in pred_class.lower():
                        st.markdown('<span class="badge badge-green">✓ HEALTHY FOLIAGE</span>', unsafe_allow_html=True)
                    elif severity.lower() in ["severe", "severe / critical", "high"]:
                        st.markdown(f'<span class="badge badge-red">⚠ HIGH SEVERITY — {severity.upper()}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge badge-yellow">● MODERATE — {severity.upper()}</span>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # ── Climate Fusion ──
                    fusion_data = None
                    city_val = st.session_state.get("city_weather_input", "").strip()
                    if city_val:
                        wd = fetch_live_weather(city_val)
                        if wd:
                            fusion_data = analyze_climate_fusion(
                                pred_class, wd["temperature"], wd["humidity"], damage["percentage"]
                            )

                            st.markdown("---")
                            st.markdown('<div class="section-header">Climate-Fused Risk Assessment</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="section-sub">Live weather at <strong>{wd["city"]}</strong> combined with visual diagnosis.</div>', unsafe_allow_html=True)

                            risk = fusion_data["climatic_risk_index"]
                            if risk < 40:
                                risk_color, risk_label = "#059669", "LOW"
                            elif risk < 65:
                                risk_color, risk_label = "#d97706", "MODERATE"
                            else:
                                risk_color, risk_label = "#dc2626", "HIGH"

                            fc1, fc2 = st.columns(2, gap="large")
                            with fc1:
                                st.markdown(f"""
                                <div class="risk-gauge">
                                    <div class="rg-title">Climate Risk Index</div>
                                    <div class="rg-value" style="color:{risk_color};">{risk}% <span style="font-size:0.8rem; font-weight:600;">{risk_label}</span></div>
                                    <div class="rg-sub">Favorability of current conditions for pathogen spread</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric("Infection Onset", fusion_data["estimated_onset_time"])
                                st.metric("Biological Stage", fusion_data["biological_stage"])
                            with fc2:
                                st.markdown(f"""
                                <div class="risk-gauge">
                                    <div class="rg-title">Pathogen Profile</div>
                                    <div class="rg-value" style="font-size:1rem; color:#0f172a;">{fusion_data['pathogen_type']}</div>
                                    <div class="rg-sub">Spread: {fusion_data['spread_rate']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.info(fusion_data["climate_diagnosis"])
                                if fusion_data["is_climate_favorable"]:
                                    st.error(f"**{fusion_data['action_urgency']}**")
                                else:
                                    st.warning(f"**{fusion_data['action_urgency']}**")

                            st.progress(min(risk / 100.0, 1.0))

                    # ── Saliency Visualizer ──
                    st.markdown("---")
                    st.markdown('<div class="section-header">Explainable AI Visualizer</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-sub">Saliency heatmap shows where the model focused. Damage mask quantifies infected tissue area.</div>', unsafe_allow_html=True)

                    viz = st.radio(
                        "View", ["All Layers", "Original", "Saliency Heatmap", "Damage Mask"],
                        horizontal=True, label_visibility="collapsed"
                    )

                    if viz == "All Layers":
                        v1, v2, v3 = st.columns(3)
                        with v1:
                            st.image(selected_image.resize((224, 224)), caption="Original", use_container_width=True)
                        with v2:
                            st.image(result["heatmap"], caption="Saliency Heatmap", use_container_width=True)
                        with v3:
                            st.image(damage["mask"], caption=f"Damage Mask ({damage['percentage']}%)", use_container_width=True)
                    elif viz == "Original":
                        st.image(selected_image.resize((350, 350)), caption="Original Specimen")
                    elif viz == "Saliency Heatmap":
                        st.image(result["heatmap"].resize((350, 350)), caption="Convolutional Saliency")
                    elif viz == "Damage Mask":
                        st.image(damage["mask"].resize((350, 350)), caption=f"Lesion Mask — {damage['stage']}")

                    st.markdown(f"**Damage Stage:** {damage['stage']}")
                    st.progress(min(damage['percentage'] / 100.0, 1.0))

                    # ── Top-5 Candidates ──
                    st.markdown("---")
                    st.markdown('<div class="section-header">Differential Diagnosis</div>', unsafe_allow_html=True)
                    cands = result["candidates"]
                    cand_df = pd.DataFrame({
                        "Pathology": [c["class_name"].replace("___", " — ").replace("_", " ") for c in cands],
                        "Probability (%)": [round(c["confidence"] * 100, 2) for c in cands]
                    })
                    st.bar_chart(cand_df.set_index("Pathology"))

                    # ── Treatment Protocol ──
                    st.markdown("---")
                    st.markdown('<div class="section-header">Treatment Protocol</div>', unsafe_allow_html=True)

                    t1, t2, t3, t4, t5 = st.tabs([
                        "Symptoms", "Organic", "Chemical", "Prevention", "Prescription"
                    ])

                    with t1:
                        st.markdown(f"**Cause:** {remedy.get('cause', 'N/A')}")
                        st.markdown(f"**Symptoms:** {remedy.get('symptoms', 'N/A')}")
                        st.markdown(f"**Favorable Conditions:** {remedy.get('favorable_conditions', 'N/A')}")

                    with t2:
                        for item in remedy.get("organic_remedies", []):
                            st.markdown(f"- {item}")
                        if not remedy.get("organic_remedies"):
                            st.info("No specific organic intervention required.")

                    with t3:
                        for chem in remedy.get("chemical_treatments", []):
                            st.markdown(f"- {chem}")
                        if remedy.get("chemical_treatments"):
                            st.caption("Follow local label rates. Observe PHI. Rotate FRAC groups.")
                        else:
                            st.info("No chemical treatment indicated.")

                    with t4:
                        for prev in remedy.get("prevention", []):
                            st.markdown(f"- {prev}")

                    with t5:
                        rx_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        wx_block = ""
                        if fusion_data and city_val:
                            wx_block = (
                                f"\nCLIMATE DATA ({city_val}):\n"
                                f"  Climate Risk: {fusion_data['climatic_risk_index']}%\n"
                                f"  Infection Onset: {fusion_data['estimated_onset_time']}\n"
                                f"  Stage: {fusion_data['biological_stage']}\n"
                                f"  Urgency: {fusion_data['action_urgency']}\n"
                            )
                        rx = (
                            "AGROSCAN AI — CROP PATHOLOGY PRESCRIPTION\n"
                            "=" * 50 + "\n"
                            f"Date       : {rx_date}\n"
                            f"Specimen   : {specimen_name}\n"
                            f"Crop       : {crop_name}\n"
                            f"Disease    : {disease_name}\n"
                            f"Severity   : {severity} ({damage['stage']})\n"
                            f"Damage     : {damage['percentage']}%\n"
                            f"Confidence : {confidence*100:.1f}%\n"
                            f"Cause      : {remedy.get('cause', 'N/A')}\n"
                            + wx_block +
                            "\nTREATMENT:\n" + "-" * 50 + "\n"
                            f"Organic  : {'; '.join(remedy.get('organic_remedies', ['—']))}\n"
                            f"Chemical : {'; '.join(remedy.get('chemical_treatments', ['—']))}\n"
                            f"Prevent  : {'; '.join(remedy.get('prevention', ['—']))}\n\n"
                            "Generated by AgroScan AI v2.0\n"
                            "=" * 50
                        )
                        st.code(rx, language=None)
                        st.download_button(
                            "Download Prescription (.txt)",
                            data=rx,
                            file_name=f"AgroScan_{crop_name}_{disease_name.replace(' ','_')}.txt",
                            mime="text/plain"
                        )

                    # ── Active Learning & Model Feedback Loop ──
                    st.markdown("---")
                    st.markdown('<div class="section-header">🤖 Active Learning & Model Self-Improvement</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-sub">Help the model learn! Confirm or correct this diagnosis so the AI adds it to its training memory and becomes more accurate.</div>', unsafe_allow_html=True)

                    fb_col1, fb_col2 = st.columns([1, 1], gap="medium")
                    with fb_col1:
                        if st.button("✅ Confirm Diagnosis is Accurate (Save to AI Memory)", use_container_width=True):
                            rec = save_sample_feedback(
                                image=selected_image,
                                predicted_class=pred_class,
                                confidence=confidence,
                                user_confirmed=True,
                                city=city_val or "Unknown",
                                temperature=wd["temperature"] if 'wd' in locals() and wd else None,
                                humidity=wd["humidity"] if 'wd' in locals() and wd else None
                            )
                            st.success(f"Sample registered (ID: {rec['id']})! The AI memory now stores this verified field sample.")
                    with fb_col2:
                        with st.expander("✏️ Correct / Provide Alternate Label"):
                            correct_label = st.selectbox(
                                "Actual pathology (if different from predicted):",
                                options=class_labels,
                                index=class_labels.index(pred_class) if pred_class in class_labels else 0
                            )
                            if st.button("💾 Save Corrected Label for Retraining"):
                                rec = save_sample_feedback(
                                    image=selected_image,
                                    predicted_class=pred_class,
                                    confidence=confidence,
                                    user_confirmed=False,
                                    corrected_class=correct_label,
                                    city=city_val or "Unknown",
                                    temperature=wd["temperature"] if 'wd' in locals() and wd else None,
                                    humidity=wd["humidity"] if 'wd' in locals() and wd else None
                                )
                                st.success(f"Correction registered (ID: {rec['id']})! The AI will learn from this in the next fine-tuning cycle.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE: Continual Learning & Retrain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "🔄 Continual Learning & Retrain":
    st.markdown('<div class="section-header">🔄 Continual Learning & Active Model Evolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">This system turns AgroScan into an evolving AI. Field samples and user feedback accumulate here, enabling one-click incremental fine-tuning to permanently improve prediction accuracy over time.</div>', unsafe_allow_html=True)

    stats = get_continual_learning_stats()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="mc-label">Total Field Samples</div><div class="mc-value">{stats["total_samples"]}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="mc-label">Pending Fine-Tuning</div><div class="mc-value" style="color:#d97706;">{stats["pending_retrain"]}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="mc-label">Verified Correct</div><div class="mc-value" style="color:#059669;">{stats["confirmed_count"]}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="mc-label">User Corrections</div><div class="mc-value" style="color:#2563eb;">{stats["corrected_count"]}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Trigger Incremental Fine-Tuning")
    st.write("When you click this button, the deep learning model loads its current weights, unfreezes its top layers with a conservative learning rate (1e-5), and trains on the newly collected field images. The improved model is saved back to disk immediately.")

    fine_col1, fine_col2 = st.columns([1, 2], gap="large")
    with fine_col1:
        epochs = st.slider("Fine-Tuning Epochs:", min_value=1, max_value=5, value=2)
        btn_fine_tune = st.button("🚀 Train Model on New Data Now", type="primary", use_container_width=True)

        if btn_fine_tune:
            if stats["pending_retrain"] == 0:
                st.warning("⚠️ No pending new samples to learn from. Upload or diagnose leaves first, then confirm or correct them.")
            else:
                with st.spinner(f"Fine-tuning MobileNetV2 on {stats['pending_retrain']} new samples..."):
                    res = trigger_incremental_fine_tuning(epochs=epochs)
                if res.get("success"):
                    st.success(f"🎉 **{res['message']}**")
                    st.metric("Training Accuracy on New Data", f"{res['accuracy']}%")
                    st.metric("Final Loss", f"{res['loss']}")
                    st.cache_resource.clear() # clear cached model so Streamlit reloads new weights
                else:
                    st.error(f"Error: {res.get('message')}")

    with fine_col2:
        st.markdown("#### 📋 Recent Feedback Registry Entries")
        if stats["records"]:
            rec_df = pd.DataFrame([
                {
                    "ID": r["id"],
                    "Time": r["timestamp"][:16].replace("T", " "),
                    "Predicted": r["predicted_class"].replace("___", " - "),
                    "Ground Truth": r["ground_truth_label"].replace("___", " - "),
                    "Status": "Learned" if r.get("used_in_retraining") else "Pending",
                    "City": r.get("city", "Unknown")
                }
                for r in reversed(stats["records"])
            ])
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("No samples recorded yet. Run a diagnosis and click 'Confirm' or 'Correct' to begin accumulating field data.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2: Spray Calculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "💊 Spray Calculator":
    st.markdown('<div class="section-header">Chemical Dosage & Tank Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Calculate water volumes, chemical quantities, and sprayer refills for your plot.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        unit = st.selectbox("Area unit", ["Acres", "Hectares"])
        area = st.number_input(f"Plot size ({unit})", min_value=0.25, max_value=500.0, value=2.0, step=0.5)
        target_crop = st.selectbox("Crop", ["Potato", "Tomato", "Corn (Maize)", "Apple", "Grape", "Pepper"])
        fungicide = st.selectbox("Fungicide", [
            "Mancozeb 75% WP — 2.5 kg/ha",
            "Chlorothalonil 720 SC — 2.0 L/ha",
            "Copper Hydroxide 77% WP — 2.0 kg/ha",
            "Metalaxyl + Mancozeb (Ridomil MZ) — 2.5 kg/ha",
            "Azoxystrobin 23% SC — 500 ml/ha"
        ])
        sprayer = st.selectbox("Sprayer", ["Knapsack 16L", "Motorized 20L", "Tractor Boom 400L"])

    hectares = area if unit == "Hectares" else area / 2.471
    water = round(hectares * 500, 1)

    if "Mancozeb" in fungicide:
        chem_desc = f"{round(hectares * 2.5, 2)} kg Mancozeb 75 WP"
    elif "Chlorothalonil" in fungicide:
        chem_desc = f"{round(hectares * 2.0, 2)} L Chlorothalonil 720 SC"
    elif "Azoxystrobin" in fungicide:
        chem_desc = f"{round(hectares * 500, 1)} ml Azoxystrobin SC"
    else:
        chem_desc = f"{round(hectares * 2.0, 2)} kg formulation"

    cap = 16 if "16" in sprayer else (20 if "20" in sprayer else 400)
    refills = int(np.ceil(water / cap))

    with c2:
        st.markdown(f"""
        <div class="tank-card" style="border-left-color: #059669;">
            <h4>Total Water Required</h4>
            <div class="tc-val" style="color:#059669;">{water} L</div>
            <div class="tc-sub">For {area} {unit} of {target_crop}</div>
        </div>
        <div class="tank-card" style="border-left-color: #0891b2;">
            <h4>Chemical Dosage</h4>
            <div class="tc-val" style="color:#0891b2;">{chem_desc}</div>
            <div class="tc-sub">Tank concentration: ~2.5–3 g/L</div>
        </div>
        <div class="tank-card" style="border-left-color: #d97706;">
            <h4>Sprayer Refills</h4>
            <div class="tc-val" style="color:#d97706;">{refills} tanks ({cap}L each)</div>
            <div class="tc-sub">~{round((water / refills) * 0.005 * 1000, 1)}g chemical per tank</div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: Weather Advisory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "🌦️ Weather Advisory":
    st.markdown('<div class="section-header">Spray Window & Weather Advisory</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter your city to get a real-time spray safety assessment.</div>', unsafe_allow_html=True)

    spray_city = st.text_input("Farm city", placeholder="e.g. Ludhiana, Nashik, Jaipur")

    live_w = None
    if spray_city and spray_city.strip():
        with st.spinner(f"Fetching weather for {spray_city}..."):
            live_w = fetch_live_weather(spray_city.strip())
        if not live_w:
            st.warning(f"No data for '{spray_city}'.")

    temp = live_w["temperature"] if live_w else 23.5
    wind = live_w["wind_speed"] if live_w else 5.2
    hum = live_w["humidity"] if live_w else 60.0
    prec = live_w["precipitation"] if live_w else 0.0

    if live_w:
        st.markdown(f'<div style="font-size:0.8rem; color:#059669; font-weight:600; margin-bottom:0.8rem;">📍 {live_w["city"]} &middot; Updated now</div>', unsafe_allow_html=True)

    # Status logic
    t_ok = 18 <= temp <= 28
    t_col = "#059669" if t_ok else "#d97706"
    t_st = "Optimal" if t_ok else ("Too hot" if temp > 28 else "Too cold")

    w_ok = wind <= 10
    w_col = "#059669" if w_ok else ("#d97706" if wind <= 20 else "#dc2626")
    w_st = "Low drift" if w_ok else ("Moderate drift" if wind <= 20 else "High drift")

    h_col = "#d97706" if hum >= 70 else "#059669"
    h_st = "Disease risk" if hum >= 70 else ("Low" if hum < 55 else "Monitor")

    p_ok = prec < 1.0
    p_col = "#059669" if p_ok else "#dc2626"
    p_st = "Rainfast OK" if p_ok else "Rain active"

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(f'<div class="wx-pill"><div class="wx-icon">🌡️</div><div class="wx-label">Temp</div><div class="wx-value" style="color:{t_col};">{temp}°C</div><div class="wx-status" style="color:{t_col};">{t_st}</div></div>', unsafe_allow_html=True)
    with w2:
        st.markdown(f'<div class="wx-pill"><div class="wx-icon">💨</div><div class="wx-label">Wind</div><div class="wx-value" style="color:{w_col};">{wind} km/h</div><div class="wx-status" style="color:{w_col};">{w_st}</div></div>', unsafe_allow_html=True)
    with w3:
        st.markdown(f'<div class="wx-pill"><div class="wx-icon">💧</div><div class="wx-label">Humidity</div><div class="wx-value" style="color:{h_col};">{hum}%</div><div class="wx-status" style="color:{h_col};">{h_st}</div></div>', unsafe_allow_html=True)
    with w4:
        st.markdown(f'<div class="wx-pill"><div class="wx-icon">🌧️</div><div class="wx-label">Rain</div><div class="wx-value" style="color:{p_col};">{prec} mm</div><div class="wx-status" style="color:{p_col};">{p_st}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    safe = (wind <= 15 and prec < 1.0 and 15 <= temp <= 32)
    if safe:
        st.success("**SPRAY WINDOW: SUITABLE** — Low drift risk, no rain washoff. Spray early morning (6–9 AM) or evening (5–7 PM).")
    elif prec >= 1.0:
        st.error("**DELAY SPRAY** — Active precipitation detected. Wait for dry conditions and a 2+ hour rainfast window.")
    elif wind > 20:
        st.error("**STOP — HIGH DRIFT** — Wind exceeds 20 km/h. Risk of off-target chemical drift.")
    else:
        st.warning("**USE CAUTION** — Conditions are marginal. Use low-drift nozzles and adjuvants.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Model Analytics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "📊 Model Analytics":
    st.markdown('<div class="section-header">Model Architecture & Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">MobileNetV2 transfer learning model trained on 38 crop disease classes.</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="metric-card"><div class="mc-label">Architecture</div><div class="mc-value" style="font-size:1.1rem;">MobileNetV2</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="metric-card"><div class="mc-label">Parameters</div><div class="mc-value">2.60M</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="metric-card"><div class="mc-label">Top-1 Accuracy</div><div class="mc-value" style="color:#059669;">89.21%</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="metric-card"><div class="mc-label">Top-3 Accuracy</div><div class="mc-value" style="color:#059669;">98.29%</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    cc, cm = st.columns(2, gap="medium")
    with cc:
        st.markdown("**Training Curves**")
        if TRAINING_CURVES_PATH.exists():
            st.image(str(TRAINING_CURVES_PATH), use_container_width=True)
        else:
            st.info("Run `python src/train.py` to generate.")
    with cm:
        st.markdown("**Confusion Matrix**")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        else:
            st.info("Run evaluation to generate.")

    st.markdown("---")
    st.markdown("**Per-Class Metrics**")
    if EVALUATION_REPORT_PATH.exists():
        df = pd.read_csv(EVALUATION_REPORT_PATH, index_col=0)
        st.dataframe(
            df.style.format("{:.2f}").background_gradient(cmap="YlGnBu", subset=["precision", "recall", "f1-score"]),
            use_container_width=True
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5: Disease Library
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "📖 Disease Library":
    st.markdown('<div class="section-header">38-Class Pathology Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Browse symptoms, treatments, and prevention strategies for all supported crop diseases.</div>', unsafe_allow_html=True)

    db = load_disease_database()
    crops = sorted(set(v["crop"] for v in db.values()))
    filt = st.selectbox("Filter by crop", ["All Crops"] + crops)
    items = {k: v for k, v in db.items() if filt == "All Crops" or v["crop"] == filt}

    for key, d in items.items():
        sev_badge = "badge-red" if d["severity"].lower() in ["severe", "high", "severe / critical"] else ("badge-yellow" if d["severity"].lower() == "moderate" else "badge-green")
        with st.expander(f"{d['crop']}  —  {d['disease_name']}"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f'<span class="badge {sev_badge}">{d["severity"]}</span>', unsafe_allow_html=True)
                st.markdown(f"\n**Cause:** {d['cause']}")
                st.markdown(f"**Symptoms:** {d['symptoms']}")
                st.markdown(f"**Favorable Conditions:** {d['favorable_conditions']}")
            with ec2:
                st.markdown("**Organic Remedies:**")
                for o in d.get("organic_remedies", []):
                    st.markdown(f"- {o}")
                st.markdown("**Chemical Treatments:**")
                for c in d.get("chemical_treatments", []):
                    st.markdown(f"- {c}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="app-footer">
    AgroScan AI v2.0 &middot; MobileNetV2 &middot; 38 Pathologies &middot; CSE276 Project<br>
    Built with TensorFlow, Streamlit & Open-Meteo
</div>
""", unsafe_allow_html=True)
