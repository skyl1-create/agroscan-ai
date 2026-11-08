# AgroScan AI — Precision Crop Pathology & Climate-Fused Agronomic Suite
# High-End Agronomic Intelligence Dashboard
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
    page_title="AgroScan AI — Smart Crop Pathology Suite",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enterprise-Grade CSS & Glassmorphism Design System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Global Typography & Base ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    #MainMenu, footer, header {visibility: hidden !important;}

    /* ── Dark Luxury Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06140d 0%, #0a2016 45%, #05160e 100%) !important;
        border-right: 1px solid rgba(52, 211, 153, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 0.8rem !important;
        border-radius: 10px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(16, 185, 129, 0.12) !important;
        border-color: rgba(52, 211, 153, 0.3) !important;
        transform: translateX(4px);
    }

    /* ── Ultra-Modern Hero Banner ── */
    .hero-container {
        background: radial-gradient(circle at 85% 15%, rgba(16, 185, 129, 0.25) 0%, transparent 60%),
                    linear-gradient(135deg, #022c22 0%, #064e3b 45%, #047857 100%);
        color: #ffffff;
        padding: 2.8rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px -15px rgba(2, 44, 34, 0.4);
    }
    .hero-container h1 {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 0.6rem 0;
        color: #ffffff !important;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-container p {
        font-size: 0.98rem;
        color: #a7f3d0;
        margin: 0;
        line-height: 1.65;
        max-width: 780px;
        font-weight: 400;
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.14);
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #6ee7b7;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 0.8rem;
    }

    /* ── Stepper Header ── */
    .step-box {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .step-circle {
        background: #059669;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .step-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .step-sub {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: -0.4rem;
        margin-bottom: 1.1rem;
    }

    /* ── Glassmorphism Metric Cards ── */
    .agro-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    .agro-card:hover {
        border-color: #10b981;
        box-shadow: 0 10px 24px -4px rgba(16, 185, 129, 0.12);
        transform: translateY(-2px);
    }
    .agro-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .agro-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
    }
    .agro-sub {
        font-size: 0.76rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    /* ── Weather Ribbon ── */
    .wx-ribbon {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.06);
    }
    .wx-icon { font-size: 1.6rem; margin-bottom: 0.2rem; }
    .wx-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #047857; letter-spacing: 0.06em; }
    .wx-num { font-size: 1.35rem; font-weight: 800; color: #064e3b; margin: 0.2rem 0; }

    /* ── Severity Badges ── */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-green { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .badge-yellow { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
    .badge-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .badge-purple { background: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }

    /* ── Clinical Rx Box ── */
    .rx-card {
        background: #f8fafc;
        border: 2px dashed #059669;
        border-radius: 14px;
        padding: 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #1e293b;
        margin: 1rem 0;
    }

    /* ── Clean Button Styling ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.8rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(5, 150, 105, 0.4) !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #047857 100%) !important;
        border-radius: 9999px !important;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-top: 3.5rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.78rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Initializing MobileNetV2 Deep Learning Weights...")
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
# Sidebar Navigation & Diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem 0 0.8rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🌱</div>
        <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; color: #34d399 !important;">AgroScan AI</div>
        <div style="font-size: 0.72rem; font-weight: 600; color: #a7f3d0 !important; letter-spacing: 0.08em; text-transform: uppercase;">Precision Crop Pathology</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigation Menu",
        [
            "🔬 Diagnostic Engine",
            "🔄 Continual Learning & Retrain",
            "💊 Spray & Tank Calculator",
            "🌦️ Live Weather Advisory",
            "📊 Model Performance & XAI",
            "📖 38-Class Knowledge Base"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#34d399 !important; margin-bottom:0.5rem;">Precision AI Controls</p>', unsafe_allow_html=True)
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=True, help="Averages 4 photometric transformations for lighting robustness")
    use_foliar_gate = st.checkbox("Foliar Vegetative Gate", value=True, help="Rejects non-plant images automatically")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.74rem; color: #cbd5e1 !important; line-height: 1.8;">
        <div style="font-weight:700; color:#34d399 !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">System Architecture</div>
        <div><strong>Model:</strong> MobileNetV2 (Transfer Learning)</div>
        <div><strong>Pathology Classes:</strong> 38 Conditions</div>
        <div><strong>Top-1 Accuracy:</strong> 89.21% (Top-3: 98.29%)</div>
        <div><strong>Latency:</strong> ~35ms / inference</div>
        <div style="margin-top:0.9rem; font-size:0.68rem; color:#64748b !important;">AgroScan v2.5 &middot; CSE276 Project</div>
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Top Hero Banner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="hero-container">
    <div class="hero-badge-pill">
        <span>✨</span> Production v2.5 &middot; MobileNetV2 Deep Learning &middot; 38 Classes
    </div>
    <h1>🌱 AgroScan AI: Precision Pathology Platform</h1>
    <p>Upload a leaf photograph or capture live from field cameras. Our neural network instantly classifies diseases across 38 crop pathologies, segments foliar damage percentages, and fuses microclimate epidemiology for precision agronomic treatment.</p>
</div>
""", unsafe_allow_html=True)

model = load_cached_model()
class_labels = load_class_labels()
ai_engine = PrecisionAIEngine(model, class_labels) if model else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: Diagnostic Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if menu == "🔬 Diagnostic Engine":

    # Step 1: Specimen
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">1</div>
        <div class="step-title">Leaf Specimen Acquisition</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Select an input source: upload a high-resolution leaf close-up, capture directly with your camera, or choose a benchmark sample.</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Specimen Input Mode",
        ["📁 Upload from Device", "📷 Live Camera Capture", "🖼️ Benchmark Field Samples"],
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_image = None
    specimen_name = "Leaf Specimen"

    col_input, col_view = st.columns([1, 1], gap="large")

    with col_input:
        if input_mode == "📁 Upload from Device":
            uploaded = st.file_uploader("Upload leaf image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                selected_image = Image.open(uploaded)
                specimen_name = uploaded.name

        elif input_mode == "📷 Live Camera Capture":
            cam_shot = st.camera_input("Take a photo of the affected crop leaf:")
            if cam_shot:
                selected_image = Image.open(cam_shot)
                specimen_name = "Field Camera Shot"

        elif input_mode == "🖼️ Benchmark Field Samples":
            samples = list(SAMPLES_DIR.glob("*_sample.jpg"))
            if samples:
                labels = [s.stem.replace("_sample", "").replace("___", " — ").replace("_", " ") for s in samples]
                choice = st.selectbox("Select verified benchmark field sample:", labels)
                idx = labels.index(choice)
                selected_image = Image.open(samples[idx])
                specimen_name = samples[idx].name

    with col_view:
        if selected_image:
            st.image(selected_image, caption=f"Active Specimen: {specimen_name}", use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#f8fafc; border:2px dashed #cbd5e1; border-radius:16px; padding:3.5rem 1.5rem; text-align:center; color:#94a3b8;">
                <div style="font-size:2.8rem; margin-bottom:0.6rem;">📷</div>
                <div style="font-size:0.95rem; font-weight:700; color:#475569;">No Specimen Loaded</div>
                <div style="font-size:0.8rem; margin-top:0.3rem;">Upload a photo or choose a benchmark sample on the left to begin.</div>
            </div>
            """, unsafe_allow_html=True)

    # Step 2: Location & Microclimate
    if selected_image:
        st.markdown("""
        <div class="step-box">
            <div class="step-circle">2</div>
            <div class="step-title">Farm Location & Live Microclimate Context</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="step-sub">Enter your farm city or click a quick-chip below. Live humidity, temperature, and wind are fused with visual diagnosis to measure spore dispersal risk.</div>', unsafe_allow_html=True)

        chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)
        with chip_col1:
            if st.button("📍 Amritsar", use_container_width=True):
                st.session_state["city_weather_input"] = "Amritsar"
        with chip_col2:
            if st.button("📍 Nashik", use_container_width=True):
                st.session_state["city_weather_input"] = "Nashik"
        with chip_col3:
            if st.button("📍 Ludhiana", use_container_width=True):
                st.session_state["city_weather_input"] = "Ludhiana"
        with chip_col4:
            if st.button("📍 Pune", use_container_width=True):
                st.session_state["city_weather_input"] = "Pune"
        with chip_col5:
            if st.button("📍 Delhi", use_container_width=True):
                st.session_state["city_weather_input"] = "Delhi"

        city_input = st.text_input(
            "Enter Farm City / Region:",
            placeholder="e.g. Amritsar, Nashik, Pune, Coimbatore, Jaipur",
            key="city_weather_input"
        )

        weather_data = None
        if city_input and city_input.strip():
            with st.spinner(f"🛰️ Fetching live satellite micro-climate for {city_input}..."):
                weather_data = fetch_live_weather(city_input.strip())

            if weather_data:
                st.success(f"📍 **Live Station Connected:** {weather_data['city']} ({weather_data['latitude']:.2f}°N, {weather_data['longitude']:.2f}°E)")
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">🌡️</div><div class="wx-title">Temperature</div><div class="wx-num">{weather_data["temperature"]}°C</div></div>', unsafe_allow_html=True)
                with w2:
                    st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">💧</div><div class="wx-title">Relative Humidity</div><div class="wx-num">{weather_data["humidity"]}%</div></div>', unsafe_allow_html=True)
                with w3:
                    st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">💨</div><div class="wx-title">Wind Velocity</div><div class="wx-num">{weather_data["wind_speed"]} km/h</div></div>', unsafe_allow_html=True)
                with w4:
                    st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">🌧️</div><div class="wx-title">Precipitation</div><div class="wx-num">{weather_data["precipitation"]} mm</div></div>', unsafe_allow_html=True)
            else:
                st.warning(f"Could not retrieve live station for '{city_input}'. Please check spelling.")

        # Step 3: Action Execution
        st.markdown("""
        <div class="step-box">
            <div class="step-circle">3</div>
            <div class="step-title">Execute Deep Learning Pathology Diagnostic</div>
        </div>
        """, unsafe_allow_html=True)

        btn_run = st.button("🚀 Run Precision Diagnostic & Lesion Analysis", type="primary", use_container_width=True)

        if btn_run:
            if not ai_engine:
                st.error("Model weights not initialized.")
            else:
                with st.spinner("Analyzing foliar structure, calculating convolutional saliency, and measuring lesion damage..."):
                    result = ai_engine.diagnose(selected_image, run_validation=use_foliar_gate)

                if not result.get("success"):
                    st.error(f"⚠️ **Foliar Quality Gate Notice:** {result.get('message')}")
                    st.info("The input photo does not exhibit sufficient vegetative tissue. Please submit a focused leaf photo.")
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

                    st.markdown("---")
                    st.markdown("### 📋 Clinical Diagnostic Assessment")

                    # Primary KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(f'<div class="agro-card"><div class="agro-label">Diagnosed Host Crop</div><div class="agro-val">{crop_name}</div><div class="agro-sub">Species Target</div></div>', unsafe_allow_html=True)
                    with k2:
                        st.markdown(f'<div class="agro-card"><div class="agro-label">Identified Pathology</div><div class="agro-val" style="font-size:1.15rem;">{disease_name}</div><div class="agro-sub">Primary Classification</div></div>', unsafe_allow_html=True)
                    with k3:
                        conf_color = "#059669" if confidence > 0.8 else ("#d97706" if confidence > 0.5 else "#dc2626")
                        st.markdown(f'<div class="agro-card"><div class="agro-label">Inference Confidence</div><div class="agro-val" style="color:{conf_color};">{confidence*100:.1f}%</div><div class="agro-sub">Softmax Certainty</div></div>', unsafe_allow_html=True)
                    with k4:
                        dmg_color = "#059669" if damage["percentage"] < 10 else ("#d97706" if damage["percentage"] < 30 else "#dc2626")
                        st.markdown(f'<div class="agro-card"><div class="agro-label">Foliar Necrotic Area</div><div class="agro-val" style="color:{dmg_color};">{damage["percentage"]}%</div><div class="agro-sub">{damage["stage"]}</div></div>', unsafe_allow_html=True)

                    # Severity Banner
                    st.markdown("<div style='margin:1rem 0;'>", unsafe_allow_html=True)
                    if status == "NOVEL_OR_MIXED_PATHOLOGY":
                        st.markdown('<span class="badge-pill badge-purple">⚠️ NOVEL / ATYPICAL PATHOLOGY DETECTED</span>', unsafe_allow_html=True)
                        st.warning(result["status_note"])
                    elif "healthy" in pred_class.lower():
                        st.markdown('<span class="badge-pill badge-green">✅ HEALTHY VIGOROUS FOLIAGE</span>', unsafe_allow_html=True)
                    elif severity.lower() in ["severe", "severe / critical", "high"]:
                        st.markdown(f'<span class="badge-pill badge-red">🚨 CRITICAL HIGH-RISK PATHOLOGY ({severity.upper()})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-pill badge-yellow">⚡ MODERATE SEVERITY PATHOLOGY ({severity.upper()})</span>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Climate Fusion Report
                    fusion_data = None
                    city_val = st.session_state.get("city_weather_input", "").strip()
                    if city_val:
                        wd = fetch_live_weather(city_val)
                        if wd:
                            fusion_data = analyze_climate_fusion(
                                pred_class, wd["temperature"], wd["humidity"], damage["percentage"]
                            )

                            st.markdown("---")
                            st.markdown("### 🌍 Climate-Fused Epidemiological Outbreak Risk")
                            st.caption(f"Calculated from live station data at **{wd['city']}** fused with visual foliar damage.")

                            risk = fusion_data["climatic_risk_index"]
                            risk_col = "#059669" if risk < 40 else ("#d97706" if risk < 65 else "#dc2626")
                            risk_txt = "LOW SPRAY URGENCY" if risk < 40 else ("MODERATE SPREAD RISK" if risk < 65 else "HIGH OUTBREAK RISK")

                            f1, f2 = st.columns(2, gap="large")
                            with f1:
                                st.markdown(f"""
                                <div class="agro-card" style="border-left: 5px solid {risk_col};">
                                    <div class="agro-label">Climate Risk Index</div>
                                    <div class="agro-val" style="color:{risk_col};">{risk}% &middot; {risk_txt}</div>
                                    <div style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">Current microclimate spore germination favorability</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric("Estimated Infection Onset", fusion_data["estimated_onset_time"])
                                st.metric("Biological Progression Stage", fusion_data["biological_stage"])
                            with f2:
                                st.info(fusion_data["climate_diagnosis"])
                                st.warning(f"⏰ **{fusion_data['action_urgency']}**")
                                st.markdown(f"**Pathogen Etiology:** {fusion_data['pathogen_type']} &middot; **Dispersal:** {fusion_data['spread_rate']}")

                            st.progress(min(risk / 100.0, 1.0))

                    # Explainable AI (XAI) Visualizer
                    st.markdown("---")
                    st.markdown("### 🔬 Multi-Layer Explainable AI (XAI) Visualizer")
                    st.caption("Inspect the exact visual evidence used by the convolutional network to identify lesions, along with the segmented necrotic tissue mask.")

                    tab_all, tab_orig, tab_heat, tab_mask = st.tabs([
                        "🖼️ 3-Way Comparative View",
                        "🍃 Original Leaf",
                        "🔥 Saliency Attention Heatmap",
                        "🎯 Segmented Lesion Mask"
                    ])

                    with tab_all:
                        v1, v2, v3 = st.columns(3)
                        with v1:
                            st.image(selected_image.resize((224, 224)), caption="1. Original Specimen", use_container_width=True)
                        with v2:
                            st.image(result["heatmap"], caption="2. Saliency Heatmap (Attention)", use_container_width=True)
                        with v3:
                            st.image(damage["mask"], caption=f"3. Lesion Mask ({damage['percentage']}%)", use_container_width=True)
                    with tab_orig:
                        st.image(selected_image.resize((380, 380)), caption="Original Leaf Specimen")
                    with tab_heat:
                        st.image(result["heatmap"].resize((380, 380)), caption="Convolutional Attention Heatmap (Warm colors highlight disease spots)")
                    with tab_mask:
                        st.image(damage["mask"].resize((380, 380)), caption=f"Segmented Lesion Mask — {damage['stage']}")

                    st.markdown(f"**Quantitative Foliar Damage Scale:** {damage['stage']} ({damage['percentage']}% Infected Surface Area)")
                    st.progress(min(damage['percentage'] / 100.0, 1.0))

                    # Top-5 Distribution
                    st.markdown("---")
                    st.markdown("#### 📊 Top Pathology Probability Distribution")
                    cands = result["candidates"]
                    cand_df = pd.DataFrame({
                        "Pathology Candidate": [c["class_name"].replace("___", " — ").replace("_", " ") for c in cands],
                        "Confidence Score (%)": [round(c["confidence"] * 100, 2) for c in cands]
                    })
                    st.bar_chart(cand_df.set_index("Pathology Candidate"))

                    # Agronomic Protocol
                    st.markdown("---")
                    st.markdown("### 📋 Agronomic Treatment & Spray Protocol")

                    t1, t2, t3, t4, t5 = st.tabs([
                        "🔬 Symptoms & Etiology",
                        "🍃 Organic Bio-Controls",
                        "🧪 Chemical Fungicides",
                        "🛡️ Cultural Prevention",
                        "📄 Download Clinical Rx"
                    ])

                    with t1:
                        st.markdown(f"**Pathogen Cause:** {remedy.get('cause', 'N/A')}")
                        st.markdown(f"**Field Symptoms:** {remedy.get('symptoms', 'N/A')}")
                        st.markdown(f"**Favorable Micro-Climate:** {remedy.get('favorable_conditions', 'N/A')}")

                    with t2:
                        for item in remedy.get("organic_remedies", []):
                            st.markdown(f"- ✅ **{item}**")
                        if not remedy.get("organic_remedies"):
                            st.info("No specific organic interventions indicated.")

                    with t3:
                        for chem in remedy.get("chemical_treatments", []):
                            st.markdown(f"- ⚠️ **{chem}**")
                        if remedy.get("chemical_treatments"):
                            st.caption("Adhere strictly to local agricultural label rates, observe Pre-Harvest Intervals (PHI), and rotate FRAC chemical groups.")
                        else:
                            st.info("No synthetic chemical treatments indicated.")

                    with t4:
                        for prev in remedy.get("prevention", []):
                            st.markdown(f"- 📌 **{prev}**")

                    with t5:
                        rx_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        wx_block = ""
                        if fusion_data and city_val:
                            wx_block = (
                                f"\nCLIMATE MICRO-CONTEXT ({city_val}):\n"
                                f"  Outbreak Risk    : {fusion_data['climatic_risk_index']}%\n"
                                f"  Estimated Onset  : {fusion_data['estimated_onset_time']}\n"
                                f"  Biological Stage : {fusion_data['biological_stage']}\n"
                                f"  Action Urgency   : {fusion_data['action_urgency']}\n"
                            )
                        rx_text = (
                            "=======================================================\n"
                            "🌱 AGROSCAN AI — CLINICAL CROP PATHOLOGY PRESCRIPTION\n"
                            "=======================================================\n"
                            f"Date of Analysis : {rx_date}\n"
                            f"Specimen ID      : {specimen_name}\n"
                            f"Host Crop        : {crop_name}\n"
                            f"Pathology        : {disease_name}\n"
                            f"Severity Index   : {severity} ({damage['stage']})\n"
                            f"Foliar Damage    : {damage['percentage']}%\n"
                            f"Inference Conf.  : {confidence*100:.1f}%\n"
                            f"Cause / Etiology : {remedy.get('cause', 'N/A')}\n"
                            + wx_block +
                            "\nRECOMMENDED MANAGEMENT PROTOCOL:\n"
                            "-------------------------------------------------------\n"
                            f"Organic Control  : {'; '.join(remedy.get('organic_remedies', ['Standard field hygiene']))}\n"
                            f"Chemical Sprays  : {'; '.join(remedy.get('chemical_treatments', ['None indicated']))}\n"
                            f"Cultural Steps   : {'; '.join(remedy.get('prevention', ['Maintain clean drip irrigation']))}\n\n"
                            "Certified by AgroScan AI MobileNetV2 Deep Learning Engine.\n"
                            "======================================================="
                        )
                        st.markdown(f'<div class="rx-card"><pre>{rx_text}</pre></div>', unsafe_allow_html=True)
                        st.download_button(
                            "📥 Download Clinical Agronomic Prescription (.txt)",
                            data=rx_text,
                            file_name=f"AgroScan_Prescription_{crop_name}_{disease_name.replace(' ','_')}.txt",
                            mime="text/plain"
                        )

                    # Continual Learning Feedback
                    st.markdown("---")
                    st.markdown("### 🤖 Active Learning & Model Self-Improvement")
                    st.caption("Help AgroScan AI learn! Confirm or correct this diagnosis so the AI adds it to its training memory and becomes more accurate.")

                    fb_col1, fb_col2 = st.columns([1, 1], gap="medium")
                    with fb_col1:
                        if st.button("✅ Confirm Diagnosis is Accurate (Save to Memory)", use_container_width=True):
                            rec = save_sample_feedback(
                                image=selected_image,
                                predicted_class=pred_class,
                                confidence=confidence,
                                user_confirmed=True,
                                city=city_val or "Unknown",
                                temperature=wd["temperature"] if 'wd' in locals() and wd else None,
                                humidity=wd["humidity"] if 'wd' in locals() and wd else None
                            )
                            st.success(f"Sample registered (ID: {rec['id']})! Stored in continual learning memory.")
                    with fb_col2:
                        with st.expander("✏️ Correct / Provide Alternate Ground Truth"):
                            correct_label = st.selectbox(
                                "Actual ground truth condition:",
                                options=class_labels,
                                index=class_labels.index(pred_class) if pred_class in class_labels else 0
                            )
                            if st.button("💾 Save Correction for Incremental Retraining"):
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
# MODULE 2: Continual Learning & Retrain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "🔄 Continual Learning & Retrain":
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">🔄</div>
        <div class="step-title">Continual Learning & Active Model Evolution</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Every confirmed or corrected specimen accumulates here. Fine-tune your deep learning model on new field data with one click to permanently increase prediction accuracy over time.</div>', unsafe_allow_html=True)

    stats = get_continual_learning_stats()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="agro-card"><div class="agro-label">Total Field Samples</div><div class="agro-val">{stats["total_samples"]}</div><div class="agro-sub">Accumulated</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="agro-card"><div class="agro-label">Pending Fine-Tuning</div><div class="agro-val" style="color:#d97706;">{stats["pending_retrain"]}</div><div class="agro-sub">New Data</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="agro-card"><div class="agro-label">Verified Correct</div><div class="agro-val" style="color:#059669;">{stats["confirmed_count"]}</div><div class="agro-sub">Ground Truth</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="agro-card"><div class="agro-label">Expert Corrections</div><div class="agro-val" style="color:#2563eb;">{stats["corrected_count"]}</div><div class="agro-sub">Hard Examples</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Execute Incremental Fine-Tuning")
    st.write("Loads the current MobileNetV2 weights, unfreezes top classification layers at an adaptive learning rate (1e-5), and trains on the new field samples. The enhanced weights are immediately saved back to disk.")

    f_col1, f_col2 = st.columns([1, 2], gap="large")
    with f_col1:
        epochs = st.slider("Fine-Tuning Epochs:", min_value=1, max_value=5, value=2)
        btn_fine_tune = st.button("🚀 Train Model on New Samples Now", type="primary", use_container_width=True)

        if btn_fine_tune:
            if stats["pending_retrain"] == 0:
                st.warning("⚠️ No pending new samples to learn from. Upload and diagnose leaves first, then click 'Confirm' or 'Correct'.")
            else:
                with st.spinner(f"Fine-tuning MobileNetV2 on {stats['pending_retrain']} new samples..."):
                    res = trigger_incremental_fine_tuning(epochs=epochs)
                if res.get("success"):
                    st.success(f"🎉 **{res['message']}**")
                    st.metric("Training Accuracy on New Data", f"{res['accuracy']}%")
                    st.metric("Final Loss", f"{res['loss']}")
                    st.cache_resource.clear()
                else:
                    st.error(f"Error: {res.get('message')}")

    with f_col2:
        st.markdown("#### 📋 Field Registry Log")
        if stats["records"]:
            rec_df = pd.DataFrame([
                {
                    "Sample ID": r["id"],
                    "Timestamp": r["timestamp"][:16].replace("T", " "),
                    "AI Prediction": r["predicted_class"].replace("___", " - "),
                    "Ground Truth": r["ground_truth_label"].replace("___", " - "),
                    "Learning Status": "Trained" if r.get("used_in_retraining") else "Pending",
                    "Location": r.get("city", "Unknown")
                }
                for r in reversed(stats["records"])
            ])
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("No field samples logged yet. Diagnose a leaf and confirm/correct it to begin building the continual dataset.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3: Spray Calculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "💊 Spray & Tank Calculator":
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">💊</div>
        <div class="step-title">Precision Chemical Dosage & Tank Mixing Calculator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Calculate exact water volumes, chemical active ingredient dosage, and knapsack tank refills tailored to your plot size.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        unit = st.selectbox("Land Area Unit:", ["Acres", "Hectares"])
        area = st.number_input(f"Farm Plot Size (in {unit}):", min_value=0.25, max_value=500.0, value=2.0, step=0.5)
        target_crop = st.selectbox("Target Host Crop:", ["Potato", "Tomato", "Corn (Maize)", "Apple", "Grape", "Pepper"])
        fungicide = st.selectbox("Fungicide Formulation:", [
            "Mancozeb 75% WP (Contact Protective) — 2.5 kg/ha",
            "Chlorothalonil 720 SC (Broad Spectrum) — 2.0 L/ha",
            "Copper Hydroxide 77% WP (Bactericide / Fungicide) — 2.0 kg/ha",
            "Metalaxyl + Mancozeb (Ridomil MZ Systemic) — 2.5 kg/ha",
            "Azoxystrobin 23% SC (Strobilurin translaminar) — 500 ml/ha"
        ])
        sprayer = st.selectbox("Sprayer Hardware:", ["Manual Knapsack (16 Liters)", "Motorized Knapsack (20 Liters)", "Tractor Boom Sprayer (400 Liters)"])

    hectares = area if unit == "Hectares" else area / 2.471
    water = round(hectares * 500, 1)

    if "Mancozeb" in fungicide:
        chem_desc = f"{round(hectares * 2.5, 2)} kg Mancozeb 75 WP"
    elif "Chlorothalonil" in fungicide:
        chem_desc = f"{round(hectares * 2.0, 2)} Liters Chlorothalonil 720 SC"
    elif "Azoxystrobin" in fungicide:
        chem_desc = f"{round(hectares * 500, 1)} ml Azoxystrobin SC"
    else:
        chem_desc = f"{round(hectares * 2.0, 2)} kg formulation"

    cap = 16 if "16" in sprayer else (20 if "20" in sprayer else 400)
    refills = int(np.ceil(water / cap))

    with c2:
        st.markdown(f"""
        <div class="agro-card" style="border-left: 5px solid #059669; margin-bottom: 1rem;">
            <div class="agro-label">Total Spray Water Volume</div>
            <div class="agro-val" style="color:#059669;">{water} Liters</div>
            <div class="agro-sub">Calibrated for {area} {unit} of {target_crop}</div>
        </div>
        <div class="agro-card" style="border-left: 5px solid #0891b2; margin-bottom: 1rem;">
            <div class="agro-label">Total Active Chemical Dosage</div>
            <div class="agro-val" style="color:#0891b2;">{chem_desc}</div>
            <div class="agro-sub">Recommended concentration: ~2.5–3g per Liter</div>
        </div>
        <div class="agro-card" style="border-left: 5px solid #d97706;">
            <div class="agro-label">Sprayer Tank Refills Required</div>
            <div class="agro-val" style="color:#d97706;">{refills} Tanks ({cap}L each)</div>
            <div class="agro-sub">Add ~{round((water / refills) * 0.005 * 1000, 1)}g chemical per refill load</div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Weather Advisory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "🌦️ Live Weather Advisory":
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">🌦️</div>
        <div class="step-title">Agro-Climatic Spray Safety & Weather Window</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Fungicide and bio-pesticide efficacy depends heavily on wind drift, foliage moisture, and rainfast windows. Enter your city for live field alerts.</div>', unsafe_allow_html=True)

    spray_city = st.text_input("Enter farm city for live microclimate advisory:", placeholder="e.g. Ludhiana, Nashik, Jaipur, Coimbatore")

    live_w = None
    if spray_city and spray_city.strip():
        with st.spinner(f"Fetching station data for {spray_city}..."):
            live_w = fetch_live_weather(spray_city.strip())
        if not live_w:
            st.warning(f"Could not connect to station for '{spray_city}'. Showing default advisory.")

    temp = live_w["temperature"] if live_w else 23.5
    wind = live_w["wind_speed"] if live_w else 5.2
    hum = live_w["humidity"] if live_w else 60.0
    prec = live_w["precipitation"] if live_w else 0.0

    if live_w:
        st.success(f"📍 **Station Active:** {live_w['city']} &middot; Verified live")

    t_ok = 18 <= temp <= 28
    t_col = "#059669" if t_ok else "#d97706"
    t_st = "Optimal (18–28°C)" if t_ok else ("Too Hot — Spray at dawn" if temp > 28 else "Too Cold — Delay spray")

    w_ok = wind <= 10
    w_col = "#059669" if w_ok else ("#d97706" if wind <= 20 else "#dc2626")
    w_st = "Safe — Low Drift" if w_ok else ("Moderate Drift Risk" if wind <= 20 else "Stop — High Drift")

    h_col = "#d97706" if hum >= 70 else "#059669"
    h_st = "High Outbreak Risk" if hum >= 70 else ("Low Spore Pressure" if hum < 55 else "Watch Closely")

    p_ok = prec < 1.0
    p_col = "#059669" if p_ok else "#dc2626"
    p_st = "Safe Rainfast Window" if p_ok else "Rain Washing Out Chemicals"

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">🌡️</div><div class="wx-title">Temperature</div><div class="wx-num" style="color:{t_col};">{temp}°C</div><div style="font-size:0.75rem; color:{t_col}; font-weight:600;">{t_st}</div></div>', unsafe_allow_html=True)
    with w2:
        st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">💨</div><div class="wx-title">Wind Speed</div><div class="wx-num" style="color:{w_col};">{wind} km/h</div><div style="font-size:0.75rem; color:{w_col}; font-weight:600;">{w_st}</div></div>', unsafe_allow_html=True)
    with w3:
        st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">💧</div><div class="wx-title">Humidity</div><div class="wx-num" style="color:{h_col};">{hum}%</div><div style="font-size:0.75rem; color:{h_col}; font-weight:600;">{h_st}</div></div>', unsafe_allow_html=True)
    with w4:
        st.markdown(f'<div class="wx-ribbon"><div class="wx-icon">🌧️</div><div class="wx-title">Precipitation</div><div class="wx-num" style="color:{p_col};">{prec} mm</div><div style="font-size:0.75rem; color:{p_col}; font-weight:600;">{p_st}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    safe = (wind <= 15 and prec < 1.0 and 15 <= temp <= 32)
    if safe:
        st.success("🟢 **OVERALL SPRAY VERDICT: EXCELLENT APPLICATION WINDOW**\n\nWind drift risk is minimal and foliage has a clean rainfast window. Optimal time for contact or systemic fungicide application. Prefer early morning (6–9 AM) or late evening (5–7 PM).")
    elif prec >= 1.0:
        st.error("🔴 **SPRAY VERDICT: DELAY SPRAY — PRECIPITATION ACTIVE**\n\nActive rainfall will wash off chemical active ingredients. Wait for foliage to dry with a 2+ hour rainfast safety window.")
    elif wind > 20:
        st.error("🔴 **SPRAY VERDICT: STOP — HIGH WIND DRIFT HAZARD**\n\nWind exceeds safe limits (>20 km/h). Severe risk of off-target drift and chemical wastage.")
    else:
        st.warning("🟡 **SPRAY VERDICT: USE CAUTION**\n\nMarginal microclimate. Use coarse droplet nozzles and anti-drift adjuvants.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5: Model Performance & XAI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "📊 Model Performance & XAI":
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">📊</div>
        <div class="step-title">Deep Learning Architecture & Quantitative Validation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Rigorous empirical validation benchmarks for the trained MobileNetV2 Transfer Learning Model across all 38 categories.</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="agro-card"><div class="agro-label">Base Architecture</div><div class="agro-val" style="font-size:1.2rem;">MobileNetV2</div><div class="agro-sub">Depthwise Convolutions</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="agro-card"><div class="agro-label">Total Parameters</div><div class="agro-val">2.60M</div><div class="agro-sub">Lightweight Mobile Engine</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="agro-card"><div class="agro-label">Top-1 Accuracy</div><div class="agro-val" style="color:#059669;">89.21%</div><div class="agro-sub">Holdout Validation</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="agro-card"><div class="agro-label">Top-3 Accuracy</div><div class="agro-val" style="color:#059669;">98.29%</div><div class="agro-sub">Clinical Differential</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    cc, cm = st.columns(2, gap="medium")
    with cc:
        st.markdown("#### 📉 Training Loss & Accuracy Convergence Curves")
        if TRAINING_CURVES_PATH.exists():
            st.image(str(TRAINING_CURVES_PATH), use_container_width=True)
        else:
            st.info("Training curves will appear after running train pipeline.")
    with cm:
        st.markdown("#### 🎯 38-Class Normalized Confusion Matrix Heatmap")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
        else:
            st.info("Confusion matrix available in reports directory.")

    st.markdown("---")
    st.markdown("#### 📋 Detailed Per-Class Precision, Recall & F1-Scores")
    if EVALUATION_REPORT_PATH.exists():
        df = pd.read_csv(EVALUATION_REPORT_PATH, index_col=0)
        st.dataframe(
            df.style.format("{:.2f}").background_gradient(cmap="YlGnBu", subset=["precision", "recall", "f1-score"]),
            use_container_width=True
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 6: 38-Class Knowledge Base
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "📖 38-Class Knowledge Base":
    st.markdown("""
    <div class="step-box">
        <div class="step-circle">📖</div>
        <div class="step-title">Comprehensive Agricultural Pathology Encyclopedia</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="step-sub">Browse etiology, visual symptoms, organic biocontrols, and registered chemical fungicides across all 38 categories.</div>', unsafe_allow_html=True)

    db = load_disease_database()
    crops = sorted(set(v["crop"] for v in db.values()))
    filt = st.selectbox("Filter by Host Crop:", ["All Crops"] + crops)
    items = {k: v for k, v in db.items() if filt == "All Crops" or v["crop"] == filt}

    for key, d in items.items():
        sev_badge = "badge-red" if d["severity"].lower() in ["severe", "high", "severe / critical"] else ("badge-yellow" if d["severity"].lower() == "moderate" else "badge-green")
        with st.expander(f"🌿 {d['crop']} — {d['disease_name']}"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f'<span class="badge-pill {sev_badge}">{d["severity"].upper()} SEVERITY</span>', unsafe_allow_html=True)
                st.markdown(f"\n**Pathogen Cause:** {d['cause']}")
                st.markdown(f"**Diagnostic Symptoms:** {d['symptoms']}")
                st.markdown(f"**Favorable Micro-Climate:** {d['favorable_conditions']}")
            with ec2:
                st.markdown("**🍃 Sustainable Bio-Controls:**")
                for o in d.get("organic_remedies", []):
                    st.markdown(f"- {o}")
                st.markdown("**🧪 Chemical Formulations:**")
                for c in d.get("chemical_treatments", []):
                    st.markdown(f"- {c}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="app-footer">
    <strong>AgroScan AI v2.5</strong> &middot; Precision Crop Pathology & Climate-Fused Agronomic Suite<br>
    Deep Learning Architecture: MobileNetV2 (38 Classes) &middot; Open-Meteo Integration &middot; CSE276 Project
</div>
""", unsafe_allow_html=True)
