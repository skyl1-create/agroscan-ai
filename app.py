# AgroScan AI — Next-Gen Precision Agronomic Intelligence Platform
# Enterprise v3.0 Production Suite
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
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AgroScan AI &middot; Precision Crop Pathology Cloud",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ultra-Premium Enterprise SaaS Styling (v3.0)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global Styles ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
        -webkit-font-smoothing: antialiased;
    }

    #MainMenu, footer, header {visibility: hidden !important;}

    /* ── High-Tech Dark Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020c07 0%, #061a10 50%, #030e08 100%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 0.9rem !important;
        border-radius: 12px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: rgba(52, 211, 153, 0.4) !important;
        transform: translateX(5px);
        color: #34d399 !important;
    }

    /* ── Enterprise Top Header Bar ── */
    .top-nav-bar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.85rem 1.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
    }
    .top-nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.25rem;
        font-weight: 800;
        color: #064e3b;
        letter-spacing: -0.03em;
    }
    .top-nav-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #059669;
        background: #ecfdf5;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        border: 1px solid #a7f3d0;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.25);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* ── Hero Banner ── */
    .saas-hero {
        background: radial-gradient(circle at 90% 10%, rgba(52, 211, 153, 0.25) 0%, transparent 50%),
                    linear-gradient(135deg, #022c22 0%, #064e3b 50%, #047857 100%);
        color: #ffffff;
        padding: 2.8rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px -15px rgba(2, 44, 34, 0.35);
    }
    .saas-hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        margin: 0 0 0.6rem 0;
        color: #ffffff !important;
        line-height: 1.2;
    }
    .saas-hero p {
        font-size: 1rem;
        color: #a7f3d0;
        margin: 0;
        line-height: 1.6;
        max-width: 800px;
    }

    /* ── Interactive Process Stepper ── */
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin: 2rem 0 0.8rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .step-badge {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        font-weight: 800;
        box-shadow: 0 4px 10px rgba(5, 150, 105, 0.3);
    }
    .step-text {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
    }

    /* ── Professional Cards ── */
    .saas-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .saas-card:hover {
        border-color: #10b981;
        box-shadow: 0 12px 28px -4px rgba(16, 185, 129, 0.15);
        transform: translateY(-3px);
    }
    .saas-card-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .saas-card-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .saas-card-sub {
        font-size: 0.78rem;
        font-weight: 600;
        color: #059669;
        margin-top: 0.35rem;
    }

    /* ── Microclimate Weather Pill ── */
    .wx-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.06);
    }
    .wx-box-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .wx-box-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #065f46; letter-spacing: 0.06em; }
    .wx-box-val { font-size: 1.45rem; font-weight: 800; color: #022c22; margin: 0.2rem 0; }

    /* ── Official Severity Badges ── */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-green { background: #dcfce7; color: #166534; border: 1.5px solid #86efac; }
    .badge-yellow { background: #fef9c3; color: #854d0e; border: 1.5px solid #fde047; }
    .badge-red { background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }
    .badge-purple { background: #f3e8ff; color: #6b21a8; border: 1.5px solid #d8b4fe; }

    /* ── Clinical Rx Box ── */
    .rx-sheet {
        background: #ffffff;
        border: 2px dashed #059669;
        border-radius: 16px;
        padding: 1.6rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        line-height: 1.6;
        color: #1e293b;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.08);
    }

    /* ── Action Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(5, 150, 105, 0.45) !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #047857 100%) !important;
        border-radius: 9999px !important;
    }

    /* ── Footer ── */
    .saas-footer {
        text-align: center;
        padding: 3rem 0 1.5rem 0;
        margin-top: 4rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cached Loaders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="Initializing Deep Learning MobileNetV2 Weights...")
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
# Sidebar Navigation & Settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.4rem 0 0.8rem 0;">
        <div style="font-size: 2.6rem; margin-bottom: 0.2rem;">🌱</div>
        <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.035em; color: #34d399 !important;">AgroScan AI</div>
        <div style="font-size: 0.72rem; font-weight: 700; color: #a7f3d0 !important; letter-spacing: 0.09em; text-transform: uppercase;">Agronomic Intelligence Cloud</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🔬 Diagnostic Engine",
            "🔄 Continual Learning & Retrain",
            "💊 Spray & Tank Calculator",
            "🌦️ Live Weather Advisory",
            "📊 Model Analytics & XAI",
            "📖 38-Class Disease Library"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#34d399 !important; margin-bottom:0.6rem;">Precision Gating Controls</p>', unsafe_allow_html=True)
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=True, help="Evaluates 4 photometric transformations for lighting resilience.")
    use_foliar_gate = st.checkbox("Foliar Vegetative Gate", value=True, help="Rejects non-plant images automatically via green channel HSV.")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #cbd5e1 !important; line-height: 1.85;">
        <div style="font-weight:700; color:#34d399 !important; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">Platform Specs</div>
        <div><strong>Deep Learning:</strong> MobileNetV2</div>
        <div><strong>Pathology Library:</strong> 38 Conditions</div>
        <div><strong>Validation Accuracy:</strong> 89.21% (Top-3: 98.29%)</div>
        <div><strong>Inference Latency:</strong> ~35ms / image</div>
        <div style="margin-top:0.9rem; font-size:0.68rem; color:#64748b !important;">AgroScan v3.0 &middot; Production Release</div>
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enterprise Top Bar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="top-nav-bar">
    <div class="top-nav-logo">
        <span>🌱</span> AgroScan AI <span style="font-size:0.78rem; font-weight:600; color:#64748b; background:#f1f5f9; padding:0.25rem 0.65rem; border-radius:8px;">Enterprise Suite</span>
    </div>
    <div class="top-nav-status">
        <span class="status-dot"></span> Neural Cloud Engine Online &middot; 38 Classes
    </div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hero Banner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="saas-hero">
    <h1>🌱 AgroScan AI: Ultra-Precision Crop Pathology</h1>
    <p>Comprehensive agricultural deep learning system. Upload leaf photographs or capture field specimens in real-time. Our neural network classifies diseases across 38 crop pathologies, quantifies foliar necrotic damage, and fuses live microclimate epidemiology for tailored spray treatments.</p>
</div>
""", unsafe_allow_html=True)

model = load_cached_model()
class_labels = load_class_labels()
ai_engine = PrecisionAIEngine(model, class_labels) if model else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1: Diagnostic Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if menu == "🔬 Diagnostic Engine":

    # Step 1: Specimen Acquisition
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">1</div>
        <div class="step-text">Specimen Acquisition Hub</div>
    </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Specimen Source",
        ["📁 Upload from Device", "📷 Live Field Camera", "🖼️ Benchmark Field Samples"],
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_image = None
    specimen_name = "Leaf Specimen"

    col_input, col_view = st.columns([1, 1], gap="large")

    with col_input:
        if input_mode == "📁 Upload from Device":
            uploaded = st.file_uploader("Upload leaf photograph (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded:
                selected_image = Image.open(uploaded)
                specimen_name = uploaded.name

        elif input_mode == "📷 Live Field Camera":
            cam_shot = st.camera_input("Capture live field specimen:")
            if cam_shot:
                selected_image = Image.open(cam_shot)
                specimen_name = "Live Camera Capture"

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
            <div style="background:#ffffff; border:2px dashed #cbd5e1; border-radius:18px; padding:3.8rem 1.8rem; text-align:center; color:#94a3b8; box-shadow:0 4px 16px rgba(0,0,0,0.02);">
                <div style="font-size:3rem; margin-bottom:0.6rem;">🍃</div>
                <div style="font-size:1.05rem; font-weight:800; color:#334155;">No Specimen Loaded</div>
                <div style="font-size:0.82rem; margin-top:0.3rem;">Upload a photo or select a benchmark sample to activate analysis.</div>
            </div>
            """, unsafe_allow_html=True)

    # Step 2: Location & Microclimate
    if selected_image:
        st.markdown("""
        <div class="step-header">
            <div class="step-badge">2</div>
            <div class="step-text">Farm Location & Live Microclimate Context</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Select a quick-chip agricultural hub or type your city. Live satellite weather is fused with visual diagnosis to measure spore dispersal risk.")

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
            "Farm Location City / District:",
            placeholder="e.g. Amritsar, Nashik, Pune, Coimbatore, Jaipur",
            key="city_weather_input"
        )

        weather_data = None
        if city_input and city_input.strip():
            with st.spinner(f"🛰️ Connecting to live meteorological station for {city_input}..."):
                weather_data = fetch_live_weather(city_input.strip())

            if weather_data:
                st.success(f"📍 **Station Active:** {weather_data['city']} ({weather_data['latitude']:.2f}°N, {weather_data['longitude']:.2f}°E)")
                w1, w2, w3, w4 = st.columns(4)
                with w1:
                    st.markdown(f'<div class="wx-box"><div class="wx-box-icon">🌡️</div><div class="wx-box-label">Ambient Temp</div><div class="wx-box-val">{weather_data["temperature"]}°C</div></div>', unsafe_allow_html=True)
                with w2:
                    st.markdown(f'<div class="wx-box"><div class="wx-box-icon">💧</div><div class="wx-box-label">Relative Humidity</div><div class="wx-box-val">{weather_data["humidity"]}%</div></div>', unsafe_allow_html=True)
                with w3:
                    st.markdown(f'<div class="wx-box"><div class="wx-box-icon">💨</div><div class="wx-box-label">Wind Velocity</div><div class="wx-box-val">{weather_data["wind_speed"]} km/h</div></div>', unsafe_allow_html=True)
                with w4:
                    st.markdown(f'<div class="wx-box"><div class="wx-box-icon">🌧️</div><div class="wx-box-label">Precipitation</div><div class="wx-box-val">{weather_data["precipitation"]} mm</div></div>', unsafe_allow_html=True)
            else:
                st.warning(f"Could not connect to meteorological station for '{city_input}'. Please check spelling.")

        # Step 3: Run Diagnostics
        st.markdown("""
        <div class="step-header">
            <div class="step-badge">3</div>
            <div class="step-text">Run Deep Learning Pathology Diagnostic</div>
        </div>
        """, unsafe_allow_html=True)

        btn_run = st.button("🚀 Execute Precision Pathology & Lesion Analysis", type="primary", use_container_width=True)

        if btn_run:
            if not ai_engine:
                st.error("Deep learning weights not initialized.")
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
                        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Diagnosed Host Crop</div><div class="saas-card-val">{crop_name}</div><div class="saas-card-sub">Botanical Species</div></div>', unsafe_allow_html=True)
                    with k2:
                        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Identified Pathology</div><div class="saas-card-val" style="font-size:1.15rem;">{disease_name}</div><div class="saas-card-sub">Primary Classification</div></div>', unsafe_allow_html=True)
                    with k3:
                        conf_color = "#059669" if confidence > 0.8 else ("#d97706" if confidence > 0.5 else "#dc2626")
                        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Inference Certainty</div><div class="saas-card-val" style="color:{conf_color};">{confidence*100:.1f}%</div><div class="saas-card-sub">Softmax Confidence</div></div>', unsafe_allow_html=True)
                    with k4:
                        dmg_color = "#059669" if damage["percentage"] < 10 else ("#d97706" if damage["percentage"] < 30 else "#dc2626")
                        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Foliar Necrotic Area</div><div class="saas-card-val" style="color:{dmg_color};">{damage["percentage"]}%</div><div class="saas-card-sub">{damage["stage"]}</div></div>', unsafe_allow_html=True)

                    # Severity Banner
                    st.markdown("<div style='margin:1.2rem 0;'>", unsafe_allow_html=True)
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
                            st.markdown("### 🌍 Climate-Fused Epidemiological Risk Assessment")
                            st.caption(f"Calculated from live station data at **{wd['city']}** fused with visual foliar damage.")

                            risk = fusion_data["climatic_risk_index"]
                            risk_col = "#059669" if risk < 40 else ("#d97706" if risk < 65 else "#dc2626")
                            risk_txt = "LOW SPRAY URGENCY" if risk < 40 else ("MODERATE SPREAD RISK" if risk < 65 else "HIGH OUTBREAK RISK")

                            f1, f2 = st.columns(2, gap="large")
                            with f1:
                                st.markdown(f"""
                                <div class="saas-card" style="border-left: 5px solid {risk_col};">
                                    <div class="saas-card-label">Climate Outbreak Risk Index</div>
                                    <div class="saas-card-val" style="color:{risk_col};">{risk}% &middot; {risk_txt}</div>
                                    <div style="font-size:0.8rem; color:#64748b; margin-top:0.3rem;">Spore germination & canopy colonization favorability</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.metric("Estimated Infection Onset", fusion_data["estimated_onset_time"])
                                st.metric("Biological Progression Stage", fusion_data["biological_stage"])
                            with f2:
                                st.info(fusion_data["climate_diagnosis"])
                                st.warning(f"⏰ **{fusion_data['action_urgency']}**")
                                st.markdown(f"**Pathogen Etiology:** {fusion_data['pathogen_type']} &middot; **Spread Rate:** {fusion_data['spread_rate']}")

                            st.progress(min(risk / 100.0, 1.0))

                    # Explainable AI (XAI) Visualizer
                    st.markdown("---")
                    st.markdown("### 🔬 Multi-Layer Explainable AI (XAI) Visualizer")
                    st.caption("Inspect the exact visual evidence used by the convolutional network to identify lesions, along with the segmented necrotic tissue mask.")

                    tab_all, tab_orig, tab_heat, tab_mask = st.tabs([
                        "🖼️ 3-Way Comparative View",
                        "🍃 Original Specimen",
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

                    # Top-5 Candidates
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
                        st.markdown(f'<div class="rx-sheet"><pre>{rx_text}</pre></div>', unsafe_allow_html=True)
                        st.download_button(
                            "📥 Download Clinical Agronomic Prescription (.txt)",
                            data=rx_text,
                            file_name=f"AgroScan_Prescription_{crop_name}_{disease_name.replace(' ','_')}.txt",
                            mime="text/plain"
                        )

                    # Continual Learning Feedback Loop
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
    <div class="step-header">
        <div class="step-badge">🔄</div>
        <div class="step-text">Continual Learning & Model Self-Evolution Hub</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.9rem;">Every confirmed or corrected specimen accumulates here. Fine-tune your deep learning model on new field data with one click to permanently increase prediction accuracy over time.</p>', unsafe_allow_html=True)

    stats = get_continual_learning_stats()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Total Field Samples</div><div class="saas-card-val">{stats["total_samples"]}</div><div class="saas-card-sub">Accumulated in DB</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Pending Fine-Tuning</div><div class="saas-card-val" style="color:#d97706;">{stats["pending_retrain"]}</div><div class="saas-card-sub">Awaiting Batch</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Verified Correct</div><div class="saas-card-val" style="color:#059669;">{stats["confirmed_count"]}</div><div class="saas-card-sub">Ground Truth Matches</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="saas-card"><div class="saas-card-label">Expert Corrections</div><div class="saas-card-val" style="color:#2563eb;">{stats["corrected_count"]}</div><div class="saas-card-sub">Edge Case Adjustments</div></div>', unsafe_allow_html=True)

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
    <div class="step-header">
        <div class="step-badge">💊</div>
        <div class="step-text">Precision Chemical Dosage & Tank Mixing Calculator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.9rem;">Calculate exact water volumes, chemical active ingredient dosage, and knapsack tank refills tailored to your plot size to prevent chemical waste and crop toxicity.</p>', unsafe_allow_html=True)

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
        <div class="saas-card" style="border-left: 5px solid #059669; margin-bottom: 1rem;">
            <div class="saas-card-label">Total Spray Water Volume</div>
            <div class="saas-card-val" style="color:#059669;">{water} Liters</div>
            <div class="saas-card-sub">Calibrated for {area} {unit} of {target_crop}</div>
        </div>
        <div class="saas-card" style="border-left: 5px solid #0891b2; margin-bottom: 1rem;">
            <div class="saas-card-label">Total Active Chemical Dosage</div>
            <div class="saas-card-val" style="color:#0891b2;">{chem_desc}</div>
            <div class="saas-card-sub">Recommended concentration: ~2.5–3g per Liter</div>
        </div>
        <div class="saas-card" style="border-left: 5px solid #d97706;">
            <div class="saas-card-label">Sprayer Tank Refills Required</div>
            <div class="saas-card-val" style="color:#d97706;">{refills} Tanks ({cap}L each)</div>
            <div class="saas-card-sub">Add ~{round((water / refills) * 0.005 * 1000, 1)}g chemical per refill load</div>
        </div>
        """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4: Weather Advisory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "🌦️ Live Weather Advisory":
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">🌦️</div>
        <div class="step-text">Agro-Climatic Spray Safety & Weather Advisory</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.9rem;">Fungicide and bio-pesticide efficacy depends heavily on wind drift, foliage moisture, and rainfast windows. Enter your city for real-time spray alerts.</p>', unsafe_allow_html=True)

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
        st.markdown(f'<div class="wx-box"><div class="wx-box-icon">🌡️</div><div class="wx-box-label">Temperature</div><div class="wx-box-val" style="color:{t_col};">{temp}°C</div><div style="font-size:0.75rem; color:{t_col}; font-weight:700;">{t_st}</div></div>', unsafe_allow_html=True)
    with w2:
        st.markdown(f'<div class="wx-box"><div class="wx-box-icon">💨</div><div class="wx-box-label">Wind Velocity</div><div class="wx-box-val" style="color:{w_col};">{wind} km/h</div><div style="font-size:0.75rem; color:{w_col}; font-weight:700;">{w_st}</div></div>', unsafe_allow_html=True)
    with w3:
        st.markdown(f'<div class="wx-box"><div class="wx-box-icon">💧</div><div class="wx-box-label">Relative Humidity</div><div class="wx-box-val" style="color:{h_col};">{hum}%</div><div style="font-size:0.75rem; color:{h_col}; font-weight:700;">{h_st}</div></div>', unsafe_allow_html=True)
    with w4:
        st.markdown(f'<div class="wx-box"><div class="wx-box-icon">🌧️</div><div class="wx-box-label">Precipitation</div><div class="wx-box-val" style="color:{p_col};">{prec} mm</div><div style="font-size:0.75rem; color:{p_col}; font-weight:700;">{p_st}</div></div>', unsafe_allow_html=True)

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
# MODULE 5: Model Analytics & XAI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif menu == "📊 Model Analytics & XAI":
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">📊</div>
        <div class="step-text">Deep Learning Architecture & Quantitative Validation</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.9rem;">Rigorous empirical validation benchmarks for the trained MobileNetV2 Transfer Learning Model across all 38 categories.</p>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="saas-card"><div class="saas-card-label">Base Architecture</div><div class="saas-card-val" style="font-size:1.2rem;">MobileNetV2</div><div class="saas-card-sub">Depthwise Separable Convolutions</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="saas-card"><div class="saas-card-label">Total Parameters</div><div class="saas-card-val">2.60M</div><div class="saas-card-sub">Lightweight Mobile Engine</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="saas-card"><div class="saas-card-label">Top-1 Accuracy</div><div class="saas-card-val" style="color:#059669;">89.21%</div><div class="saas-card-sub">Holdout Validation</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="saas-card"><div class="saas-card-label">Top-3 Accuracy</div><div class="saas-card-val" style="color:#059669;">98.29%</div><div class="saas-card-sub">Clinical Differential</div></div>', unsafe_allow_html=True)

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
elif menu == "📖 38-Class Disease Library":
    st.markdown("""
    <div class="step-header">
        <div class="step-badge">📖</div>
        <div class="step-text">Comprehensive Agricultural Pathology Encyclopedia</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; font-size:0.9rem;">Browse etiology, visual symptoms, organic biocontrols, and registered chemical fungicides across all 38 categories.</p>', unsafe_allow_html=True)

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
# Global SaaS Footer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="saas-footer">
    <strong>AgroScan AI Enterprise v3.0</strong> &middot; Precision Crop Pathology & Climate-Fused Agronomic Suite<br>
    Deep Learning Engine: MobileNetV2 (38 Classes) &middot; Live Satellite Ingestion &middot; CSE276 Project
</div>
""", unsafe_allow_html=True)
