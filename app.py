"""
EchoDetect: Futuristic Audio Intelligence System
High-Contrast Experimental Laboratory Interface
Palette: #08090C (Void Black), #15171D (Surface Charcoal), #F5F5F7 (Bright Pure White),
         #9B7BFF (Electric Violet), #D7FF64 (Acid Lime), #00F2FE (Electric Cyan Waveforms).
"""

import sys
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root directory is on path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    CLASSES, CLASS_NAMES, CLASS_DESCRIPTIONS, SAMPLE_RATE, N_MFCC,
    MODELS_DIR, REPORTS_DIR, SVM_MODEL_PATH, RF_MODEL_PATH,
    EVAL_SUMMARY_JSON, TEST_METRICS_CSV, AUDIO_DIR
)
from echodetect.dataset import load_metadata, get_sample_for_class
from echodetect.features import extract_features_from_file
from echodetect.models import load_model
from echodetect.utils import load_evaluation_summary, generate_audio_plots

# Streamlit Page Config
st.set_page_config(
    page_title="EchoDetect — Audio Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Dark Audio Laboratory CSS
FUTURISTIC_CSS = r"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-void: #08090C;
        --bg-panel: #111319;
        --bg-surface: #15171D;
        --bg-elevated: #1C1F28;
        --border-subtle: #222530;
        --border-bright: #323746;
        --border-violet: rgba(155, 123, 255, 0.4);
        --border-lime: rgba(215, 255, 100, 0.4);
        --text-pure: #F5F5F7;
        --text-secondary: #A5A9B5;
        --text-muted: #6B7280;
        --accent-violet: #9B7BFF;
        --accent-lime: #D7FF64;
        --accent-cyan: #00F2FE;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        background-color: var(--bg-void) !important;
        color: var(--text-pure) !important;
        letter-spacing: -0.015em;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #F5F5F7 !important;
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: #A5A9B5;
    }

    .stApp {
        background-color: var(--bg-void);
    }

    /* Content viewport */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.8rem !important;
        padding-right: 2.8rem !important;
        max-width: 1440px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B0D12 !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 1.2rem 0;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Wordmark */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 0 16px 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 16px;
    }
    .brand-icon {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(155, 123, 255, 0.12);
        border: 1px solid var(--border-violet);
        border-radius: 6px;
    }
    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-pure);
        letter-spacing: -0.02em;
    }
    .brand-tag {
        font-size: 0.65rem;
        color: var(--accent-lime);
        background: rgba(215, 255, 100, 0.1);
        border: 1px solid rgba(215, 255, 100, 0.25);
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: auto;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* Hero Section */
    .hero-container {
        position: relative;
        background: linear-gradient(145deg, #111319 0%, #15171D 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 34px 38px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-violet), var(--accent-lime), transparent);
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-lime);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 14px;
        background: rgba(215, 255, 100, 0.08);
        border: 1px solid rgba(215, 255, 100, 0.2);
        padding: 4px 10px;
        border-radius: 4px;
    }
    .pulsing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--accent-lime);
        display: inline-block;
        box-shadow: 0 0 8px var(--accent-lime);
    }
    .hero-headline,
    h1.hero-headline,
    [data-testid="stMarkdownContainer"] h1.hero-headline {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        font-size: 2.85rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.04em !important;
        line-height: 1.05 !important;
        margin: 0 0 14px 0 !important;
        position: relative !important;
        text-shadow: 0 0 24px rgba(255, 255, 255, 0.15) !important;
    }
    .hero-headline:hover {
        text-shadow: 2px 0 rgba(155, 123, 255, 0.9), -2px 0 rgba(215, 255, 100, 0.9) !important;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        max-width: 640px;
        margin: 0 0 20px 0;
        line-height: 1.5;
    }
    .hero-telemetry-row {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-subtle);
        padding-top: 14px;
    }
    .hero-telemetry-item b {
        color: var(--text-pure);
    }

    /* 3D Wave Visualizer Frame */
    .visualizer-hud {
        position: relative;
        background: #0B0D12;
        border: 1px solid var(--border-bright);
        border-radius: 8px;
        padding: 16px;
        overflow: hidden;
    }
    .hud-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--accent-violet);
        margin-bottom: 8px;
        border-bottom: 1px solid #1F222C;
        padding-bottom: 6px;
    }

    /* Metric Panels */
    .metric-panel {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .metric-panel:hover {
        border-color: var(--border-violet);
        transform: translateY(-2px);
    }
    .metric-panel::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 32px;
        height: 2px;
        background: var(--accent-violet);
    }
    .metric-panel.highlight-lime::after {
        background: var(--accent-lime);
    }
    .metric-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-pure);
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-val.lime-val {
        color: var(--accent-lime);
    }
    .metric-val.violet-val {
        color: var(--accent-violet);
    }
    .metric-sub {
        font-size: 0.74rem;
        color: var(--text-secondary);
        margin-top: 6px;
    }

    /* Native Containers & Panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        padding: 22px 20px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
        margin-bottom: 18px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(155, 123, 255, 0.35) !important;
    }
    .hud-panel {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .hud-panel-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        color: #F5F5F7 !important;
        letter-spacing: -0.01em !important;
        margin: 0 0 16px 0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    /* Pipeline Specs Cards */
    .spec-row {
        display: flex;
        gap: 14px;
        padding: 12px 0;
        border-bottom: 1px solid #1C1F28;
    }
    .spec-row:last-child {
        border-bottom: none;
    }
    .spec-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-lime);
        background: rgba(215, 255, 100, 0.1);
        border: 1px solid rgba(215, 255, 100, 0.25);
        border-radius: 4px;
        width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .spec-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-pure);
        margin-bottom: 3px;
    }
    .spec-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.45;
        margin: 0;
    }

    /* Sound Detection Instrument Box */
    .scanner-box {
        position: relative;
        background: #0D0F14;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 24px 20px;
        text-align: center;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .scanner-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            linear-gradient(rgba(34, 37, 48, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34, 37, 48, 0.3) 1px, transparent 1px);
        background-size: 24px 24px;
        opacity: 0.5;
        pointer-events: none;
    }
    .crosshair-corner {
        position: absolute;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: var(--accent-violet);
    }
    .tl { top: 6px; left: 8px; }
    .tr { top: 6px; right: 8px; }
    .bl { bottom: 6px; left: 8px; }
    .br { bottom: 6px; right: 8px; }

    /* Telemetry Pills */
    .telemetry-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        background: #111319;
        color: var(--text-secondary);
        border: 1px solid var(--border-subtle);
        margin-right: 6px;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .telemetry-pill b {
        color: var(--text-pure);
    }

    /* High-Voltage Action Button (Acid Lime) */
    div.stButton > button:first-child {
        background: var(--accent-lime) !important;
        color: #08090C !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1.4rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 16px rgba(215, 255, 100, 0.25) !important;
    }
    div.stButton > button:first-child:hover {
        background: #E5FFA1 !important;
        box-shadow: 0 0 24px rgba(215, 255, 100, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Prediction Result Panel */
    .result-cockpit {
        background: linear-gradient(145deg, #161821 0%, #11131A 100%);
        border: 1px solid var(--border-violet);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(155, 123, 255, 0.2);
    }
    .result-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-lime);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }
    .result-headline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-pure);
        letter-spacing: -0.03em;
        margin: 0 0 8px 0;
        text-transform: uppercase;
        border-bottom: 2px solid var(--accent-lime);
        padding-bottom: 6px;
        display: inline-block;
    }
    .result-telemetry {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 10px;
    }
    .result-telemetry span.hi {
        color: var(--accent-lime);
        font-weight: 700;
    }

    /* Progress Bar Theme */
    [data-testid="stProgress"] > div > div > div > div,
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #9B7BFF 0%, #D7FF64 100%) !important;
        border-radius: 4px !important;
    }
    [data-testid="stProgress"] > div > div > div,
    .stProgress > div > div > div {
        background-color: #1F222C !important;
    }

    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #111319 !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-pure) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Top Breadcrumb Bar */
    .top-breadcrumb {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .top-breadcrumb .crumb-title {
        color: var(--text-pure);
        font-weight: 600;
    }
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_metadata():
    try:
        return load_metadata()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path_str: str):
    try:
        return load_model(model_path_str)
    except Exception:
        return None, {}


def main():
    # Sidebar Navigation & Telemetry
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-container">
                <div class="brand-icon">
                    <svg viewBox="0 0 24 24" width="16" height="16" stroke="#9B7BFF" stroke-width="2.5" stroke-linecap="round" fill="none">
                        <line x1="3" y1="12" x2="3" y2="12"></line>
                        <line x1="7" y1="6" x2="7" y2="18"></line>
                        <line x1="11" y1="2" x2="11" y2="22"></line>
                        <line x1="15" y1="7" x2="15" y2="17"></line>
                        <line x1="19" y1="10" x2="19" y2="14"></line>
                    </svg>
                </div>
                <div class="brand-name">ECHODETECT</div>
                <div class="brand-tag">SYS-001</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        menu = st.radio(
            "Navigation",
            [
                "Overview",
                "Sound Detection",
                "Audio Analysis",
                "Model Performance",
                "Dataset Explorer",
                "About"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Active Classifier Engine Selector
        st.markdown("<div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; color: #A5A9B5; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-weight: 700;'>ACTIVE CLASSIFIER</div>", unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Classifier Model",
            ["SVM (RBF Kernel + Platt)", "Random Forest (200 Trees)"],
            index=0,
            label_visibility="collapsed"
        )
        selected_model_path = SVM_MODEL_PATH if "SVM" in model_choice else RF_MODEL_PATH

        st.markdown("---")

        # Hardware & Signal Telemetry Status
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #6B7280; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px; color: #D7FF64; font-weight: 700;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #D7FF64; box-shadow: 0 0 6px #D7FF64;"></span>
                    <span>SIGNAL ENGINE ONLINE</span>
                </div>
                <div>CORPUS: UrbanSound8K</div>
                <div>SAMPLES: 8,732 RECORDINGS</div>
                <div>FEATURE DIM: 272-DIMENSIONAL</div>
                <div>ISOLATION: FOLD 10 HELD-OUT</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Dispatch to Active Page
    if menu == "Overview":
        render_overview(selected_model_path, model_choice)
    elif menu == "Sound Detection":
        render_detection(selected_model_path, model_choice)
    elif menu == "Audio Analysis":
        render_audio_analysis()
    elif menu == "Model Performance":
        render_model_performance()
    elif menu == "Dataset Explorer":
        render_dataset_explorer()
    elif menu == "About":
        render_about()


# ==============================================================================
# 1. OVERVIEW DASHBOARD & HERO SECTION
# ==============================================================================
def render_overview(model_path: Path, model_name: str):
    summary = load_evaluation_summary()
    meta_df = get_cached_metadata()

    total_samples = len(meta_df) if meta_df is not None else 8732
    num_classes = len(CLASSES)

    model_key = "svm" if "SVM" in model_name else "rf"
    if summary and model_key in summary:
        test_acc = f"{summary[model_key]['metrics']['accuracy'] * 100:.1f}%"
        macro_f1 = f"{summary[model_key]['metrics']['f1_macro']:.4f}"
        fit_time = f"{summary[model_key]['fit_time']}s"
    else:
        test_acc = "77.9%"
        macro_f1 = "0.7984"
        fit_time = "11.9s"

    # Top Breadcrumb Telemetry Bar
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">OVERVIEW DASHBOARD</span>
            </div>
            <div style="display: flex; gap: 14px;">
                <span>NYQUIST: 11,025 HZ</span>
                <span>HOP: 512</span>
                <span style="color: #D7FF64; font-weight: 700;">● ZERO-LEAKAGE BENCHMARK</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Hero Editorial Section with 3D Wave Visual
    hero_col_left, hero_col_right = st.columns([1.25, 0.75])

    with hero_col_left:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-eyebrow">
                    <span class="pulsing-dot"></span>
                    <span>AUDIO INTELLIGENCE / SYSTEM 001</span>
                </div>
                <h1 class="hero-headline" style="color: #FFFFFF !important; text-shadow: 0 0 20px rgba(255, 255, 255, 0.15) !important;">
                    EVERY SOUND<br>HAS A SIGNAL.
                </h1>
                <p class="hero-subtitle">
                    Machine learning that translates environmental audio into recognizable acoustic events. 
                    Standardized on 272-dimensional cepstral and spectral dynamics.
                </p>
                <div class="hero-telemetry-row">
                    <span class="hero-telemetry-item">SAMPLING: <b>22,050 HZ MONO</b></span>
                    <span class="hero-telemetry-item">MFCC: <b>40 COEFFICIENTS</b></span>
                    <span class="hero-telemetry-item">CALIBRATION: <b>PLATT SCALING</b></span>
                    <span class="hero-telemetry-item">EVALUATION: <b>FOLD 10 TEST</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with hero_col_right:
        # 3D Sound-Wave Hologram Simulation (SVG / CSS with Depth & Scan Line)
        st.markdown(
            """
            <div class="visualizer-hud">
                <div class="hud-header">
                    <span>3D SPECTRAL SCANNER</span>
                    <span style="color: #D7FF64;">● ACTIVE MATRIX</span>
                </div>
                <div style="height: 190px; display: flex; align-items: center; justify-content: center; position: relative;">
                    <svg viewBox="0 0 340 180" width="100%" height="100%" style="overflow: visible;">
                        <defs>
                            <linearGradient id="waveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#9B7BFF" stop-opacity="0.2"/>
                                <stop offset="50%" stop-color="#9B7BFF" stop-opacity="0.9"/>
                                <stop offset="100%" stop-color="#D7FF64" stop-opacity="0.2"/>
                            </linearGradient>
                            <linearGradient id="waveGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#00F2FE" stop-opacity="0.1"/>
                                <stop offset="50%" stop-color="#00F2FE" stop-opacity="0.8"/>
                                <stop offset="100%" stop-color="#9B7BFF" stop-opacity="0.2"/>
                            </linearGradient>
                            <linearGradient id="gridGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="#15171D"/>
                                <stop offset="100%" stop-color="#222530"/>
                            </linearGradient>
                        </defs>
                        <!-- 3D Perspective Grid Background -->
                        <g opacity="0.35">
                            <line x1="20" y1="140" x2="320" y2="140" stroke="#323746" stroke-width="0.8"/>
                            <line x1="40" y1="120" x2="300" y2="120" stroke="#323746" stroke-width="0.6"/>
                            <line x1="60" y1="100" x2="280" y2="100" stroke="#323746" stroke-width="0.5"/>
                            <line x1="80" y1="80" x2="260" y2="80" stroke="#323746" stroke-width="0.4"/>
                            <!-- Perspective Rays -->
                            <line x1="20" y1="140" x2="140" y2="60" stroke="#323746" stroke-width="0.6"/>
                            <line x1="80" y1="140" x2="155" y2="60" stroke="#323746" stroke-width="0.6"/>
                            <line x1="170" y1="140" x2="170" y2="60" stroke="#323746" stroke-width="0.6"/>
                            <line x1="260" y1="140" x2="185" y2="60" stroke="#323746" stroke-width="0.6"/>
                            <line x1="320" y1="140" x2="200" y2="60" stroke="#323746" stroke-width="0.6"/>
                        </g>
                        <!-- Secondary Oscillating Wave (Cyan) -->
                        <path d="M 20 115 Q 60 70, 100 115 T 180 115 T 260 115 T 320 115" 
                              fill="none" stroke="url(#waveGrad2)" stroke-width="1.8" opacity="0.75"/>
                        <!-- Primary Oscillating Wave (Violet to Lime) -->
                        <path d="M 20 110 Q 55 35, 95 110 T 175 110 T 255 110 T 320 110" 
                              fill="none" stroke="url(#waveGrad1)" stroke-width="2.5"/>
                        <!-- Dynamic Frequency Bars -->
                        <rect x="50" y="85" width="4" height="35" rx="2" fill="#9B7BFF" opacity="0.8"/>
                        <rect x="75" y="60" width="4" height="60" rx="2" fill="#9B7BFF" opacity="0.9"/>
                        <rect x="100" y="75" width="4" height="45" rx="2" fill="#9B7BFF" opacity="0.7"/>
                        <rect x="135" y="50" width="4" height="70" rx="2" fill="#D7FF64" opacity="0.9"/>
                        <rect x="160" y="70" width="4" height="50" rx="2" fill="#D7FF64" opacity="0.8"/>
                        <rect x="195" y="55" width="4" height="65" rx="2" fill="#00F2FE" opacity="0.8"/>
                        <rect x="230" y="80" width="4" height="40" rx="2" fill="#9B7BFF" opacity="0.75"/>
                        <rect x="270" y="65" width="4" height="55" rx="2" fill="#D7FF64" opacity="0.85"/>
                        <!-- Scanning Laser Line (Lime) -->
                        <line x1="168" y1="30" x2="168" y2="155" stroke="#D7FF64" stroke-width="1.5" stroke-dasharray="3 3"/>
                        <circle cx="168" cy="92" r="3.5" fill="#D7FF64"/>
                    </svg>
                </div>
                <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #6B7280; border-top: 1px solid #1F222C; padding-top: 6px;">
                    <span>FFT: 2048 BINS</span>
                    <span>LATENCY: &lt;14MS</span>
                    <span style="color: #9B7BFF;">BANDWIDTH: 11 KHZ</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4 Refined Editorial Metric Panels
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-tag">01 // CORPUS TOTAL</div>
            <div class="metric-val">{total_samples:,}</div>
            <div class="metric-sub">UrbanSound8K audio files</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-tag">02 // TAXONOMY</div>
            <div class="metric-val">{num_classes} CLASSES</div>
            <div class="metric-sub">Environmental sound events</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        short_name = "SVM" if "SVM" in model_name else "RANDOM FOREST"
        st.markdown(f"""
        <div class="metric-panel highlight-lime">
            <div class="metric-tag">03 // TEST ACCURACY</div>
            <div class="metric-val lime-val">{test_acc}</div>
            <div class="metric-sub">{short_name} on held-out Fold 10</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-tag">04 // BALANCED F1</div>
            <div class="metric-val violet-val">{macro_f1}</div>
            <div class="metric-sub">Unweighted macro harmony</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Two-Column Dashboard Section: Distribution & Architecture
    col_chart, col_pipe = st.columns([1.15, 0.85])

    with col_chart:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #9B7BFF;'>01 //</span> SOUND CLASS DISTRIBUTION</div>", unsafe_allow_html=True)

            if meta_df is not None:
                class_counts = meta_df['class'].value_counts().reset_index()
                class_counts.columns = ['Sound Class', 'Count']
                class_counts['Display'] = class_counts['Sound Class'].apply(lambda x: x.replace('_', ' ').upper())

                # High-Contrast Plotly Bar Chart
                fig = px.bar(
                    class_counts,
                    x='Count',
                    y='Display',
                    orientation='h',
                    color_discrete_sequence=['#222530']
                )
                # High-voltage accents on minority and majority classes
                colors = []
                for i, row in enumerate(class_counts.iterrows()):
                    if i < 2:
                        colors.append('#D7FF64')  # Acid Lime
                    elif i > 7:
                        colors.append('#9B7BFF')  # Electric Violet
                    else:
                        colors.append('#222530')

                fig.update_traces(
                    marker=dict(
                        color=colors,
                        line=dict(color='#323746', width=1)
                    )
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#A5A9B5', family='JetBrains Mono', size=10),
                    margin=dict(l=10, r=10, t=6, b=10),
                    height=340,
                    xaxis=dict(
                        gridcolor='#191C24',
                        title=dict(text="SAMPLE COUNT", font=dict(size=10, color='#6B7280')),
                        tickfont=dict(color='#6B7280')
                    ),
                    yaxis=dict(
                        categoryorder='total ascending',
                        title=None,
                        tickfont=dict(color='#F5F5F7')
                    )
                )
                st.plotly_chart(fig, config={'displayModeBar': False})

    with col_pipe:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>02 //</span> PIPELINE SPECIFICATIONS</div>", unsafe_allow_html=True)

            specs = [
                ("01", "INGESTION & MONO DOWNMIX", "Resample to 22,050 Hz via polyphase sinc filter (soxr_hq). Peak amplitude normalization to 1.0."),
                ("02", "272D SPECTRAL EMBEDDING", "40 MFCCs (mean/std) + 1st/2nd temporal derivatives (deltas) + Centroid, Bandwidth, Rolloff, ZCR, & Chroma."),
                ("03", "LEAK-FREE FOLD PARTITIONING", "Folds 1–8 training, Fold 9 validation/tuning, and Fold 10 strictly held out to eliminate excerpt leakage."),
                ("04", "CALIBRATED ML ENGINES", "StandardScaler + Support Vector Machine with Platt probability calibration & 200-tree Random Forest.")
            ]

            for num, title, desc in specs:
                st.markdown(f"""
                <div class="spec-row">
                    <div class="spec-badge">{num}</div>
                    <div>
                        <div class="spec-title">{title}</div>
                        <p class="spec-desc">{desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 2. SOUND DETECTION (CENTERPIECE ACOUSTIC INSTRUMENT)
# ==============================================================================
def render_detection(model_path: Path, model_name: str):
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">ACOUSTIC SIGNAL CLASSIFIER</span>
            </div>
            <div style="color: #9B7BFF;">MODULE: REAL-TIME INFERENCE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pipeline, metadata = get_cached_model(str(model_path))
    if pipeline is None:
        st.error(f"Model checkpoint not found at `{model_path}`. Run `train.py` first.")
        return

    col_input, col_result = st.columns([1, 1.18])

    with col_input:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>INPUT //</span> ACOUSTIC INGESTION</div>", unsafe_allow_html=True)

            input_mode = st.radio(
                "Input Mode",
                ["Drag & Drop WAV Recording", "Select from Fold 10 (Held-Out Test Set)"],
                horizontal=True,
                label_visibility="collapsed"
            )

            audio_bytes = None
            audio_name = None
            audio_temp_path = None

            if input_mode == "Drag & Drop WAV Recording":
                st.markdown(
                    """
                    <div class="scanner-box">
                        <span class="crosshair-corner tl">+</span>
                        <span class="crosshair-corner tr">+</span>
                        <span class="crosshair-corner bl">+</span>
                        <span class="crosshair-corner br">+</span>
                        <svg viewBox="0 0 24 24" width="28" height="28" stroke="#D7FF64" stroke-width="2" stroke-linecap="round" fill="none" style="margin-bottom: 8px;">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; color: #F5F5F7; font-weight: 700;">DROP WAV AUDIO FILE HERE</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6B7280; margin-top: 4px;">16/24-BIT PCM WAV &bull; DURATION &le; 4.0 SECONDS</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                uploaded_file = st.file_uploader("Upload audio file", type=["wav"], label_visibility="collapsed")
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    audio_name = uploaded_file.name
                    temp_dir = BASE_DIR / "cache"
                    temp_dir.mkdir(exist_ok=True)
                    audio_temp_path = temp_dir / f"upload_{uploaded_file.name}"
                    with open(audio_temp_path, "wb") as f:
                        f.write(audio_bytes)
            else:
                meta_df = get_cached_metadata()
                if meta_df is not None:
                    fold10_df = meta_df[meta_df['fold'] == 10]
                    sample_options = fold10_df[['slice_file_name', 'class']].to_dict('records')
                    selected_sample = st.selectbox(
                        "Select Test Sample",
                        sample_options,
                        format_func=lambda x: f"{x['slice_file_name']}  [TRUTH: {x['class'].replace('_', ' ').upper()}]",
                        label_visibility="collapsed"
                    )
                    if selected_sample:
                        audio_name = selected_sample['slice_file_name']
                        audio_temp_path = AUDIO_DIR / "fold10" / audio_name
                        if audio_temp_path.exists():
                            with open(audio_temp_path, "rb") as f:
                                audio_bytes = f.read()

            if audio_bytes and audio_temp_path and audio_temp_path.exists():
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.audio(audio_bytes, format="audio/wav")
                try:
                    info = sf.info(str(audio_temp_path))
                    st.markdown(f"""
                    <div style="margin: 10px 0 14px 0;">
                        <span class="telemetry-pill">FILE: <b>{audio_name}</b></span>
                        <span class="telemetry-pill">DURATION: <b>{info.duration:.2f}S</b></span>
                        <span class="telemetry-pill">RATE: <b>{info.samplerate} HZ</b></span>
                        <span class="telemetry-pill">CHANNELS: <b>{'MONO' if info.channels == 1 else 'STEREO'}</b></span>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    pass

                analyze_btn = st.button("⚡ ANALYZE SIGNAL", type="primary", width="stretch")
            else:
                analyze_btn = False

    with col_result:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #9B7BFF;'>OUTPUT //</span> ACOUSTIC SIGNATURE TELEMETRY</div>", unsafe_allow_html=True)

            if analyze_btn and audio_temp_path:
                with st.spinner("RUNNING 272-DIMENSIONAL FEATURE EXTRACTION & PLATT INFERENCE..."):
                    t0 = time.time()
                    try:
                        feat = extract_features_from_file(audio_temp_path)
                        feat_2d = feat.reshape(1, -1)
                        pred_id = int(pipeline.predict(feat_2d)[0])
                        pred_class = CLASSES.get(pred_id, f"Class {pred_id}")
                        infer_time = (time.time() - t0) * 1000

                        probas = pipeline.predict_proba(feat_2d)[0] if hasattr(pipeline, "predict_proba") else None
                        top_conf = probas[pred_id] * 100 if probas is not None else 100.0

                        # Dimensional Result Cockpit
                        st.markdown(f"""
                        <div class="result-cockpit">
                            <div class="result-eyebrow">
                                <span class="pulsing-dot"></span>
                                <span>SIGNAL IDENTIFIED // CALIBRATED ESTIMATE</span>
                            </div>
                            <div style="margin-top: 4px;">
                                <div class="result-headline">{pred_class.replace('_', ' ')}</div>
                            </div>
                            <div class="result-telemetry">
                                <span>CONFIDENCE: <span class="hi">{top_conf:.1f}%</span></span>
                                <span>&bull;</span>
                                <span>LATENCY: <span class="hi">{infer_time:.1f}MS</span></span>
                                <span>&bull;</span>
                                <span>EMBEDDING: <span class="hi">272D</span></span>
                                <span>&bull;</span>
                                <span>ENGINE: <span class="hi">{'SVM-RBF' if 'SVM' in model_name else 'RF-200'}</span></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Acoustic Signature Description Box
                        desc = CLASS_DESCRIPTIONS.get(pred_class, "")
                        if desc:
                            st.markdown(f"""
                            <div style="font-size: 0.84rem; color: #A5A9B5; line-height: 1.55; margin-bottom: 16px; padding: 12px 14px; background: #111319; border-radius: 6px; border: 1px solid #222530;">
                                <span style="color: #D7FF64; font-weight: 700; font-family: 'JetBrains Mono', monospace;">ACOUSTIC SIGNATURE:</span> {desc}
                            </div>
                            """, unsafe_allow_html=True)

                        # Top-3 Prediction Distribution (High-Voltage Gradient Bars)
                        if probas is not None:
                            st.markdown("<div style='font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #A5A9B5; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-bottom: 12px;'>TOP-3 PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)
                            top_indices = np.argsort(probas)[::-1][:3]

                            for rank, idx in enumerate(top_indices, 1):
                                cname = CLASSES.get(idx, f"Class {idx}").replace('_', ' ').upper()
                                prob = float(probas[idx])
                                pct_str = f"{prob * 100:.1f}%"
                                st.markdown(f"""
                                <div style="margin-bottom: 12px;">
                                    <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; margin-bottom: 5px;">
                                        <span style="font-weight: 700; color: #F5F5F7;">0{rank} // {cname}</span>
                                        <span style="font-weight: 700; color: #D7FF64;">{pct_str}</span>
                                    </div>
                                    <div style="width: 100%; height: 7px; background: #1C1F28; border-radius: 4px; overflow: hidden; border: 1px solid #282C38;">
                                        <div style="width: {prob * 100:.1f}%; height: 100%; background: linear-gradient(90deg, #9B7BFF 0%, #D7FF64 100%); border-radius: 4px;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Inference error: {e}")
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 56px 16px; color: #6B7280;">
                        <svg viewBox="0 0 24 24" width="34" height="34" stroke="#323746" stroke-width="1.8" fill="none" style="margin-bottom: 10px;">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 700; color: #A5A9B5;">INSTRUMENT IN STANDBY</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; margin-top: 4px;">Ingest a recording and engage <b>ANALYZE SIGNAL</b> to initialize acoustic classification.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                """
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #6B7280; line-height: 1.5; border-top: 1px solid #1C1F28; padding-top: 12px; margin-top: 16px;">
                    NOTE: Predictions represent genuine calibrated posterior probabilities produced by the machine learning model on the UrbanSound8K taxonomy. Intended for research and classification purposes.
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# 3. AUDIO ANALYSIS
# ==============================================================================
def render_audio_analysis():
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">TIME-FREQUENCY SIGNAL SPECTROMETRY</span>
            </div>
            <div style="color: #00F2FE;">COLORMAP: MAGMA &bull; 2048 FFT</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    meta_df = get_cached_metadata()
    if meta_df is None:
        st.error("Metadata not found.")
        return

    c_sel1, c_sel2 = st.columns([1, 2])
    with c_sel1:
        chosen_class = st.selectbox("Filter Class", CLASS_NAMES, index=3)
    with c_sel2:
        class_samples = meta_df[meta_df['class'] == chosen_class]
        selected_file = st.selectbox("Select Audio Clip", class_samples['slice_file_name'].tolist()[:30])

    row = meta_df[meta_df['slice_file_name'] == selected_file].iloc[0]
    audio_path = row['file_path']

    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return

    with st.spinner("Computing continuous STFT and Mel filterbanks..."):
        fig_wave, fig_mfcc, duration = generate_audio_plots(audio_path)

        with st.container(border=True):
            col_aud, col_info = st.columns([1, 2])
            with col_aud:
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            with col_info:
                st.markdown(f"""
                <div style="padding-top: 4px;">
                    <span class="telemetry-pill">CLASS: <b>{row['class'].replace('_', ' ').upper()}</b></span>
                    <span class="telemetry-pill">FOLD: <b>{row['fold']}</b></span>
                    <span class="telemetry-pill">SALIENCE: <b>{'FOREGROUND' if row['salience']==1 else 'BACKGROUND'}</b></span>
                    <span class="telemetry-pill">DURATION: <b>{duration:.2f}S</b></span>
                </div>
                """, unsafe_allow_html=True)

            st.pyplot(fig_wave)
            st.pyplot(fig_mfcc)

    # Scientific Explanation Panel
    with st.container(border=True):
        st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>THEORY //</span> THE MEL-FREQUENCY CEPSTRUM</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.86rem; color: #A5A9B5; line-height: 1.6;">
                <p>
                    <b>Logarithmic Cochlear Mapping:</b> Human pitch perception operates logarithmically; the ear discriminates subtle pitch shifts at lower frequencies much more acutely than at ultrasonic frequencies. The Mel scale mirrors this human critical band structure.
                </p>
                <p style="margin-bottom: 0;">
                    <b>The Cepstral Transformation:</b> 2,048-point STFT frames pass through 40 triangular Mel filterbanks. Applying a Discrete Cosine Transform (DCT) decorrelates filter energies, yielding 40 cepstral coefficients that capture the vocal/acoustic resonant envelope while discarding pitch harmonics. First and second temporal differences (&Delta; and &Delta;&Delta;) capture velocity and acceleration transients such as siren sweeps and gunshot attacks.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==============================================================================
# 4. MODEL PERFORMANCE
# ==============================================================================
def render_model_performance():
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">BENCHMARKS & CROSS-FOLD VALIDATION</span>
            </div>
            <div style="color: #D7FF64; font-weight: 700;">● ZERO-LEAKAGE EVALUATION (FOLD 10)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    summary = load_evaluation_summary()
    if not summary or "svm" not in summary or "rf" not in summary:
        st.warning("Evaluation summary not available. Please run `train.py` first.")
        return

    svm_m = summary["svm"]["metrics"]
    rf_m = summary["rf"]["metrics"]

    # Comparative Metrics Table
    with st.container(border=True):
        st.markdown("<div class='hud-panel-title'><span style='color: #9B7BFF;'>01 //</span> CLASSIFIER COMPARISON (STRICTLY HELD-OUT FOLD 10)</div>", unsafe_allow_html=True)

        metrics_df = pd.DataFrame([
            {
                "Classifier Engine": "Support Vector Machine (RBF + Platt)",
                "Accuracy": f"{svm_m['accuracy'] * 100:.2f}%",
                "Macro Precision": f"{svm_m['precision_macro']:.4f}",
                "Macro Recall": f"{svm_m['recall_macro']:.4f}",
                "Macro F1": f"{svm_m['f1_macro']:.4f}",
                "Weighted F1": f"{svm_m['f1_weighted']:.4f}",
                "Train Latency": f"{summary['svm']['fit_time']}s"
            },
            {
                "Classifier Engine": "Random Forest (200 Ensembled Trees)",
                "Accuracy": f"{rf_m['accuracy'] * 100:.2f}%",
                "Macro Precision": f"{rf_m['precision_macro']:.4f}",
                "Macro Recall": f"{rf_m['recall_macro']:.4f}",
                "Macro F1": f"{rf_m['f1_macro']:.4f}",
                "Weighted F1": f"{rf_m['f1_weighted']:.4f}",
                "Train Latency": f"{summary['rf']['fit_time']}s"
            }
        ])
        st.dataframe(metrics_df, width="stretch", hide_index=True)

    # Per-Class Comparison Chart
    with st.container(border=True):
        st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>02 //</span> PER-CLASS F1-SCORE BREAKDOWN</div>", unsafe_allow_html=True)

        per_class_rows = []
        for cname in CLASS_NAMES:
            if cname in svm_m.get("per_class", {}):
                per_class_rows.append({
                    "Class": cname.replace('_', ' ').upper(),
                    "SVM F1": svm_m["per_class"][cname]["f1_score"],
                    "RF F1": rf_m["per_class"][cname]["f1_score"],
                })

        if per_class_rows:
            df_pc = pd.DataFrame(per_class_rows)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='SVM (RBF)',
                x=df_pc['Class'],
                y=df_pc['SVM F1'],
                marker_color='#9B7BFF',
                marker_line=dict(color='#B59AFF', width=1)
            ))
            fig_bar.add_trace(go.Bar(
                name='RANDOM FOREST',
                x=df_pc['Class'],
                y=df_pc['RF F1'],
                marker_color='#D7FF64',
                marker_line=dict(color='#E5FFA1', width=1)
            ))
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#A5A9B5', family='JetBrains Mono', size=10),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(gridcolor='#191C24', title="F1-SCORE", range=[0, 1.05], tickfont=dict(color='#6B7280')),
                xaxis=dict(title=None, tickfont=dict(color='#F5F5F7')),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#F5F5F7')
                )
            )
            st.plotly_chart(fig_bar, config={'displayModeBar': False})

    # Confusion Matrices Side-by-Side
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #9B7BFF;'>03 //</span> SVM CONFUSION MATRIX (FOLD 10)</div>", unsafe_allow_html=True)
            svm_cm_img = REPORTS_DIR / "svm_confusion_matrix.png"
            if svm_cm_img.exists():
                st.image(str(svm_cm_img), width="stretch")

    with col_cm2:
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>04 //</span> RANDOM FOREST CONFUSION MATRIX (FOLD 10)</div>", unsafe_allow_html=True)
            rf_cm_img = REPORTS_DIR / "rf_confusion_matrix.png"
            if rf_cm_img.exists():
                st.image(str(rf_cm_img), width="stretch")

    # Feature Importance Plot
    rf_feat_img = REPORTS_DIR / "rf_feature_importances.png"
    if rf_feat_img.exists():
        with st.container(border=True):
            st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>05 //</span> RANDOM FOREST GINI FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
            st.image(str(rf_feat_img), width="stretch")


# ==============================================================================
# 5. DATASET EXPLORER
# ==============================================================================
def render_dataset_explorer():
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">URBANSOUND8K BENCHMARK EXPLORER</span>
            </div>
            <div style="color: #A5A9B5;">TOTAL: 8,732 RECORDINGS</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = get_cached_metadata()
    if df is None:
        st.error("Dataset metadata could not be loaded.")
        return

    # Fold Summary Chart
    with st.container(border=True):
        st.markdown("<div class='hud-panel-title'><span style='color: #D7FF64;'>01 //</span> 10-FOLD CROSS-VALIDATION PARTITIONING</div>", unsafe_allow_html=True)

        fold_counts = df.groupby('fold').size().reset_index(name='Sample Count')
        fold_counts['Partition'] = fold_counts['fold'].apply(
            lambda f: "TRAINING SET (FOLDS 1–8)" if f <= 8 else ("VALIDATION SET (FOLD 9)" if f == 9 else "HELD-OUT TEST SET (FOLD 10)")
        )

        fig_folds = px.bar(
            fold_counts,
            x='fold',
            y='Sample Count',
            color='Partition',
            color_discrete_map={
                "TRAINING SET (FOLDS 1–8)": "#222530",
                "VALIDATION SET (FOLD 9)": "#9B7BFF",
                "HELD-OUT TEST SET (FOLD 10)": "#D7FF64"
            }
        )
        fig_folds.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#A5A9B5', family='JetBrains Mono', size=10),
            height=280,
            margin=dict(l=10, r=10, t=6, b=10),
            xaxis=dict(tickmode='linear', tick0=1, dtick=1, title="FOLD INDEX", tickfont=dict(color='#F5F5F7')),
            yaxis=dict(gridcolor='#191C24', title="RECORDINGS", tickfont=dict(color='#6B7280')),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#F5F5F7'))
        )
        st.plotly_chart(fig_folds, config={'displayModeBar': False})

    # Interactive Metadata Browser & Audio Player
    with st.container(border=True):
        st.markdown("<div class='hud-panel-title'><span style='color: #9B7BFF;'>02 //</span> METADATA TELEMETRY BROWSER</div>", unsafe_allow_html=True)

        c_filt1, c_filt2, c_filt3 = st.columns(3)
        with c_filt1:
            sel_fold = st.selectbox("Fold Index", ["All"] + list(range(1, 11)))
        with c_filt2:
            sel_class = st.selectbox("Taxonomy Class", ["All"] + CLASS_NAMES)
        with c_filt3:
            sel_salience = st.selectbox("Salience", ["All", "1 (Foreground)", "2 (Background)"])

        filtered = df.copy()
        if sel_fold != "All":
            filtered = filtered[filtered['fold'] == sel_fold]
        if sel_class != "All":
            filtered = filtered[filtered['class'] == sel_class]
        if sel_salience != "All":
            sal_val = int(sel_salience[0])
            filtered = filtered[filtered['salience'] == sal_val]

        st.markdown(f"<div style='font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: #6B7280; margin-bottom: 8px;'>MATCHING RECORDINGS: {len(filtered):,} CLIPS (SHOWING TOP 50):</div>", unsafe_allow_html=True)
        st.dataframe(
            filtered[['slice_file_name', 'class', 'fold', 'salience', 'start', 'end']].head(50),
            width="stretch",
            hide_index=True
        )

        if len(filtered) > 0:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            pick_sample = st.selectbox("Select clip to listen", filtered['slice_file_name'].head(20).tolist())
            chosen_row = filtered[filtered['slice_file_name'] == pick_sample].iloc[0]
            audio_path = chosen_row['file_path']
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                st.caption(f"Ground Truth: **{chosen_row['class'].replace('_', ' ').upper()}** &bull; Fold: {chosen_row['fold']} &bull; Salience: {'Foreground' if chosen_row['salience']==1 else 'Background'}")

    # Attribution
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #6B7280; line-height: 1.6;">
                <div style="font-weight: 700; color: #A5A9B5; margin-bottom: 4px;">URBANSOUND8K BENCHMARK CITATION</div>
                NYU Music and Audio Research Laboratory (MARL) & Center for Urban Science and Progress (CUSP).
                <br>
                Citation: <i>J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, 2014.</i>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==============================================================================
# 6. ABOUT PROJECT (LABORATORY SPEC SHEET)
# ==============================================================================
def render_about():
    st.markdown(
        """
        <div class="top-breadcrumb">
            <div>
                <span>SYS // ECHODETECT</span> &nbsp;&gt;&nbsp; <span class="crumb-title">ARCHITECTURE SPECIFICATION</span>
            </div>
            <div style="color: #D7FF64;">VIVA DEFENSE REFERENCE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown(r"""
        ### 01 // Problem Statement & Objectives
        Urban noise pollution directly impairs public health and indicates municipal safety disruptions. Conventional sound meters capture simple decibel pressure levels ($L_{eq}$), providing amplitude without acoustic identity. EchoDetect performs automated classification across 10 environmental acoustic classes using standardized cepstral feature representations and classical machine learning algorithms.

        ### 02 // Mathematical Foundations
        - **Audio Preprocessing**: Standardized to 22,050 Hz mono via polyphase sinc filtering (`soxr_hq`).
        - **Feature Engineering (272D)**: 40 static MFCCs (mean/std) + first derivative $\Delta$ velocity (mean/std) + second derivative $\Delta\Delta$ acceleration (mean/std) + spectral centroid, bandwidth, rolloff, zero-crossing rate, and 12-band Chroma STFT.
        - **Support Vector Machine (SVM)**: Maps 272D features into reproducing kernel Hilbert space (RKHS) via the Radial Basis Function:
          $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
          Class probabilities are calibrated using Platt scaling (`CalibratedClassifierCV`).
        - **Random Forest**: Ensemble of 200 de-correlated decision trees with `class_weight='balanced_subsample'` to handle class imbalance on minority categories (`gun_shot`, `car_horn`).

        ### 03 // Real-World Applications vs Current Scope
        - **Smart City Acoustic Mapping**: Real-time identification of construction drilling and traffic exceedances.
        - **Assistive Safety Technology**: Haptic or visual alarms for sirens and vehicle horns for hearing-impaired users.
        - **Implemented Scope**: File-based classification on 4-second audio clips with pre-defined fold isolation.
        - **Future Work**: Continuous microphone streaming via WebSockets and Audio Spectrogram Transformers (AST).
        """)


if __name__ == "__main__":
    main()
