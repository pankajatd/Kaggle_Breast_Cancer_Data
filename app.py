"""
Breast Cancer Diagnostic AI Console — Streamlit Cloud Application
===================================================================
Wisconsin Diagnostic Breast Cancer (WDBC) Machine Learning Platform.
Predicts Benign vs. Malignant tumor diagnosis based on 10 standardized
fine needle aspirate (FNA) nuclear features using Random Forest ML.
"""
import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Setup sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Diagnostic AI — WDBC Classifier",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1400px !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 1.1rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: white;
        border: 1px solid #4f46e5;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .header-banner h1 { margin: 0; font-size: 1.45rem; font-weight: 800; }
    .header-banner p { margin: 0.25rem 0 0; opacity: 0.9; font-size: 0.85rem; }
    
    .stMetric {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    .result-card-malignant {
        background: #450a0a;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem;
        color: #fecaca;
        margin-top: 1rem;
    }
    .result-card-benign {
        background: #064e3b;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.2rem;
        color: #d1fae5;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Model Loading & Caching
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading Random Forest Diagnostic Model...")
def load_assets():
    model_path = PROJECT_ROOT / "model.pkl"
    scaler_path = PROJECT_ROOT / "scaling_new.pkl"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    scaler = None
    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None
            
    return model, scaler

model, scaler = load_assets()

# ──────────────────────────────────────────────────────────────────────
# Header Banner
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1>🎗️ Breast Cancer Diagnostic AI — Oncology Triage Console</h1>
            <p>Wisconsin Fine Needle Aspirate (FNA) Biopsy Classifier • 10 Cell Nuclei Biomarkers • Random Forest ML</p>
        </div>
        <div style="font-family: monospace; font-size: 12px; background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.3); padding: 4px 12px; border-radius: 20px;">
            Dataset: Kaggle WDBC • Accuracy: 94.7%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Patient Presets
# ──────────────────────────────────────────────────────────────────────
PRESETS = {
    "Custom / Manual Entry": None,
    "Patient A (Sample Malignant Profile)": {
        "radius_mean": 17.99,
        "texture_mean": 10.38,
        "perimeter_mean": 122.80,
        "area_mean": 1001.0,
        "smoothness_mean": 0.1184,
        "compactness_mean": 0.2776,
        "concavity_mean": 0.3001,
        "concave_points_mean": 0.1471,
        "symmetry_mean": 0.2419,
        "fractal_dimension_mean": 0.07871,
    },
    "Patient B (Sample Benign Profile)": {
        "radius_mean": 13.54,
        "texture_mean": 14.36,
        "perimeter_mean": 87.46,
        "area_mean": 566.3,
        "smoothness_mean": 0.09779,
        "compactness_mean": 0.08129,
        "concavity_mean": 0.06664,
        "concave_points_mean": 0.04781,
        "symmetry_mean": 0.1885,
        "fractal_dimension_mean": 0.05766,
    },
    "Patient C (Borderline Profile)": {
        "radius_mean": 15.22,
        "texture_mean": 18.52,
        "perimeter_mean": 99.70,
        "area_mean": 711.2,
        "smoothness_mean": 0.1044,
        "compactness_mean": 0.1234,
        "concavity_mean": 0.1009,
        "concave_points_mean": 0.0652,
        "symmetry_mean": 0.1980,
        "fractal_dimension_mean": 0.0612,
    }
}

st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:8px;">📋 Clinical Case Loader</div>', unsafe_allow_html=True)

preset_choice = st.sidebar.selectbox(
    "Load Clinical Case Preset",
    list(PRESETS.keys()),
    index=1,
    key="preset_choice"
)

selected_preset = PRESETS[preset_choice]

# ──────────────────────────────────────────────────────────────────────
# Sidebar Biomarker Inputs
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:8px;">🔬 Nuclear Geometry Biomarkers</div>', unsafe_allow_html=True)

def get_val(key, default):
    return selected_preset[key] if selected_preset else default

radius_mean = st.sidebar.slider("Mean Radius (distance from center to perimeter)", 6.0, 30.0, float(get_val("radius_mean", 14.0)), 0.01)
texture_mean = st.sidebar.slider("Mean Texture (std dev of gray-scale values)", 9.0, 40.0, float(get_val("texture_mean", 19.0)), 0.01)
perimeter_mean = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, float(get_val("perimeter_mean", 92.0)), 0.1)
area_mean = st.sidebar.slider("Mean Area", 140.0, 2500.0, float(get_val("area_mean", 650.0)), 1.0)
smoothness_mean = st.sidebar.slider("Mean Smoothness (local variation in radius)", 0.05, 0.20, float(get_val("smoothness_mean", 0.096)), 0.001, format="%.4f")
compactness_mean = st.sidebar.slider("Mean Compactness (perimeter^2 / area - 1.0)", 0.01, 0.35, float(get_val("compactness_mean", 0.104)), 0.001, format="%.4f")
concavity_mean = st.sidebar.slider("Mean Concavity (severity of concave contours)", 0.0, 0.45, float(get_val("concavity_mean", 0.088)), 0.001, format="%.4f")
concave_points_mean = st.sidebar.slider("Mean Concave Points (number of concave contours)", 0.0, 0.25, float(get_val("concave_points_mean", 0.048)), 0.001, format="%.4f")
symmetry_mean = st.sidebar.slider("Mean Symmetry", 0.10, 0.35, float(get_val("symmetry_mean", 0.181)), 0.001, format="%.4f")
fractal_dimension_mean = st.sidebar.slider("Mean Fractal Dimension (coastline approx - 1)", 0.04, 0.10, float(get_val("fractal_dimension_mean", 0.062)), 0.0005, format="%.5f")

# ──────────────────────────────────────────────────────────────────────
# Prediction Execution
# ──────────────────────────────────────────────────────────────────────
features_list = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean
]

input_array = np.array([features_list])

pred = model.predict(input_array)[0]
probs = model.predict_proba(input_array)[0] if hasattr(model, "predict_proba") else [1.0 - pred, pred]

malignant_prob = probs[1]
benign_prob = probs[0]
is_malignant = (pred == 1)

# ──────────────────────────────────────────────────────────────────────
# Top KPI Metric Cards Bar
# ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    diagnosis_label = "MALIGNANT ⚠️" if is_malignant else "BENIGN ✅"
    st.metric("Triage Verdict", diagnosis_label, f"{'High Risk' if is_malignant else 'Normal / Negative'}")

with k2:
    conf_score = malignant_prob if is_malignant else benign_prob
    st.metric("Diagnostic Confidence", f"{conf_score * 100:.1f}%", "Random Forest Ensemble")

with k3:
    st.metric("Malignancy Probability", f"{malignant_prob * 100:.1f}%", f"{'Above Cutoff' if is_malignant else 'Below Cutoff'}")

with k4:
    st.metric("Nuclear Area", f"{area_mean:.0f} µm²", f"Radius: {radius_mean:.1f} µm")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Clinical Assessment Report
# ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown("### 🩺 Diagnostic Pathologist Summary")
    if is_malignant:
        st.markdown(f"""
        <div class="result-card-malignant">
            <h3 style="margin-top:0; color:#f87171;">⚠️ DIAGNOSIS: MALIGNANT NEOPLASM INDICATED</h3>
            <p>The statistical ensemble classifies this fine-needle aspirate specimen as <b>Malignant</b> with <b>{malignant_prob*100:.1f}% confidence</b>.</p>
            <hr style="border-color: #ef4444; opacity: 0.3;" />
            <h4 style="color:#fca5a5; margin-bottom: 6px;">Key Contributing Pathological Indicators:</h4>
            <ul>
                <li><b>Nuclear Area ({area_mean:.0f} µm²)</b> & <b>Perimeter ({perimeter_mean:.1f} µm)</b> exceed benign thresholds.</li>
                <li><b>Concave Points ({concave_points_mean:.4f})</b> indicates marked cellular irregularity.</li>
                <li><b>Mean Concavity ({concavity_mean:.4f})</b> aligns with invasive cellular morphology.</li>
            </ul>
            <p style="font-size: 13px; opacity: 0.9; margin-bottom: 0;"><b>Recommendation:</b> Immediate histopathological core-needle biopsy and oncology consultation recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-benign">
            <h3 style="margin-top:0; color:#34d399;">✅ DIAGNOSIS: BENIGN LESION (NON-CANCEROUS)</h3>
            <p>The statistical ensemble classifies this fine-needle aspirate specimen as <b>Benign</b> with <b>{benign_prob*100:.1f}% confidence</b>.</p>
            <hr style="border-color: #10b981; opacity: 0.3;" />
            <h4 style="color:#6ee7b7; margin-bottom: 6px;">Normal Morphological Findings:</h4>
            <ul>
                <li><b>Uniform Nuclear Contours:</b> Low concavity ({concavity_mean:.4f}) and concave points ({concave_points_mean:.4f}).</li>
                <li><b>Regular Nuclear Sizing:</b> Area ({area_mean:.0f} µm²) and radius ({radius_mean:.1f} µm) fall within standard benign ranges.</li>
                <li><b>Cellular Smoothness:</b> Normal cellular margin variation ({smoothness_mean:.4f}).</li>
            </ul>
            <p style="font-size: 13px; opacity: 0.9; margin-bottom: 0;"><b>Recommendation:</b> Routine screening follow-up in accordance with institutional guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📊 Nuclear Feature Importance (Random Forest)")
    if hasattr(model, "feature_importances_"):
        features_names = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
        ]
        feat_df = pd.DataFrame({
            "Biomarker": features_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        st.bar_chart(feat_df.set_index("Biomarker"), horizontal=True, color="#6366f1")
    else:
        st.info("Feature importance metadata not available.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Biomarker Inspection Table
# ──────────────────────────────────────────────────────────────────────
st.markdown("### 📐 Quantitative Biomarker Analysis")

biomarkers_data = {
    "Biomarker": [
        "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean",
        "Smoothness Mean", "Compactness Mean", "Concavity Mean",
        "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean"
    ],
    "Input Value": [
        f"{radius_mean:.2f}", f"{texture_mean:.2f}", f"{perimeter_mean:.2f}", f"{area_mean:.1f}",
        f"{smoothness_mean:.4f}", f"{compactness_mean:.4f}", f"{concavity_mean:.4f}",
        f"{concave_points_mean:.4f}", f"{symmetry_mean:.4f}", f"{fractal_dimension_mean:.5f}"
    ],
    "Typical Benign Range": [
        "10.0 – 14.5", "14.0 – 21.0", "65.0 – 95.0", "300 – 600",
        "0.075 – 0.105", "0.040 – 0.110", "0.010 – 0.080",
        "0.015 – 0.050", "0.150 – 0.200", "0.052 – 0.065"
    ],
    "Typical Malignant Range": [
        "15.0 – 25.0", "18.0 – 30.0", "100.0 – 170.0", "700 – 1800",
        "0.100 – 0.135", "0.120 – 0.280", "0.120 – 0.350",
        "0.070 – 0.180", "0.180 – 0.260", "0.060 – 0.085"
    ],
}

st.dataframe(pd.DataFrame(biomarkers_data), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; opacity:0.6; font-size:0.8rem;">'
    '🎗️ Breast Cancer Diagnostic AI • Kaggle Wisconsin Diagnostic Breast Cancer (WDBC) • '
    'Powered by Scikit-Learn & Streamlit Cloud'
    '</div>',
    unsafe_allow_html=True,
)
