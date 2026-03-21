"""
WoundScope – Streamlit Demo App

Run: streamlit run src/app.py
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import torch.nn as nn
import torchvision.models as tvm

from data_loader import WOUND_CLASSES, BODY_LOCATIONS, VAL_TRANSFORM, NUM_LOCATIONS
from utils import GradCAM, overlay_gradcam

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WoundScope",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASELINE_CKPT = "models/baseline_model.pth"
BODY_MAP_PATH   = "dataset/BodyMapAllRGB.png"

WOUND_COLORS = {
    "Diabetic": "#FF6B6B",
    "Pressure": "#4ECDC4",
    "Surgical": "#45B7D1",
    "Venous":   "#96CEB4",
}

WOUND_DESCRIPTIONS = {
    "Diabetic": "Typically on feet/lower legs. Punched-out appearance with surrounding callus.",
    "Pressure": "Over bony prominences from sustained pressure. Ranges from redness to deep tissue damage.",
    "Surgical": "Post-operative incision wounds. May present with dehiscence or delayed healing.",
    "Venous":   "Lower leg (gaiter area). Irregular borders with surrounding skin changes.",
}

LOCATION_LABELS = {
    "head_neck":       "Head / Neck",
    "chest":           "Chest",
    "abdomen":         "Abdomen",
    "back":            "Back",
    "upper_extremity": "Upper Extremity",
    "lower_extremity": "Lower Extremity",
}

st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }

    .disclaimer {
        border-left: 3px solid #f0a500;
        padding: 0.45rem 0.9rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #c8a84b;
        background: rgba(240,165,0,0.08);
        margin-bottom: 0.5rem;
    }

    .pred-card {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .pred-label {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.6;
        margin-bottom: 0.2rem;
    }
    .pred-class {
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .pred-conf {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.3rem 0 0.1rem;
    }
    .pred-sub {
        font-size: 0.78rem;
        opacity: 0.5;
    }
    .pred-desc {
        margin-top: 0.9rem;
        font-size: 0.85rem;
        opacity: 0.8;
        line-height: 1.5;
    }

    .section-label {
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.45;
        margin-bottom: 0.6rem;
    }

    hr { margin: 1rem 0; opacity: 0.15; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(BASELINE_CKPT):
        return None, None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)

    model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(WOUND_CLASSES))
    target_layer = model.layer4[-1]

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    gradcam = GradCAM(model, target_layer)

    return model, gradcam, device, False


@st.cache_data
def preprocess_image(file_bytes):
    pil_img = Image.open(file_bytes).convert("RGB")
    tensor = VAL_TRANSFORM(pil_img).unsqueeze(0)
    return pil_img, tensor


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model, img_tensor, loc_idx, device, is_multimodal, gradcam):
    img_tensor = img_tensor.to(device)
    loc_tensor = torch.tensor([loc_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(img_tensor, loc_tensor) if is_multimodal else model(img_tensor)

    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())

    img_tensor = img_tensor.detach().requires_grad_(True)
    cam, _ = gradcam.generate(img_tensor, class_idx=pred_idx)

    return WOUND_CLASSES[pred_idx], float(probs[pred_idx]), probs, cam


def get_loc_importance(model):
    weights = model.loc_embedding.weight.detach().cpu().numpy()
    norms = np.linalg.norm(weights, axis=1)
    return {BODY_LOCATIONS[i]: float(norms[i]) for i in range(len(BODY_LOCATIONS))}


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────

def prob_chart(probs, pred_class):
    colors = [
        WOUND_COLORS[cls] if cls == pred_class else "rgba(255,255,255,0.15)"
        for cls in WOUND_CLASSES
    ]
    fig = go.Figure(go.Bar(
        x=probs,
        y=WOUND_CLASSES,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=13),
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=40, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, min(max(probs) * 1.25, 1.0)]),
        yaxis=dict(tickfont=dict(size=13)),
        height=160,
        showlegend=False,
    )
    return fig


def loc_chart(importance, selected_loc):
    labels = [LOCATION_LABELS[k] for k in BODY_LOCATIONS]
    values = [importance.get(k, 0) for k in BODY_LOCATIONS]
    sel_label = LOCATION_LABELS[selected_loc]
    colors = [
        "rgba(255,255,255,0.7)" if l == sel_label else "rgba(255,255,255,0.2)"
        for l in labels
    ]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=20, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        height=200,
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("# 🩺 WoundScope")
    st.markdown(
        '<div class="disclaimer">⚠ Research demo only — not a medical device or clinical tool.</div>',
        unsafe_allow_html=True,
    )

    model_result = load_model()
    if model_result[0] is None:
        st.error("No trained model found. Run `train_baseline.py` and `train_multimodal.py` first.")
        return

    model, gradcam, device, is_multimodal = model_result
    device_str = f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    st.caption(f"ResNet50 · image classification · {device_str}")

    st.divider()

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_img, col_loc = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-label">Wound Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded:
            pil_img, img_tensor = preprocess_image(uploaded)
            st.image(pil_img, use_container_width=True)

    with col_loc:
        st.markdown('<div class="section-label">Body Location</div>', unsafe_allow_html=True)
        if os.path.exists(BODY_MAP_PATH):
            st.image(BODY_MAP_PATH, use_container_width=True)
        selected_loc = st.selectbox(
            "location",
            options=list(LOCATION_LABELS.keys()),
            format_func=lambda x: LOCATION_LABELS[x],
            index=5,  # lower_extremity default
            label_visibility="collapsed",
        )
        loc_idx = BODY_LOCATIONS.index(selected_loc)

    # ── Classify ──────────────────────────────────────────────────────────────
    st.divider()
    classify = st.button(
        "Classify Wound",
        type="primary",
        use_container_width=True,
        disabled=(uploaded is None),
    )

    if not classify or uploaded is None:
        return

    # ── Inference ─────────────────────────────────────────────────────────────
    with st.spinner("Running inference..."):
        pred_class, confidence, probs, cam = run_inference(
            model, img_tensor, loc_idx, device, is_multimodal, gradcam
        )

    color = WOUND_COLORS[pred_class]
    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.markdown(f"""
        <div class="pred-card" style="border-color:{color}33; background:{color}0d;">
            <div class="pred-label">Prediction</div>
            <div class="pred-class" style="color:{color};">{pred_class} Wound</div>
            <div class="pred-conf" style="color:{color};">{confidence:.0%}</div>
            <div class="pred-sub">confidence</div>
            <div class="pred-desc">{WOUND_DESCRIPTIONS[pred_class]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Location: {LOCATION_LABELS[selected_loc]}")

        st.markdown('<div class="section-label" style="margin-top:1rem;">Class Probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(prob_chart(probs, pred_class), use_container_width=True, config={"displayModeBar": False})

    with res_right:
        st.markdown('<div class="section-label">Grad-CAM · What the model saw</div>', unsafe_allow_html=True)
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (pil_img.width, pil_img.height), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
        overlay = overlay_gradcam(pil_img, cam_resized)
        st.image(overlay, use_container_width=True)


    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### WoundScope")
        st.markdown("""
**Dataset:** AZH Wound Dataset · 730 images · 4 classes

**Architecture:** ResNet50 (fine-tuned) → 4-class classifier

**Classes:** Diabetic · Pressure · Surgical · Venous

**Test accuracy:** 74% · macro-F1 0.70
        """)
        st.divider()
        st.caption("Research demo only. Not a medical device.")


if __name__ == "__main__":
    main()
