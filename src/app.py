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
from utils import GradCAM, ViTGradCAM, overlay_gradcam

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WoundScope",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

V3_CKPT       = "models/woundscope_v3.pth"
BASELINE_CKPT = "models/baseline_model.pth"
BODY_MAP_IMAGES = {
    "head_neck":       "dataset/azh_raw/BodyMap/FrontBody.png",
    "chest":           "dataset/azh_raw/BodyMap/FrontBody.png",
    "abdomen":         "dataset/azh_raw/BodyMap/FrontBody.png",
    "back":            "dataset/azh_raw/BodyMap/BackBody.png",
    "upper_extremity": "dataset/azh_raw/BodyMap/RightHand.png",
    "lower_extremity": "dataset/azh_raw/BodyMap/RightLeg.png",
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL  = "mistralai/Mistral-7B-Instruct-v0.1"

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

    .clinical-note {
        border-left: 3px solid rgba(255,255,255,0.2);
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        background: rgba(255,255,255,0.04);
        font-size: 0.88rem;
        line-height: 1.7;
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

    # Try v3 (ViT multimodal) first, fall back to baseline (ResNet50)
    if os.path.exists(V3_CKPT):
        ckpt_path = V3_CKPT
        use_v3 = True
    elif os.path.exists(BASELINE_CKPT):
        ckpt_path = BASELINE_CKPT
        use_v3 = False
    else:
        return None, None, None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if use_v3:
        import timm
        from train_v3 import WoundScopeV3
        arch = ckpt.get("arch", "vit_small_patch16_224")
        model = WoundScopeV3(arch=arch, num_classes=len(WOUND_CLASSES))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        gradcam = ViTGradCAM(model.backbone)
        model_name = f"ViT-Small + location · {arch}"
    else:
        arch = ckpt.get("arch", "resnet50")
        model = tvm.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(WOUND_CLASSES))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        gradcam = GradCAM(model, model.layer4[-1])
        model_name = "ResNet50 (baseline)"

    return model, gradcam, device, use_v3, model_name


@st.cache_resource
def get_hf_client():
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(token=HF_TOKEN)
    except ImportError:
        return None


@st.cache_data
def preprocess_image(file_bytes):
    pil_img = Image.open(file_bytes).convert("RGB")
    tensor = VAL_TRANSFORM(pil_img).unsqueeze(0)
    return pil_img, tensor


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model, img_tensor, loc_idx, device, use_v3, gradcam):
    img_tensor = img_tensor.to(device)
    loc_tensor = torch.tensor([loc_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(img_tensor, loc_tensor) if use_v3 else model(img_tensor)

    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())

    # GradCAM needs gradients
    img_tensor = img_tensor.detach().requires_grad_(True)
    if use_v3:
        # For ViT GradCAM, run through backbone only
        cam, _ = gradcam.generate(img_tensor, class_idx=pred_idx)
    else:
        cam, _ = gradcam.generate(img_tensor, class_idx=pred_idx)

    return WOUND_CLASSES[pred_idx], float(probs[pred_idx]), probs, cam


def generate_clinical_note(wound_type, location, confidence, client):
    """Call HF inference API to generate a clinical summary."""
    if client is None:
        return None

    location_label = LOCATION_LABELS.get(location, location)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise clinical wound care assistant. "
                "Provide brief, accurate clinical notes. Do not use bullet points. "
                "Write in plain paragraph form. 3 sentences maximum."
            ),
        },
        {
            "role": "user",
            "content": (
                f"A wound classification model detected a {wound_type} wound "
                f"at the {location_label} with {confidence:.0%} confidence. "
                f"Write a brief clinical note covering: typical characteristics of this wound type, "
                f"key assessment considerations for this body location, and general care priorities."
            ),
        },
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=HF_MODEL,
            max_tokens=180,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Try smaller fallback model
        try:
            response = client.chat_completion(
                messages=messages,
                model="HuggingFaceH4/zephyr-7b-beta",
                max_tokens=180,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None


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


# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("# 🩺 WoundScope")
    st.markdown(
        '<div class="disclaimer">⚠ Research demo only — not a medical device or clinical tool.</div>',
        unsafe_allow_html=True,
    )

    result = load_model()
    if result[0] is None:
        st.error("No trained model found. Run `train_baseline.py` or `train_v3.py` first.")
        return

    model, gradcam, device, use_v3, model_name = result
    hf_client = get_hf_client()

    device_str = f"cuda ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "cpu"
    st.caption(f"{model_name} · {device_str}")

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
        selected_loc = st.selectbox(
            "location",
            options=list(LOCATION_LABELS.keys()),
            format_func=lambda x: LOCATION_LABELS[x],
            index=5,  # lower_extremity default
            label_visibility="collapsed",
        )
        loc_idx = BODY_LOCATIONS.index(selected_loc)
        body_map_img = BODY_MAP_IMAGES.get(selected_loc)
        if body_map_img and os.path.exists(body_map_img):
            st.image(body_map_img, use_container_width=True)

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
            model, img_tensor, loc_idx, device, use_v3, gradcam
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

    # ── Clinical Note (LLM) ───────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">Clinical Note · AI-generated</div>', unsafe_allow_html=True)

    with st.spinner("Generating clinical note..."):
        note = generate_clinical_note(pred_class, selected_loc, confidence, hf_client)

    if note:
        st.markdown(f'<div class="clinical-note">{note}</div>', unsafe_allow_html=True)
        st.caption("Generated by LLM — for reference only, not clinical advice.")
    else:
        st.caption("Clinical note unavailable (set HF_TOKEN env var or check API access).")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### WoundScope")
        st.markdown(f"""
**Dataset:** AZH Wound Dataset · 4 classes

**Architecture:** {model_name}

**Classes:** Diabetic · Pressure · Surgical · Venous
        """)
        st.divider()
        st.caption("Research demo only. Not a medical device.")


if __name__ == "__main__":
    main()
