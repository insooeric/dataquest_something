"""
WoundScope – Streamlit Demo App

Run: streamlit run src_finetuning/app.py
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_pretrain"))

# Load .env from repo root
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

import torch.nn as nn
import torchvision.models as tvm

from data_loader import (
    WOUND_CLASSES, BODY_LOCATIONS, VAL_TRANSFORM, NUM_LOCATIONS,
    SEVERITY_NAMES_BY_TYPE,
)
from utils import GradCAM, overlay_gradcam

# ── Config ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WoundScope",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

V3_CKPT       = "models/woundscope_v3.pth"
V3_CKPT_TMP   = "/tmp/woundscope_v3.pth"
BASELINE_CKPT = "models/baseline_model.pth"
HF_REPO_ID    = "geek933/woundscope"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"


def ensure_model():
    if os.path.exists(V3_CKPT) or os.path.exists(V3_CKPT_TMP):
        return
    from huggingface_hub import hf_hub_download
    with st.spinner("Downloading model weights (~84 MB)..."):
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="woundscope_v3.pth",
            local_dir="/tmp",
            token=HF_TOKEN or None,
        )


BODY_MAP_IMAGES = {
    "head_neck":       "assets/FrontBody.png",
    "chest":           "assets/FrontBody.png",
    "abdomen":         "assets/FrontBody.png",
    "back":            "assets/BackBody.png",
    "upper_extremity": "assets/RightHand.png",
    "lower_extremity": "assets/RightLeg.png",
}

WOUND_COLORS = {
    "Diabetic":   "#FF6B6B",
    "Pressure":   "#4ECDC4",
    "Surgical":   "#45B7D1",
    "Venous":     "#96CEB4",
    "Arterial":   "#F7A440",
    "Burns":      "#E05C5C",
    "Laceration": "#A78BFA",
}

WOUND_DESCRIPTIONS = {
    "Diabetic":   "Typically on feet/lower legs. Punched-out appearance with surrounding callus.",
    "Pressure":   "Over bony prominences from sustained pressure. Ranges from redness to deep tissue damage.",
    "Surgical":   "Post-operative incision wounds. May present with dehiscence or delayed healing.",
    "Venous":     "Lower leg (gaiter area). Irregular borders with surrounding skin changes.",
    "Arterial":   "Distal extremities. Pale, punched-out appearance; painful, poor pulses.",
    "Burns":      "Thermal injury. Classified by depth: superficial to full-thickness.",
    "Laceration": "Traumatic wound from blunt or sharp force. Variable depth and contamination.",
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
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero ── */
    .hero { padding: 1.5rem 0 1.2rem; margin-bottom: 0.5rem; }
    .hero-title { font-size: 1.9rem; font-weight: 700; color: #f0f6fc; margin: 0 0 0.2rem; }
    .hero-sub   { font-size: 0.85rem; color: #7d8590; margin: 0; }
    .hero-badge {
        display: inline-block; margin-top: 0.7rem;
        padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.71rem; font-weight: 500;
        background: rgba(240,165,0,0.12); color: #e3a84e;
        border: 1px solid rgba(240,165,0,0.25);
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #7d8590;
        margin-bottom: 0.5rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .cam-label {
        display: flex; align-items: center; gap: 0.45rem;
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #7d8590;
        margin-bottom: 0.5rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .cam-dot { width: 7px; height: 7px; border-radius: 50%; background: #f85149; flex-shrink: 0; }

    /* ── Prediction card ── */
    .pred-card {
        border-radius: 12px; border: 1px solid rgba(255,255,255,0.09);
        padding: 1.5rem; margin-bottom: 1rem;
    }
    .pred-label {
        font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #7d8590; margin-bottom: 0.45rem;
    }
    .pred-class { font-size: 1.7rem; font-weight: 700; line-height: 1.15; margin-bottom: 0.8rem; }
    .pred-divider { height: 1px; background: rgba(255,255,255,0.07); margin: 0.9rem 0; }
    .pred-metric-val { font-size: 2.2rem; font-weight: 800; line-height: 1; }
    .pred-metric-lbl { font-size: 0.7rem; color: #7d8590; margin-top: 0.15rem; }
    .pred-desc { margin-top: 0.9rem; font-size: 0.83rem; color: #8b949e; line-height: 1.6; }

    /* ── Badges ── */
    .severity-badge {
        display: inline-flex; align-items: center; gap: 0.3rem;
        margin-top: 0.65rem; padding: 0.28rem 0.75rem;
        border-radius: 6px; font-size: 0.75rem; font-weight: 600; border: 1px solid;
    }
    .severity-dot { width: 6px; height: 6px; border-radius: 50%; }
    .location-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        margin-top: 0.45rem; padding: 0.25rem 0.7rem;
        border-radius: 6px; font-size: 0.75rem; font-weight: 500;
        color: #8b949e; background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
    }

    /* ── Clinical note ── */
    .clinical-note {
        border-left: 2px solid #1f6feb;
        padding: 0.85rem 1rem; border-radius: 0 10px 10px 0;
        background: rgba(31,111,235,0.07);
        font-size: 0.85rem; line-height: 1.75; color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(V3_CKPT):
        ckpt_path, use_v3 = V3_CKPT, True
    elif os.path.exists(V3_CKPT_TMP):
        ckpt_path, use_v3 = V3_CKPT_TMP, True
    elif os.path.exists(BASELINE_CKPT):
        ckpt_path, use_v3 = BASELINE_CKPT, False
    else:
        return None, None, None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if use_v3:
        from finetune import WoundScope
        saved_num_classes = ckpt.get("num_classes", len(WOUND_CLASSES))
        model = WoundScope(num_classes=saved_num_classes)
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError:
            st.error(
                "Checkpoint is from an old architecture. "
                "Please retrain with `train.py` to use the current model."
            )
            return None, None, None, None, None
        model.to(device).eval()
        gradcam    = GradCAM(model, model.backbone.stage4[-1])
        model_name = "WoundCNN-v1 + location (from scratch)"
    else:
        import timm
        arch = ckpt.get("arch", "resnet50")
        if arch == "efficientnet_b0":
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(WOUND_CLASSES))
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            gradcam = GradCAM(model, model.blocks[-1])
        else:
            model = tvm.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(WOUND_CLASSES))
            model.load_state_dict(ckpt["model_state"])
            model.to(device).eval()
            gradcam = GradCAM(model, model.layer4[-1])
        model_name = f"{arch} (baseline)"

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
    tensor  = VAL_TRANSFORM(pil_img).unsqueeze(0)
    return pil_img, tensor


# ── Inference ───────────────────────────────────────────────────────────────────

def run_inference(model, img_tensor, loc_idx, device, use_v3, gradcam):
    img_tensor = img_tensor.to(device)
    loc_tensor = torch.tensor([loc_idx], dtype=torch.long).to(device)

    with torch.no_grad():
        out = model(img_tensor, loc_tensor) if use_v3 else model(img_tensor)

    if isinstance(out, tuple):
        wound_logits, sev_logits = out
    else:
        wound_logits = out
        sev_logits   = None

    probs    = F.softmax(wound_logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())

    sev_idx = None
    if sev_logits is not None:
        sev_idx = int(sev_logits.argmax(dim=1).item())

    # Grad-CAM (run backbone only — no tuple issue)
    img_tensor = img_tensor.detach().requires_grad_(True)
    cam, _ = gradcam.generate(img_tensor, loc_tensor, class_idx=pred_idx)

    return WOUND_CLASSES[pred_idx], float(probs[pred_idx]), probs, cam, sev_idx


def generate_clinical_note(wound_type, location, confidence, severity_label, client):
    if client is None:
        return None

    location_label = LOCATION_LABELS.get(location, location)
    sev_str = f" ({severity_label})" if severity_label else ""
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
                f"A wound classification model detected a {wound_type}{sev_str} wound "
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


# ── Charts ──────────────────────────────────────────────────────────────────────

def prob_chart(probs, pred_class):
    colors = [
        WOUND_COLORS.get(cls, "rgba(255,255,255,0.12)")
        if cls == pred_class else "rgba(255,255,255,0.08)"
        for cls in WOUND_CLASSES
    ]
    fig = go.Figure(go.Bar(
        x=probs,
        y=WOUND_CLASSES,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#8b949e"),
        hoverinfo="skip",
    ))
    bar_height = max(200, len(WOUND_CLASSES) * 34)
    fig.update_layout(
        margin=dict(l=0, r=48, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, min(max(probs) * 1.3, 1.0)]),
        yaxis=dict(tickfont=dict(size=12, color="#8b949e"), tickcolor="rgba(0,0,0,0)"),
        height=bar_height,
        showlegend=False,
        bargap=0.35,
    )
    return fig


# ── Main UI ─────────────────────────────────────────────────────────────────────

def main():
    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🩺 WoundScope</div>
        <div class="hero-sub">AI-assisted wound classification · ViT-Small + location encoding</div>
        <div class="hero-badge">⚠ Research demo only — not a medical device</div>
    </div>
    """, unsafe_allow_html=True)

    ensure_model()
    result = load_model()
    if result[0] is None:
        st.error("No trained model found. Run `train.py` first.")
        return

    model, gradcam, device, use_v3, model_name = result
    hf_client = get_hf_client()

    # ── Inputs ─────────────────────────────────────────────────────────────────
    col_img, col_loc = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown('<div class="section-label">Wound Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded:
            pil_img, img_tensor = preprocess_image(uploaded)
            st.image(pil_img, width='stretch')

    with col_loc:
        st.markdown('<div class="section-label">Body Location</div>', unsafe_allow_html=True)
        selected_loc = st.selectbox(
            "location",
            options=list(LOCATION_LABELS.keys()),
            format_func=lambda x: LOCATION_LABELS[x],
            index=5,
            label_visibility="collapsed",
        )
        loc_idx      = BODY_LOCATIONS.index(selected_loc)
        body_map_img = BODY_MAP_IMAGES.get(selected_loc)
        if body_map_img and os.path.exists(body_map_img):
            st.image(body_map_img, width='stretch')

    # ── Classify ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    classify = st.button(
        "Classify Wound",
        type="primary",
        width='stretch',
        disabled=(uploaded is None),
    )

    if not classify or uploaded is None:
        return

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Running inference..."):
        pred_class, confidence, probs, cam, sev_idx = run_inference(
            model, img_tensor, loc_idx, device, use_v3, gradcam
        )

    color = WOUND_COLORS.get(pred_class, "#888888")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0 0 1.5rem'>", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        desc = WOUND_DESCRIPTIONS.get(pred_class, "")
        loc_name = LOCATION_LABELS[selected_loc]
        st.markdown(
            f'<div class="pred-card" style="border-color:{color}30;background:linear-gradient(135deg,{color}09 0%,rgba(255,255,255,0.02) 100%);">'
            f'<div class="pred-label">Classification Result</div>'
            f'<div class="pred-class" style="color:{color};">{pred_class} Wound</div>'
            f'<div class="pred-divider"></div>'
            f'<div class="pred-metric-val" style="color:{color};">{confidence:.0%}</div>'
            f'<div class="pred-metric-lbl">Confidence</div>'
            f'<div class="location-pill">&#128205; {loc_name}</div>'
            f'<div class="pred-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class Probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(
            prob_chart(probs, pred_class),
            width='stretch',
            config={"displayModeBar": False, "scrollZoom": False, "staticPlot": True},
        )

    with res_right:
        st.markdown('<div class="cam-label"><span class="cam-dot"></span>Grad-CAM · Model attention</div>', unsafe_allow_html=True)
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (pil_img.width, pil_img.height), Image.BILINEAR
            )
        ).astype(np.float32) / 255.0
        overlay = overlay_gradcam(pil_img, cam_resized)
        st.image(overlay, width='stretch')

        # Clinical note
        if hf_client:
            with st.spinner("Generating clinical note..."):
                note = generate_clinical_note(
                    pred_class, selected_loc, confidence, None, hf_client
                )
            if note:
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">Clinical Note</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="clinical-note">{note}</div>', unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### WoundScope")
        st.markdown(f"""
**Model:** {model_name}

**Classes ({len(WOUND_CLASSES)}):** {' · '.join(WOUND_CLASSES)}
        """)
        st.divider()
        st.caption("Research demo only. Not a medical device.")


if __name__ == "__main__":
    main()
