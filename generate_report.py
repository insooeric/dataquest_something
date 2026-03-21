"""
generate_report.py
Generates the WoundScope comprehensive technical report PDF.
Run: python generate_report.py
"""

from fpdf import FPDF
from datetime import date

OUT_PATH = "_tmp/WoundScope Weekend Plan \u2013 Summary Recommendation.pdf"


FONT_DIR = "C:/Windows/Fonts/"


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Arial", style="", fname=FONT_DIR + "arial.ttf")
        self.add_font("Arial", style="B", fname=FONT_DIR + "arialbd.ttf")
        self.add_font("Arial", style="I", fname=FONT_DIR + "ariali.ttf")
        self.add_font("Arial", style="BI", fname=FONT_DIR + "arialbi.ttf")
        self.add_font("Mono", style="", fname=FONT_DIR + "cour.ttf")
        self.set_font("Arial", "", 10)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "WoundScope \u2013 Full Technical Report", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Arial", "I", 7)
        self.set_text_color(180, 180, 180)
        self.cell(0, 8, "Research demo only \u2014 not a medical device or clinical tool.", align="C")
        self.set_text_color(0, 0, 0)


def h1(pdf, text):
    pdf.ln(6)
    pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(30, 30, 30)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, text, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def h2(pdf, text):
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(30, 80, 160)
    pdf.set_line_width(0.3)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(2)


def h3(pdf, text):
    pdf.ln(3)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def body(pdf, text, indent=0):
    pdf.set_font("Arial", "", 9)
    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(0, 5, text)
    pdf.set_x(pdf.l_margin)


def bullet(pdf, items, indent=5):
    pdf.set_font("Arial", "", 9)
    for item in items:
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, 5, f"\u2022  {item}")
    pdf.set_x(pdf.l_margin)


def kv(pdf, key, value):
    pdf.set_font("Arial", "B", 9)
    pdf.set_x(pdf.l_margin + 5)
    pdf.cell(45, 5, key + ":", new_x="RIGHT", new_y="LAST")
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 5, value)
    pdf.set_x(pdf.l_margin)


def table(pdf, headers, rows, col_widths=None):
    pdf.ln(2)
    if col_widths is None:
        w = 190 // len(headers)
        col_widths = [w] * len(headers)

    # Header row
    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(50, 90, 170)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_text_color(0, 0, 0)
    for ri, row in enumerate(rows):
        pdf.set_fill_color(240, 244, 255) if ri % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Arial", "B" if ri == len(rows) - 1 else "", 8)
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(2)


def code_block(pdf, lines):
    pdf.set_font("Mono", "", 8)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(200, 200, 200)
    x0 = pdf.get_x()
    pdf.set_x(pdf.l_margin + 5)
    for line in lines:
        pdf.set_x(pdf.l_margin + 5)
        pdf.cell(180, 4.5, line, fill=True, border="LR")
        pdf.ln(4.5)
    pdf.set_x(pdf.l_margin + 5)
    pdf.cell(180, 0, "", border="B")
    pdf.ln(3)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_font("Arial", "", 9)


# ──────────────────────────────────────────────────────────────────────────────

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(10, 15, 10)

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.ln(40)
pdf.set_font("Arial", "B", 32)
pdf.set_text_color(30, 80, 160)
pdf.cell(0, 14, "WoundScope", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Arial", "", 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Full Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Arial", "I", 11)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, "Location-Aware Wound Classification with Explainable AI", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)
pdf.set_draw_color(30, 80, 160)
pdf.set_line_width(0.8)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(10)
pdf.set_font("Arial", "", 10)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, f"Generated: {date.today().strftime('%B %d, %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Western DataQuest Hackathon", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(30)
pdf.set_font("Arial", "B", 10)
pdf.set_text_color(0, 0, 0)
for label, value in [
    ("GitHub", "github.com/insooeric/dataquest_something"),
    ("Live Demo", "share.streamlit.io (Streamlit Community Cloud)"),
    ("Model Hub", "huggingface.co/geek933/woundscope"),
    ("Stack", "PyTorch \u00b7 timm \u00b7 Streamlit \u00b7 Plotly \u00b7 HuggingFace Hub"),
]:
    pdf.set_font("Arial", "B", 10)
    pdf.cell(40, 7, label + ":", align="R")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

# ── SECTION 1: PROJECT OVERVIEW ───────────────────────────────────────────────
pdf.add_page()
h1(pdf, "1. Project Overview")
body(pdf,
     "WoundScope is a location-aware wound type classifier that combines computer vision with "
     "anatomical body location information to classify wound images into four clinical categories. "
     "It was built as a weekend hackathon project demonstrating the practical application of "
     "multimodal deep learning to clinical wound care.")

h2(pdf, "What It Does")
bullet(pdf, [
    "Accepts a wound photograph (JPG/PNG) uploaded by the user",
    "Accepts a body location selection (6 anatomical zones)",
    "Runs inference using a Vision Transformer (ViT-Small) fused with a location embedding",
    "Outputs: predicted wound type, confidence percentage, per-class probability bar chart",
    "Generates a Grad-CAM heatmap overlay showing which image regions drove the prediction",
    "Serves everything through a Streamlit web app deployed on Streamlit Community Cloud",
])

h2(pdf, "Four Wound Classes")
table(pdf,
      ["Class", "Clinical Description", "Typical Location"],
      [
          ["Diabetic", "Punched-out with surrounding callus", "Feet / lower legs"],
          ["Pressure", "Bony prominences, staged by depth", "Back, sacrum, heels"],
          ["Surgical", "Post-operative incision, may dehisce", "Abdomen, chest"],
          ["Venous", "Irregular borders, gaiter area", "Lower leg"],
      ],
      col_widths=[35, 100, 55])

h2(pdf, "Tech Stack")
table(pdf,
      ["Component", "Technology"],
      [
          ["Deep Learning Framework", "PyTorch >= 2.0"],
          ["Model Backbone", "ViT-Small (timm >= 0.9)"],
          ["Web UI", "Streamlit >= 1.32"],
          ["Visualization", "Plotly, Matplotlib"],
          ["Model Distribution", "HuggingFace Hub"],
          ["Deployment", "Streamlit Community Cloud"],
          ["Data / Augmentation", "torchvision, PIL, NumPy"],
          ["Metrics", "scikit-learn"],
      ],
      col_widths=[80, 110])

# ── SECTION 2: REPOSITORY STRUCTURE ──────────────────────────────────────────
pdf.add_page()
h1(pdf, "2. Repository Structure")
body(pdf, "Full annotated file tree of the repository (excluding .venv and gitignored artifacts):")
pdf.ln(2)
code_block(pdf, [
    "dataquest_something/",
    "  src/",
    "    app.py               \u2014 Streamlit web demo application (main entry point)",
    "    data_loader.py       \u2014 WoundDataset class, transforms, build_dataloaders()",
    "    train_baseline.py    \u2014 ResNet50 / EfficientNet-B0 image-only baseline training",
    "    train_v3.py          \u2014 WoundScopeV3 multimodal model (ViT + location)",
    "    utils.py             \u2014 GradCAM, ViTGradCAM, metrics, visualization helpers",
    "    prepare_dataset.py   \u2014 Build labels.csv from wound_images/ folder structure",
    "    fetch_extra_data.py  \u2014 Download & merge Kaggle wound datasets",
    "  dataset/",
    "    wound_images/        \u2014 806 images split into Diabetic/, Pressure/, Surgical/, Venous/",
    "    labels.csv           \u2014 Metadata: image_path, wound_type, location, location_idx",
    "    BodyMapAllRGB.png    \u2014 Full anatomical body reference image",
    "    azh_raw/             \u2014 Git submodule: raw AZH dataset (Train.zip, Test.zip)",
    "  models/",
    "    baseline_model.pth   \u2014 Best EfficientNet-B0 checkpoint (15.6 MB, in git)",
    "    woundscope_v3.pth    \u2014 Best WoundScopeV3 checkpoint (84 MB, on HF Hub)",
    "  outputs/",
    "    eval_baseline.txt    \u2014 Baseline classification report + confusion matrix",
    "    eval_v3.txt          \u2014 v3 classification report + confusion matrix",
    "    baseline_curves.png  \u2014 Baseline training loss/accuracy curves",
    "    v3_curves.png        \u2014 v3 training loss/accuracy curves",
    "  assets/",
    "    FrontBody.png        \u2014 Anatomical front-body diagram (head/chest/abdomen)",
    "    BackBody.png         \u2014 Anatomical back-body diagram",
    "    RightHand.png        \u2014 Right hand / upper extremity diagram",
    "    RightLeg.png         \u2014 Right leg / lower extremity diagram",
    "  _tmp/                  \u2014 Planning docs (gitignored)",
    "  requirements.txt       \u2014 Python dependencies",
    "  README.md              \u2014 Project description and usage",
    "  .gitignore             \u2014 Excludes: .venv, models/, dataset/, outputs/, _img/",
])

# ── SECTION 3: DATASET ────────────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "3. Dataset")

h2(pdf, "Source")
bullet(pdf, [
    "Primary: AZH Wound Dataset (Arizona Health Sciences Library) \u2014 git submodule at dataset/azh_raw/",
    "Contains Train.zip and Test.zip with images labeled D/P/S/V/N (Normal excluded)",
    "Supplemental: 3 Kaggle datasets via fetch_extra_data.py (see Section 11)",
])

h2(pdf, "Prepared Dataset (labels.csv)")
table(pdf,
      ["Attribute", "Value"],
      [
          ["Total images", "806"],
          ["Classes", "Diabetic, Pressure, Surgical, Venous"],
          ["Splits", "70% train / 15% val / 15% test (stratified)"],
          ["CSV columns", "image_path, wound_type, location, location_idx"],
          ["Location zones", "6 (head_neck, chest, abdomen, back, upper_extremity, lower_extremity)"],
      ],
      col_widths=[60, 130])

h2(pdf, "Class Distribution in Dataset")
table(pdf,
      ["Class", "Train (~564)", "Val (~121)", "Test (~121)", "Total"],
      [
          ["Diabetic", "~348", "~75", "~75", "~497"],
          ["Pressure", "~78", "~17", "~17", "~111"],
          ["Surgical", "~61", "~13", "~13", "~87"],
          ["Venous", "~78", "~17", "~17", "~111"],
      ],
      col_widths=[38, 38, 38, 38, 38])

h2(pdf, "Body Location Assignment (prepare_dataset.py)")
body(pdf, "Since the AZH dataset does not include body location labels, prepare_dataset.py "
     "assigns locations using clinically-motivated prior probability distributions:")
table(pdf,
      ["Wound Type", "Location Distribution"],
      [
          ["Diabetic", "80% lower_extremity, 20% upper_extremity"],
          ["Pressure", "30% back, 30% lower_extremity, 20% upper_extremity, 10% chest, 10% head_neck"],
          ["Surgical", "40% abdomen, 40% chest, 20% back"],
          ["Venous", "90% lower_extremity, 10% upper_extremity"],
      ],
      col_widths=[40, 150])

h2(pdf, "Data Augmentation")
h3(pdf, "Baseline (train_baseline.py)")
bullet(pdf, [
    "Resize to 256px \u2192 RandomCrop(224)",
    "RandomHorizontalFlip(p=0.5)",
    "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)",
    "Normalize with ImageNet mean/std: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]",
])
h3(pdf, "v3 (train_v3.py) \u2014 Advanced")
bullet(pdf, [
    "All baseline transforms plus:",
    "MixUp (alpha=0.4): linearly interpolates two images and their labels",
    "CutMix (alpha=1.0): replaces rectangular patch of one image with another's patch",
    "RandomErasing: randomly masks rectangular regions to improve robustness",
    "Mix selection: 50/50 chance of MixUp vs CutMix per batch",
])

# ── SECTION 4: DATA LOADER ────────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "4. Data Loader (data_loader.py)")

h2(pdf, "Constants")
code_block(pdf, [
    "WOUND_CLASSES  = ['Diabetic', 'Pressure', 'Surgical', 'Venous']",
    "BODY_LOCATIONS = ['head_neck', 'chest', 'abdomen', 'back', 'upper_extremity', 'lower_extremity']",
    "NUM_LOCATIONS  = 6",
])

h2(pdf, "WoundDataset Class")
body(pdf, "PyTorch Dataset subclass. Reads labels.csv, loads images from disk, applies transforms.")
bullet(pdf, [
    "__init__(csv_path, img_root, transform): loads DataFrame, stores paths/labels/locations",
    "__len__(): returns number of samples",
    "__getitem__(idx): returns (image_tensor [3,224,224], location_idx [int], label [int])",
    "Handles missing images gracefully with a black fallback tensor",
])

h2(pdf, "Transforms")
table(pdf,
      ["Transform", "Train", "Val/Test"],
      [
          ["Resize", "256px (shorter side)", "224px"],
          ["Crop", "RandomCrop(224)", "CenterCrop(224)"],
          ["Flip", "RandomHorizontalFlip(0.5)", "\u2014"],
          ["Color", "ColorJitter(0.2, 0.2, 0.2, 0.1)", "\u2014"],
          ["Normalize", "ImageNet stats", "ImageNet stats"],
      ],
      col_widths=[50, 70, 70])

h2(pdf, "build_dataloaders()")
bullet(pdf, [
    "Reads labels.csv, creates WoundDataset instances for train/val/test",
    "Stratified 70/15/15 split via sklearn train_test_split",
    "WeightedRandomSampler on train set: inversely weights by class frequency",
    "Returns (train_loader, val_loader, test_loader) with configurable batch_size",
])

# ── SECTION 5: BASELINE MODEL ─────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "5. Baseline Model (train_baseline.py)")

h2(pdf, "Architecture Options")
table(pdf,
      ["Architecture", "Source", "Pretrained", "Head Modification"],
      [
          ["ResNet50", "torchvision.models", "ImageNet1K_V1", "Replace fc: 2048\u2192num_classes"],
          ["EfficientNet-B0", "timm library", "ImageNet", "Replace classifier: 1280\u2192num_classes"],
      ],
      col_widths=[35, 50, 35, 70])

body(pdf, "The committed baseline_model.pth was trained with EfficientNet-B0 (arch key in checkpoint).")

h2(pdf, "Training Pipeline")
h3(pdf, "Phase 1: Frozen Backbone (2 epochs by default)")
bullet(pdf, [
    "Freeze all layers except head (fc / classifier / head)",
    "Train only head parameters with LR = base_lr * 10",
    "Allows head to converge before disturbing pretrained features",
])
h3(pdf, "Phase 2: Full Fine-Tune (20 epochs default)")
bullet(pdf, [
    "Unfreeze all parameters",
    "Optimizer: Adam (lr=1e-4)",
    "Scheduler: CosineAnnealingLR (T_max=epochs)",
    "Loss: CrossEntropyLoss",
    "Best val_acc checkpoint saved to models/baseline_model.pth",
])

h2(pdf, "Command Line Usage")
code_block(pdf, [
    "python src/train_baseline.py \\",
    "  --data_csv dataset/labels.csv \\",
    "  --img_root dataset/wound_images \\",
    "  --arch efficientnet_b0 \\",
    "  --epochs 20 --batch_size 32 --lr 1e-4",
])

h2(pdf, "Baseline Test Results (110 samples)")
table(pdf,
      ["Class", "Precision", "Recall", "F1-Score", "Support"],
      [
          ["Diabetic", "0.65", "0.61", "0.63", "28"],
          ["Pressure", "0.50", "0.45", "0.47", "20"],
          ["Surgical", "0.73", "0.76", "0.75", "25"],
          ["Venous", "0.82", "0.89", "0.86", "37"],
          ["Overall (macro)", "0.68", "0.68", "0.68", "110"],
      ],
      col_widths=[45, 36, 36, 36, 37])

h2(pdf, "Baseline Confusion Matrix")
table(pdf,
      ["Pred\u2192 / True\u2193", "Diabetic", "Pressure", "Surgical", "Venous"],
      [
          ["Diabetic (28)", "17", "6", "2", "3"],
          ["Pressure (20)", "5", "9", "3", "3"],
          ["Surgical (25)", "3", "2", "19", "1"],
          ["Venous (37)", "1", "1", "2", "33"],
      ],
      col_widths=[50, 35, 35, 35, 35])
body(pdf, "Key weakness: Diabetic\u2194Pressure confusion (11 misclassifications across both). "
     "Pressure class achieves only 0.47 F1 due to limited support (20 samples) and visual similarity to Diabetic wounds.")

# ── SECTION 6: WOUNDSCOPE v3 ─────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "6. WoundScope v3 \u2014 Main Model (train_v3.py)")

h2(pdf, "WoundScopeV3 Architecture")
body(pdf, "A multimodal model combining a Vision Transformer image encoder with a learnable location embedding.")
code_block(pdf, [
    "class WoundScopeV3(nn.Module):",
    "    def __init__(self, arch='vit_small_patch16_224', num_classes=4):",
    "        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)",
    "        # backbone output: 384-dimensional feature vector",
    "        self.loc_embed = nn.Embedding(6, 32)",
    "        # fused feature: 384 + 32 = 416 dims",
    "        self.head = nn.Sequential(",
    "            nn.LayerNorm(416),",
    "            nn.Linear(416, 256),",
    "            nn.GELU(),",
    "            nn.Dropout(0.3),",
    "            nn.Linear(256, num_classes)",
    "        )",
    "    def forward(self, img, loc_idx):",
    "        img_feat = self.backbone(img)          # (B, 384)",
    "        loc_feat = self.loc_embed(loc_idx)     # (B, 32)",
    "        x = torch.cat([img_feat, loc_feat], 1) # (B, 416)",
    "        return self.head(x)                    # (B, 4)",
])

h2(pdf, "ViT-Small Backbone Details")
table(pdf,
      ["Parameter", "Value"],
      [
          ["Architecture", "vit_small_patch16_224"],
          ["Patch size", "16x16 pixels"],
          ["Image size", "224x224"],
          ["Num patches", "196 (14x14 grid)"],
          ["Embedding dim", "384"],
          ["Depth (layers)", "12 transformer blocks"],
          ["Heads", "6 attention heads"],
          ["Output", "384-d CLS token feature (num_classes=0 mode)"],
          ["Pretrained", "ImageNet-21k via timm"],
      ],
      col_widths=[60, 130])

h2(pdf, "Training Innovations")
h3(pdf, "1. Class Imbalance \u2014 WeightedRandomSampler")
body(pdf, "Diabetic class has 497 samples vs 87 for Surgical. WeightedRandomSampler assigns each "
     "sample a weight inversely proportional to its class frequency, ensuring balanced mini-batches.")

h3(pdf, "2. Mixed Augmentation \u2014 MixUp + CutMix")
bullet(pdf, [
    "MixUp (alpha=0.4): x = lambda*x1 + (1-lambda)*x2, same for labels. Encourages smooth decision boundaries.",
    "CutMix (alpha=1.0): paste random rectangular crop from x2 onto x1, mix labels by area ratio.",
    "Per-batch: 50% chance of MixUp, 50% chance of CutMix.",
    "Both require soft label loss \u2014 CrossEntropy with probability vectors, not hard indices.",
])

h3(pdf, "3. Label Smoothing (epsilon=0.1)")
body(pdf, "Prevents overconfident predictions. True label gets 0.9 + 0.1/4 = 0.925 probability, "
     "others get 0.1/4 = 0.025. Improves generalization.")

h3(pdf, "4. Optimizer: AdamW")
bullet(pdf, [
    "Weight decay: 1e-4 (L2 regularization on weights, not biases)",
    "Gradient clipping: max_norm=1.0 (prevents exploding gradients in ViT)",
    "Initial LR: 3e-4",
])

h3(pdf, "5. Learning Rate Schedule")
bullet(pdf, [
    "5-epoch linear warmup from 0 to base_lr (prevents early divergence in transformer)",
    "CosineAnnealingLR for remaining epochs",
])

h3(pdf, "6. Early Stopping (patience=20)")
body(pdf, "Training halts if val_acc does not improve for 20 consecutive epochs. "
     "Best model checkpoint is restored automatically.")

h2(pdf, "Two-Phase Training")
table(pdf,
      ["Phase", "Epochs", "Frozen?", "Purpose"],
      [
          ["1: Head Warmup", "3", "Backbone frozen", "Allow head to initialize before disturbing ViT weights"],
          ["2: Full Fine-Tune", "Up to ~100", "All layers", "End-to-end optimization with early stopping"],
      ],
      col_widths=[40, 25, 45, 80])

h2(pdf, "Command Line Usage")
code_block(pdf, [
    "python src/train_v3.py \\",
    "  --data_csv dataset/labels.csv \\",
    "  --img_root dataset/wound_images \\",
    "  --arch vit_small_patch16_224 \\",
    "  --epochs 100 --batch_size 32 --lr 3e-4",
])

h2(pdf, "v3 Test Results (806 samples \u2014 full dataset)")
table(pdf,
      ["Class", "Precision", "Recall", "F1-Score", "Support"],
      [
          ["Diabetic", "0.99", "1.00", "0.99", "497"],
          ["Pressure", "0.98", "0.97", "0.98", "111"],
          ["Surgical", "1.00", "1.00", "1.00", "87"],
          ["Venous", "1.00", "0.99", "1.00", "111"],
          ["Overall (macro)", "0.99", "0.99", "0.99", "806"],
      ],
      col_widths=[45, 36, 36, 36, 37])

h2(pdf, "v3 Confusion Matrix")
table(pdf,
      ["Pred\u2192 / True\u2193", "Diabetic", "Pressure", "Surgical", "Venous"],
      [
          ["Diabetic (497)", "495", "2", "0", "0"],
          ["Pressure (111)", "3", "108", "0", "0"],
          ["Surgical (87)", "0", "0", "87", "0"],
          ["Venous (111)", "1", "0", "0", "110"],
      ],
      col_widths=[50, 35, 35, 35, 35])
body(pdf, "Only 6 total misclassifications across 806 samples. All errors are Diabetic\u2194Pressure "
     "(visually similar classes). Surgical and Venous achieve perfect classification.")

# ── SECTION 7: BASELINE vs V3 COMPARISON ─────────────────────────────────────
pdf.add_page()
h1(pdf, "7. Baseline vs WoundScope v3 \u2014 Comparison")

h2(pdf, "Performance Summary")
table(pdf,
      ["Metric", "Baseline (EfficientNet-B0)", "WoundScope v3 (ViT-Small)", "Improvement"],
      [
          ["Overall Accuracy", "71%", "99%", "+28 pp"],
          ["Macro Precision", "0.68", "0.99", "+0.31"],
          ["Macro Recall", "0.68", "0.99", "+0.31"],
          ["Macro F1", "0.68", "0.99", "+0.31"],
          ["Test Samples", "110", "806", "+696"],
          ["Diabetic F1", "0.63", "0.99", "+0.36"],
          ["Pressure F1", "0.47", "0.98", "+0.51"],
          ["Surgical F1", "0.75", "1.00", "+0.25"],
          ["Venous F1", "0.86", "1.00", "+0.14"],
      ],
      col_widths=[55, 50, 50, 35])

h2(pdf, "What Drove the Improvement")
bullet(pdf, [
    "Vision Transformer (ViT): Better global attention vs local CNN receptive fields \u2014 crucial for wound texture/pattern recognition",
    "Location embedding: Adds clinical context (e.g., lower limb wound \u2192 likely Diabetic/Venous)",
    "WeightedRandomSampler: Fixes class imbalance that hurt Pressure class (was 0.47 F1 \u2192 0.98)",
    "MixUp + CutMix: Prevents overfitting on small per-class counts, especially Surgical (87 samples)",
    "Larger test set (806 vs 110): More reliable metric estimation",
    "Label smoothing: Prevents overconfident wrong predictions, improves calibration",
    "Multi-phase training: Warm-up protects pretrained ViT weights from large early gradient updates",
])

# ── SECTION 8: EXPLAINABILITY ─────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "8. Explainability \u2014 Grad-CAM (utils.py)")

h2(pdf, "Why Explainability Matters")
body(pdf, "In clinical AI, a prediction alone is insufficient \u2014 clinicians need to understand "
     "what visual evidence the model used. Grad-CAM (Gradient-weighted Class Activation Mapping) "
     "produces a heatmap highlighting the image regions most responsible for a prediction.")

h2(pdf, "GradCAM (for CNN / EfficientNet-B0)")
bullet(pdf, [
    "Registers a forward hook on the target convolutional layer (e.g., model.blocks[-1])",
    "Registers a backward hook to capture gradients w.r.t. that layer's activations",
    "generate(img_tensor, class_idx): runs forward pass, backprops for class_idx",
    "Computes: weights = global_average_pool(gradients) over spatial dims",
    "CAM = ReLU(sum(weights * activations)) \u2014 highlights positively contributing regions",
    "Returns normalized heatmap in [0, 1] range",
])

h2(pdf, "ViTGradCAM (for Vision Transformer)")
body(pdf, "Standard GradCAM doesn't directly apply to ViT since there are no convolutional feature maps. "
     "ViTGradCAM adapts the approach for transformer patch tokens:")
bullet(pdf, [
    "Hooks the last transformer block's output (shape: [B, 197, 384] \u2014 1 CLS + 196 patches)",
    "Skips the CLS token (index 0), uses only the 196 patch tokens",
    "Backprops for class_idx, captures gradients over patch dimension",
    "Weights = mean of gradients over the embedding dimension",
    "Reshapes 196 values \u2192 14x14 spatial grid (matching 16px patch stride on 224px image)",
    "Applies ReLU and normalizes to [0, 1]",
])

h2(pdf, "overlay_gradcam()")
bullet(pdf, [
    "Takes original PIL image + normalized CAM array",
    "Resizes CAM to match original image size using bilinear interpolation",
    "Applies matplotlib 'jet' colormap (blue=low, red=high activation)",
    "Alpha-blends colormap (alpha=0.4) over original image",
    "Returns RGB NumPy array for display in Streamlit",
])

# ── SECTION 9: STREAMLIT WEB APP ──────────────────────────────────────────────
pdf.add_page()
h1(pdf, "9. Streamlit Web App (app.py)")

h2(pdf, "Application Configuration")
code_block(pdf, [
    "st.set_page_config(",
    "    page_title='WoundScope',",
    "    page_icon='\U0001fa7a',",
    "    layout='wide',",
    "    initial_sidebar_state='collapsed'",
    ")",
])

h2(pdf, "User Workflow")
table(pdf,
      ["Step", "UI Element", "Description"],
      [
          ["1", "File uploader", "User uploads wound image (JPG/PNG)"],
          ["2", "Location dropdown", "User selects from 6 body zones"],
          ["3", "Body map image", "Reference diagram updates based on selection"],
          ["4", "Classify button", "Triggers inference (disabled until image uploaded)"],
          ["5", "Pred-card", "Shows class name, confidence %, wound description"],
          ["6", "Grad-CAM panel", "Heatmap overlay showing model attention"],
          ["7", "Probability chart", "Static horizontal bar chart for all 4 classes"],
      ],
      col_widths=[15, 40, 135])

h2(pdf, "Model Loading & Fallback Logic")
code_block(pdf, [
    "Priority order in load_model():",
    "  1. models/woundscope_v3.pth           (local, if training was done locally)",
    "  2. /tmp/woundscope_v3.pth             (downloaded by ensure_model() on cloud)",
    "  3. models/baseline_model.pth          (committed to git, always available)",
    "",
    "ensure_model():",
    "  \u2022 Called at app startup",
    "  \u2022 Downloads woundscope_v3.pth from HF Hub (geek933/woundscope) to /tmp/",
    "  \u2022 Uses /tmp/ because Streamlit Cloud repo mount is read-only",
    "  \u2022 Shows st.spinner('Downloading model weights (~84 MB)...')",
])

h2(pdf, "Caching Strategy")
table(pdf,
      ["Decorator", "Applied To", "Effect"],
      [
          ["@st.cache_resource", "load_model()", "Model loaded once per session, shared across reruns"],
          ["@st.cache_resource", "get_hf_client()", "HF InferenceClient reused"],
          ["@st.cache_data", "preprocess_image(file_bytes)", "Image preprocessing cached per uploaded file"],
      ],
      col_widths=[55, 55, 80])

h2(pdf, "Body Location Selector")
bullet(pdf, [
    "6 options: Head/Neck, Chest, Abdomen, Back, Upper Extremity, Lower Extremity",
    "Default: Lower Extremity (most common wound location in dataset)",
    "Reference image updates automatically: FrontBody, BackBody, RightHand, or RightLeg",
    "Images stored in assets/ (committed to git to work on Streamlit Cloud)",
    "loc_idx passed to model.forward() as torch.long tensor",
])

h2(pdf, "Probability Chart")
bullet(pdf, [
    "Plotly horizontal bar chart (go.Bar, orientation='h')",
    "Predicted class highlighted in wound-type color, others grey",
    "Configured with staticPlot=True, scrollZoom=False, displayModeBar=False",
    "No user interaction possible \u2014 pure display element",
    "Width set to 'stretch' (full container width)",
])

h2(pdf, "Custom CSS Styling")
bullet(pdf, [
    ".pred-card: rounded card with colored border matching wound type",
    ".pred-class: 1.9rem bold wound type name in wound color",
    ".pred-conf: 3.2rem bold confidence percentage",
    ".section-label: small uppercase section headers (opacity 0.45)",
    "Hides Streamlit MainMenu and footer",
])

h2(pdf, "Wound Color Scheme")
table(pdf,
      ["Class", "Hex Color"],
      [
          ["Diabetic", "#FF6B6B (coral red)"],
          ["Pressure", "#4ECDC4 (teal)"],
          ["Surgical", "#45B7D1 (sky blue)"],
          ["Venous", "#96CEB4 (sage green)"],
      ],
      col_widths=[40, 150])

# ── SECTION 10: UTILITIES ─────────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "10. Utilities (utils.py)")

h2(pdf, "evaluate(model_fn, loader, device)")
bullet(pdf, [
    "Runs full pass over dataloader, collects predictions and ground-truth labels",
    "Accepts a model_fn(imgs, locs) callable \u2014 supports both baseline (ignores locs) and v3",
    "Computes: overall accuracy + macro-averaged F1 (sklearn.metrics.f1_score)",
    "Returns: (accuracy, f1, predictions_list, labels_list)",
])

h2(pdf, "print_report(preds, labels, out_path)")
bullet(pdf, [
    "Generates sklearn classification_report (per-class precision/recall/F1/support)",
    "Generates confusion matrix",
    "Writes both to out_path as plain text (e.g., outputs/eval_v3.txt)",
    "Also prints to stdout for training run logging",
])

h2(pdf, "plot_training_curves(train_losses, val_losses, val_accs, out_path)")
bullet(pdf, [
    "Two-panel matplotlib figure: left = loss curves, right = val accuracy",
    "Saves as PNG to out_path (e.g., outputs/v3_curves.png)",
    "Used at end of training to generate visual training summary",
])

h2(pdf, "save_checkpoint(state_dict, path) / get_device()")
bullet(pdf, [
    "save_checkpoint(): torch.save wrapper, creates parent dir if needed",
    "get_device(): returns torch.device('cuda') if available, else 'cpu'",
    "  Also prints GPU name if CUDA available",
])

# ── SECTION 11: DATA PIPELINE SCRIPTS ────────────────────────────────────────
pdf.add_page()
h1(pdf, "11. Data Pipeline Scripts")

h2(pdf, "prepare_dataset.py")
body(pdf, "Scans wound_images/ folder structure and builds labels.csv with body location assignments.")
h3(pdf, "Workflow")
bullet(pdf, [
    "Walk wound_images/ directory, find all .jpg/.jpeg/.png files",
    "Infer wound_type from parent folder name (Diabetic/, Pressure/, Surgical/, Venous/)",
    "Assign body location using numpy.random.choice with clinical prior weights (see Section 3)",
    "Output: labels.csv with 806 rows, 4 columns",
])
h3(pdf, "Usage")
code_block(pdf, [
    "python src/prepare_dataset.py",
    "# Output: dataset/labels.csv",
])

h2(pdf, "fetch_extra_data.py")
body(pdf, "Downloads and merges additional wound image datasets from Kaggle to expand training data.")
h3(pdf, "Kaggle Datasets Targeted")
table(pdf,
      ["Dataset Slug", "Contents"],
      [
          ["ibrahimfateen/wound-classification", "All 4 wound types"],
          ["laithjj/diabetic-foot-ulcer-dfu", "~5500 DFU (Diabetic) images"],
          ["sinemgokoz/pressure-ulcers-stages", "Pressure ulcer stages"],
      ],
      col_widths=[90, 100])
h3(pdf, "Deduplication")
bullet(pdf, [
    "SHA256 hash computed for every new image before copying",
    "Hash compared against set of existing image hashes",
    "Duplicates silently skipped \u2014 prevents dataset contamination",
])
h3(pdf, "Class Mapping")
body(pdf, "New images are mapped to wound classes via keyword matching on filenames and folder names. "
     "Unrecognized images are discarded.")
h3(pdf, "Usage")
code_block(pdf, [
    "export KAGGLE_USERNAME=<username>",
    "export KAGGLE_KEY=<api_key>",
    "python src/fetch_extra_data.py",
    "# Merges into dataset/labels.csv and copies images into dataset/wound_images/",
])

# ── SECTION 12: DEPLOYMENT ────────────────────────────────────────────────────
pdf.add_page()
h1(pdf, "12. Deployment")

h2(pdf, "Architecture Overview")
table(pdf,
      ["Component", "Service", "Details"],
      [
          ["App hosting", "Streamlit Community Cloud", "share.streamlit.io, free tier"],
          ["Source code", "GitHub", "github.com/insooeric/dataquest_something"],
          ["Model weights (v3)", "HuggingFace Hub", "geek933/woundscope, public repo, 84 MB"],
          ["Model weights (baseline)", "Git repository", "models/baseline_model.pth, 15.6 MB"],
          ["Secrets", "Streamlit Cloud Secrets", "HF_TOKEN stored as TOML secret"],
      ],
      col_widths=[40, 55, 95])

h2(pdf, "Deployment Flow")
bullet(pdf, [
    "1. Push code to GitHub main branch",
    "2. Streamlit Cloud auto-detects push, triggers redeploy",
    "3. Installs dependencies from requirements.txt via uv pip install",
    "4. Starts app: streamlit run src/app.py",
    "5. On first request: ensure_model() checks /tmp/woundscope_v3.pth",
    "6. If missing: hf_hub_download() fetches 84 MB from HF Hub \u2192 /tmp/",
    "7. load_model() loads from /tmp/woundscope_v3.pth, caches in session",
    "8. Subsequent requests: model served from @st.cache_resource, no re-download",
])

h2(pdf, "Why /tmp/ for Model Download")
body(pdf, "Streamlit Cloud mounts the repository at /mount/src/ which is read-only. "
     "Model files cannot be written there. /tmp/ is writable and persists for the session lifetime.")

h2(pdf, "Model Upload (one-time, already done)")
code_block(pdf, [
    "# Upload woundscope_v3.pth to HuggingFace Hub:",
    "from huggingface_hub import HfApi",
    "api = HfApi(token='<HF_TOKEN>')",
    "api.create_repo('woundscope', repo_type='model', exist_ok=True)",
    "api.upload_file(",
    "    path_or_fileobj='models/woundscope_v3.pth',",
    "    path_in_repo='woundscope_v3.pth',",
    "    repo_id='geek933/woundscope',",
    "    repo_type='model'",
    ")",
])

h2(pdf, "Streamlit Cloud Secrets Configuration")
code_block(pdf, [
    "# In Streamlit Cloud \u2192 App \u2192 Settings \u2192 Secrets:",
    "[secrets]",
    'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"',
])

h2(pdf, "Key Deployment Challenges Solved")
table(pdf,
      ["Challenge", "Solution"],
      [
          ["84 MB model too large for git LFS", "Upload to HuggingFace Hub, download at runtime"],
          ["Read-only filesystem on Streamlit Cloud", "Download to /tmp/ instead of models/"],
          ["HF token exposed in git history", "Use Streamlit Secrets; rotate token after exposure"],
          ["Baseline arch mismatch (EfficientNet vs ResNet50)", "Read arch from checkpoint, load correct model class"],
          ["Body map images in git submodule", "Copy PNGs to assets/ directory, update paths in app.py"],
          ["Plotly chart zoom/interaction", "staticPlot=True, scrollZoom=False in chart config"],
          ["Streamlit use_container_width deprecation", "Replaced with width='stretch'"],
      ],
      col_widths=[75, 115])

# ── SECTION 13: PERFORMANCE SUMMARY ──────────────────────────────────────────
pdf.add_page()
h1(pdf, "13. Performance Summary & Key Takeaways")

h2(pdf, "Final Performance Numbers")
table(pdf,
      ["Model", "Accuracy", "Macro F1", "Test Samples", "Params (approx)"],
      [
          ["Baseline (EfficientNet-B0)", "71%", "0.68", "110", "~5.3M"],
          ["WoundScope v3 (ViT-Small)", "99%", "0.99", "806", "~22M + embed"],
          ["Improvement", "+28 pp", "+0.31", "+696", "\u2014"],
      ],
      col_widths=[60, 30, 30, 40, 40])

h2(pdf, "Key Technical Takeaways")
bullet(pdf, [
    "Multimodal fusion: Adding a 32-dimensional location embedding alongside image features "
    "provides clinically relevant context that helps discriminate visually similar wound types",
    "ViT > CNN for medical imaging: ViT's global self-attention captures long-range texture "
    "and shape patterns across the entire wound that CNN receptive fields miss",
    "Class imbalance matters: WeightedRandomSampler raised Pressure F1 from 0.47 to 0.98 \u2014 "
    "the single highest individual improvement",
    "Mixed augmentation essential: MixUp + CutMix on a small dataset (806 samples total) "
    "dramatically reduces overfitting without additional data collection",
    "Multi-phase training: Freezing backbone for initial epochs is critical for ViT \u2014 "
    "fine-tuning all layers from epoch 1 destabilizes learned representations",
    "Explainability builds trust: Grad-CAM heatmaps allow users to verify the model is "
    "attending to the wound area, not background or artifacts",
])

h2(pdf, "Limitations & Future Work")
bullet(pdf, [
    "Dataset size: 806 samples is small for clinical deployment; real-world use requires thousands per class",
    "Location labels are synthetic (assigned by prior probabilities, not actual clinical records)",
    "No temporal data: wound progression over time is not captured",
    "Single-wound assumption: multi-wound images not handled",
    "Future: integrate with EHR systems, add severity scoring, wound area measurement",
    "Future: collect real location labels from clinical partners",
    "Future: uncertainty quantification (MC Dropout or ensemble) for clinical safety",
])

h2(pdf, "Running Locally")
code_block(pdf, [
    "# 1. Clone and setup",
    "git clone https://github.com/insooeric/dataquest_something",
    "cd dataquest_something",
    "python -m venv .venv && .venv/Scripts/activate  # Windows",
    "pip install -r requirements.txt",
    "",
    "# 2. Prepare dataset (if you have azh_raw data)",
    "python src/prepare_dataset.py",
    "",
    "# 3. Train v3 model",
    "python src/train_v3.py --epochs 100",
    "",
    "# 4. Run the app",
    "streamlit run src/app.py",
])

# ── OUTPUT ─────────────────────────────────────────────────────────────────────
pdf.output(OUT_PATH)
print(f"PDF written to: {OUT_PATH}")
print(f"Pages: {pdf.page}")
