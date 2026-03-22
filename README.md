# WoundScope

AI-assisted wound classification demo. Upload a wound image, select a body location, and get a predicted wound type with confidence score, Grad-CAM attention heatmap, and an AI-generated clinical note.

**[Live Demo](https://dataquestsomething-igpqi77pcnsz4o3garx334.streamlit.app/)**

---

## What it does

- Classifies wounds into 7 types: Diabetic, Pressure, Surgical, Venous, Arterial, Burns, Laceration
- Takes body location as an additional input signal
- Highlights what the model focused on via Grad-CAM
- Generates a brief clinical note using a language model

## Model

Custom CNN built from scratch — ResNet-style backbone (4 stages, 64→512 channels) with location embeddings and multi-task heads for wound classification and severity grading. No pretrained weights. Trained on ~6,600 images across 5 Kaggle datasets. Test accuracy: 96.0% (weighted), AUC-ROC: 0.992.

---

*Research demo only. Not a medical device.*
