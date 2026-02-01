# Multi-Modal Misinformation Forensics (Hack/Kaggle)

This repo is a **multi-modal misinformation & authenticity forensics** system that can analyze:

- **Text** (headline/caption/transcript): AI-generated vs human-written, and fake-news likelihood
- **Images**: synthetic/manipulated probability (deepfake-style visual artifacts)
- **Text–image consistency**: whether the image and caption semantically match (CLIP)
- **Truth Vault cross-check** (optional): compare against a local archive of verified content embeddings

It includes training scripts for each “detective” model and a **Fusion Judge** that combines all signals into a final verdict.

---

## What’s inside (high level)

**Core inference + orchestration**
- `misinfo_forensics.py` – main inference engine (`MisinfoForensics`) that loads models, runs the 5 signals, and returns a verdict + explanation.

**Interactive UI**
- `forensics_dashboard.py` – Gradio web app for text/image/video analysis.
  - If a video is uploaded, it can optionally extract a transcript with Whisper.

**Training scripts**
- `train_roberta_detective.py` – RoBERTa text classifier training (binary classification).
- `train_ai_head.py` – trains the AI-text head in the dual-head RoBERTa setup.
- `train_cifake_forensics.py` – fine-tunes EfficientNet branch for CIFAKE (REAL vs FAKE images).
- `train_clip_detective.py` – fine-tunes CLIP projections / similarity for consistency.
- `train_fusion_judge.py` – trains the fusion layer (“judge”) using the 5-score vector.
- `training_pipeline.py` – example end-to-end training pipeline using `MisinfoDataset`.

**Dataset utilities**
- `misinformation_dataset.py` – `MisinfoDataset` (text + image + optional video frames) with augmentations.
- `data_manager.py` – harmonizes datasets (CIFAKE + Fakeddit; NewsCLIPpings is currently commented out).

**Local model assets / weights**
- `models/clip-vit-b32/` and `models/roberta-base/` – local HuggingFace-style model folders.
- `*.pth` – trained weights for individual branches and fusion model.

---

## Quickstart (Windows)

### 1) Create environment + install deps

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- `openai-whisper` is optional unless you want video transcription.
- If Whisper is used, this repo uses `imageio-ffmpeg` to avoid requiring a system `ffmpeg` install.

### 2) Run the dashboard

```powershell
python forensics_dashboard.py
```

Then open the local Gradio URL that prints in the terminal.

---

## How the forensics decision works

`MisinfoForensics.analyze(...)` combines five signals:

1. **AI Text Score** (RoBERTa head): probability text is AI-generated
2. **Misinformation Score** (RoBERTa head): probability text is misinformation/fake-news
3. **Deepfake/Image Forensics Score** (EfficientNet): probability image is synthetic/manipulated
4. **CLIP Similarity**: semantic alignment between caption and image
5. **Vault Discrepancy** (optional): match quality vs “Truth Vault” archive embeddings

These are fused into a final binary decision (REAL vs FAKE) by the **fusion layer**.

---

## Inputs supported

### Text
- Headline, caption, post text, or transcript

### Image
- Any typical image format supported by Pillow

### Video
- The dashboard accepts a video file.
- It can extract a transcript using Whisper if installed.

---

## Configuration

### Model paths
This repo ships a local CLIP model folder under:
- `models/clip-vit-b32/`

The code defaults to this relative path.

If you want to use a different CLIP model folder, pass it explicitly:

```python
from misinfo_forensics import MisinfoForensics
forensics = MisinfoForensics(clip_model_dir="models/clip-vit-b32")
```

### Gemini explanations (optional)
`misinfo_forensics.py` can generate richer explanations using Google Gemini.

- Install:
  ```bash
  pip install google-generativeai
  ```
- Create a `.env` file in the repo root with:
  ```
  GOOGLE_API_KEY=your_key_here
  ```

If Gemini isn’t available, the system falls back to rule-based explanations.

### Whisper transcript extraction (optional)
- Whisper is only needed for video transcription.
- You can select the model size via:
  - environment variable: `WHISPER_MODEL` (default: `base`)

---

## Data files in this repo

These filenames are present in the workspace and are commonly used by the training scripts:

- `clip_train.csv`, `clip_val.csv` – training/validation for CLIP consistency training
- `Final_Fusion_Train.csv` – training set for fusion layer (expects columns like `text`, `image_path`, `label`)
- `roberta_train_synthetic.csv` – text training data for RoBERTa scripts
- `WELFake_Dataset.csv` – common fake-news dataset CSV
- `final_test.json`, `text_only.json`, `image_only.json` – example inference/test inputs

---

## Training (typical workflow)

You can train components independently, then train the fusion judge.

### A) Train the text detective(s)
- RoBERTa-based classification:
  ```powershell
  python train_roberta_detective.py
  ```

- AI-text head training:
  ```powershell
  python train_ai_head.py
  ```

### B) Train visual forensics (CIFAKE)
```powershell
python train_cifake_forensics.py
```

### C) Train CLIP detective (consistency)
```powershell
python train_clip_detective.py
```

### D) Train Fusion Judge
```powershell
python train_fusion_judge.py
```

Outputs are saved as `.pth` checkpoints in the repo root by default.

---

## Troubleshooting

### Dashboard exits immediately (common causes)
- **Missing deps**: run `pip install -r requirements.txt`.
- **Torch GPU/CUDA mismatch**: if CUDA isn’t set up, the code will fall back to CPU, but large models will be slower.

### CLIP model path errors
If you see an error about failing to load CLIP:
- Confirm `models/clip-vit-b32/` exists in the repo
- Or pass `clip_model_dir=...` explicitly when constructing `MisinfoForensics`

### Truth Vault file missing
The dashboard uses `guardian_embeddings.pkl` by default. If it is missing, vault matching is simply disabled.

---

## Repo structure (quick map)

- Inference/UI: `misinfo_forensics.py`, `forensics_dashboard.py`
- Training: `train_*.py`, `training_pipeline.py`
- Dataset: `misinformation_dataset.py`, `data_manager.py`
- Models: `models/`, `*.pth`

---

## License / attribution
This repository is intended for research/competition use. Make sure you comply with the licenses/terms of any datasets and pretrained models you use.
