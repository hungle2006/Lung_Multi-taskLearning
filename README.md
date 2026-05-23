<div align="center">

<br>

```
██████╗ ███╗   ██╗███████╗██╗   ██╗███╗   ███╗ ██████╗      █████╗ ██╗
██╔══██╗████╗  ██║██╔════╝██║   ██║████╗ ████║██╔═══██╗    ██╔══██╗██║
██████╔╝██╔██╗ ██║█████╗  ██║   ██║██╔████╔██║██║   ██║    ███████║██║
██╔═══╝ ██║╚██╗██║██╔══╝  ██║   ██║██║╚██╔╝██║██║   ██║    ██╔══██║██║
██║     ██║ ╚████║███████╗╚██████╔╝██║ ╚═╝ ██║╚██████╔╝    ██║  ██║██║
╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚═╝
```

### Lung Sound Diagnostic System

*Dual-branch multi-task CNN × QLoRA-finetuned LLM for clinical-grade respiratory analysis*

<br>

![Version](https://img.shields.io/badge/version-5.2.0-0ea5e9?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-3b82f6?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-ef4444?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-0.110%2B-10b981?style=flat-square&logo=fastapi&logoColor=white)
![CUDA](https://img.shields.io/badge/cuda-11.8%2B-76b900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square)

<br>

[**Overview**](#-overview) · [**Architecture**](#-architecture) · [**Quick Start**](#-quick-start) · [**API**](#-api-reference) · [**Training**](#-training-guide) · [**Changelog**](#-changelog)

<br>

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Overview](#-overview) |
| 2 | [System Architecture](#-system-architecture) |
| 3 | [Data Pipeline](#-data-pipeline) |
| 4 | [DualBranch Model](#-dualbranch-multi-task-model) |
| 5 | [QLoRA Fine-Tuned LLM](#-qlora-fine-tuned-llm) |
| 6 | [Dual-Target Grad-CAM](#-dual-target-grad-cam) |
| 7 | [Inference Flow](#-inference-flow-v52) |
| 8 | [Weights & Checkpoints](#-weights--checkpoints) |
| 9 | [Installation](#-installation) |
| 10 | [Quick Start](#-quick-start) |
| 11 | [API Reference](#-api-reference) |
| 12 | [Project Structure](#-project-structure) |
| 13 | [Configuration](#-configuration-reference) |
| 14 | [Training Guide](#-training-guide) |
| 15 | [Evaluation & Metrics](#-evaluation--metrics) |
| 16 | [Deployment](#-deployment) |
| 17 | [Troubleshooting](#-troubleshooting) |
| 18 | [Changelog](#-changelog) |
| 19 | [Citation](#-citation) |
| 20 | [License](#-license) |

---

## 🫁 Overview

PneumoAI is an end-to-end clinical decision-support system that analyzes digital stethoscope recordings to detect and classify respiratory diseases. Two AI components operate in tandem:

| Component | Role | Architecture |
|-----------|------|-------------|
| **DualBranchModel** | Acoustic feature extraction & multi-task classification | ResNet18 + FPN + CrossAttentionFusion + PatientAttention |
| **PneumoGPT** | Structured clinical reasoning & report generation | QLoRA fine-tuned Qwen2.5-7B-Instruct |

<br>

### Event Classifications *(per respiratory cycle)*

```
  ●  Normal    No adventitious sounds detected
  ◐  Crackle   Discontinuous, explosive sounds — fine or coarse
  ◑  Wheeze    Continuous, musical high-pitched sounds
  ●  Both      Co-occurrence of crackle and wheeze (rhonchi)
```

### Disease Classifications *(patient-level)*

```
  ○  Healthy       Normal lung sounds, no pathology             ── LOW severity
  ◔  Infectious    Respiratory infection — pneumonia, bronchitis ── MEDIUM severity
  ●  Obstructive   Obstructive lung disease — COPD, asthma      ── HIGH severity
```

<br>

### Key Capabilities

- **Multi-format audio** — WAV / MP3 / FLAC / OGG / M4A / WEBM, up to 100 MB
- **Cycle segmentation** — overlapping 6-second windows (50% hop) via sliding window
- **Dual-branch inference** — event classification per cycle + patient-level disease aggregation
- **Top-3 cycle selection** — disease-branch Grad-CAM peak activation priority
- **Sequential QLoRA reasoning** — independent 6-step analysis per cycle → majority-vote verdict
- **Real-time REST API** — with Three.js interactive 3D lung frontend

---

flowchart TB

%% =====================================================
%% STYLE
%% =====================================================

classDef stage fill:#0F172A,color:#fff,stroke:#38BDF8,stroke-width:3px
classDef model fill:#111827,color:#fff,stroke:#A78BFA,stroke-width:2px
classDef xai fill:#1F2937,color:#fff,stroke:#FB7185,stroke-width:2px
classDef llm fill:#172554,color:#fff,stroke:#60A5FA,stroke-width:2px
classDef output fill:#052E16,color:#fff,stroke:#4ADE80,stroke-width:2px

%% =====================================================
%% INPUT
%% =====================================================

A["🎧 AUDIO INPUT<br/><br/>
wav · mp3 · flac · ogg · m4a · webm"]

B["⚙️ PREPROCESSING<br/><br/>
• Butterworth Bandpass (100–2000 Hz)<br/>
• Z-score Normalization<br/>
• Sliding Window (6 s / 3 s)<br/>
• Log-Mel Spectrogram"]

C["📦 OUTPUT FEATURES<br/><br/>
N × [1,1,128,188]"]

A --> B --> C

class A,B,C stage

%% =====================================================
%% CNN
%% =====================================================

subgraph CNN["🧠 DUALBRANCHMODEL"]
direction LR

%% Shared Stem
S["SharedStem<br/><br/>
ResNet18<br/>
conv1 → layer2"]

%% Event Branch
E["🔊 EVENT BRANCH<br/><br/>
layer3 + layer4 + FPN<br/>
Embedding: emb_e [256]<br/>
LayerNorm + MLP<br/>
→ 4 Events"]

%% Disease Branch
D["🩺 DISEASE BRANCH<br/><br/>
layer3 + layer4 + FPN<br/>
Embedding: emb_d [256]<br/>
PatientAttention<br/>
LayerNorm + MLP<br/>
→ 3 Diseases"]

%% Fusion
F["🔀 CrossAttentionFusion<br/><br/>
Bidirectional Multi-Head Attention<br/>
4 Attention Heads"]

S --> E
S --> D

E --> F
D --> F

end

C --> S

class S,E,D,F model

%% =====================================================
%% OUTPUTS
%% =====================================================

P1["📊 EVENT PROBABILITIES<br/><br/>
Per Respiratory Cycle"]

P2["📊 DISEASE PROBABILITIES<br/><br/>
Patient Level"]

F --> P1
F --> P2

class P1,P2 output

%% =====================================================
%% XAI
%% =====================================================

subgraph XAI["🔥 EXPLAINABLE AI"]
direction TB

X1["Dual-Target Grad-CAM"]

X2["cam_pred · cam_alt · cam_diff"]

X3["Top-3 Cycle Selector<br/><br/>
Abnormal cycles prioritized"]

X1 --> X2 --> X3

end

P1 --> X1
P2 --> X1

class X1,X2,X3 xai

%% =====================================================
%% QLORA
%% =====================================================

subgraph LLM["🤖 QLORA REASONING ENGINE"]
direction TB

Q0["Qwen2.5-7B-Instruct<br/>
QLoRA Fine-tuned"]

Q1["6-Step Clinical Reasoning<br/><br/>
1. Event Assessment<br/>
2. Disease Assessment<br/>
3. Retrieval Evaluation<br/>
4. Prototype Similarity<br/>
5. Conflict Detection<br/>
6. Final Conclusion"]

Q0 --> Q1

end

X3 --> Q0

class Q0,Q1 llm

%% =====================================================
%% FINAL
%% =====================================================

V1["🗳️ MAJORITY VOTE<br/><br/>
3 Cycles → Final Verdict"]

V2["📄 JSON RESPONSE"]

V3["🌐 THREE.JS FRONTEND"]

Q1 --> V1 --> V2 --> V3

class V1,V2,V3 output

## 🔊 Data Pipeline

### Audio Preprocessing Flow

```
  Raw Audio Bytes
       │
       ▼
  librosa.load(sr=16 000, mono=True)
  ├─ Resample to 16 kHz
  └─ Force mono
       │
       ▼
  Butterworth Bandpass Filter
  ├─ Order  : 5th
  ├─ Pass   : 100 – 2 000 Hz
  └─ Method : zero-phase filtfilt
       │
       ▼
  Z-score Normalization
  └─ (x − μ) / (σ + 1e-8)
       │
       ▼
  Sliding Window Segmentation
  ├─ Window : 6 s  =  96 000 samples
  └─ Hop    : 3 s  =  48 000 samples  (50 % overlap)
       │
       ▼
  Log-Mel Spectrogram  (per segment)
  ├─ n_fft      =  1 024
  ├─ hop_length =    512
  ├─ n_mels     =    128
  ├─ f_min      =     50 Hz
  ├─ f_max      =  4 000 Hz
  ├─ power      =    2.0
  └─ log_mel    =  log(mel + 1e-6)  →  per-segment z-norm
       │
       ▼
  Tensor  [1, 1, 128, 188]   (B, C, F, T)
```

### Spectrogram Parameters

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| Sample rate | 16 000 Hz | Standard medical audio |
| Window duration | 6 s | One full respiratory cycle |
| Hop duration | 3 s | 50 % overlap |
| FFT size | 1 024 | ≈ 64 ms resolution |
| Hop length | 512 samples | ≈ 32 ms frame shift |
| Mel bins | 128 | Clinical frequency resolution |
| Time frames | 188 | ⌊(96 000 − 1 024) / 512⌋ + 1 |
| Output shape | `[1, 1, 128, 188]` | (B, C, F, T) |

> **Design rationale** — The bandpass filter removes sub-100 Hz body-motion artifacts and high-frequency noise outside the clinical stethoscope range. `fmax=4 000 Hz` covers the full diagnostic band of crackles (200–2 000 Hz) and wheezes (100–1 000 Hz). The 50 % overlapping window guarantees that short, transient events straddling window boundaries are always captured at least once.

### Edge Cases

| Condition | Handling |
|-----------|----------|
| Recording < 6 s | Tile-repeat the signal to fill exactly 6 s |
| Single short cycle | Produces exactly one cycle from the full signal |
| Long recordings | `⌊(len − win) / hop⌋ + 1` cycles produced |

---

## 🧠 DualBranch Multi-Task Model

### Task Overview

| Task | Granularity | Labels | Classes |
|------|-------------|--------|---------|
| **Event classification** | Per respiratory cycle | Normal · Crackle · Wheeze · Both | 4 |
| **Disease classification** | Per patient (aggregated) | Healthy · Infectious · Obstructive | 3 |

> **Key insight** — Event and disease labels are *independent*. A "Normal" cycle does not imply a "Healthy" patient — a COPD patient may have silent intervals between wheeze episodes.

### Module Architecture

```
Input  [B, 1, 128, 188]
    │
    ▼
┌──────────────────────────────────────────────┐
│  SharedStem   (ResNet18  conv1 → layer2)     │
│  ├─ c2  [B,  64, 32, 47]                    │
│  └─ c3  [B, 128, 16, 24]                    │
└───────────────────┬──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌────────────────┐     ┌────────────────┐
│  EventBranch   │     │ DiseaseBranch  │
│  layer3 + 4    │     │  layer3 + 4   │
│     + FPN      │     │     + FPN     │
│                │     │               │
│  c4 [B,256,8,12]     │  c4 [B,256,8,12]
│  c5 [B,512,4, 6]     │  c5 [B,512,4, 6]
│    ↓ FPN merge │     │   ↓ FPN merge │
│  emb_e  [256]  │     │  emb_d  [256] │
└───────┬────────┘     └────────┬──────┘
        └──────────┬────────────┘
                   ▼
    ┌──────────────────────────────┐
    │     CrossAttentionFusion     │
    │    bidirectional  MHA 4h     │
    │                              │
    │  e_ctx  = Attn(Q=e, K=d, V=d)│
    │  gate_e = σ(Linear([e,e_ctx]))│
    │  emb_e* = LN(e + gate_e·e_ctx)│
    │                              │
    │  d_ctx  = Attn(Q=d, K=e, V=e)│
    │  gate_d = σ(Linear([d,d_ctx]))│
    │  emb_d* = LN(d + gate_d·d_ctx)│
    └──────────┬──────────┬────────┘
               │          │
         emb_e*│          │emb_d*
               ▼          ▼
    ┌──────────────┐   ┌────────────────────┐
    │  EventHead   │   │  PatientAttention  │
    │  LN → MLP    │   │  softmax weights   │
    │  → 4 classes │   │  over N cycles     │
    └──────────────┘   └─────────┬──────────┘
                                 ▼
                        ┌────────────────┐
                        │  DiseaseHead   │
                        │  LN → MLP      │
                        │  → 3 classes   │
                        └────────────────┘
```

### Classification Head Architecture

```
LayerNorm(256)
    → Linear(256 → 256)  GELU  Dropout(0.4)
    → Linear(256 → 128)  GELU  Dropout(0.3)
    → Linear(128 → n_classes)
```

### Parameter Summary

| Module | Parameters |
|--------|-----------|
| SharedStem | ~1.7 M |
| EventBranch (BranchUpper + FPN) | ~8.4 M |
| DiseaseBranch (BranchUpper + FPN) | ~8.4 M |
| CrossAttentionFusion | ~0.5 M |
| PatientAttention | ~33 K |
| EventHead | ~100 K |
| DiseaseHead | ~100 K |
| **Total** | **~19.2 M** |

---

## 💬 QLoRA Fine-Tuned LLM

### Model Configuration

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | QLoRA (4-bit NF4 via `bitsandbytes`) |
| LoRA rank / alpha | 16 / 32 |
| Target modules | `q_proj` `v_proj` `k_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| Training dtype | `bfloat16` |
| Trainable params | 83.9 M of 7 699 M (1.09 %) |
| Max sequence length | 1 024 tokens |
| Max new tokens | 800 |

### Sequential Per-Cycle Architecture (v5.2)

```
  v5.1  ──  [Cycle 1 + Cycle 2 + Cycle 3] ──▶  single QLoRA call  ──▶  1 result
                              context interference ✗  token budget ✗

  v5.2  ──  Cycle 1  ──▶  QLoRA call 1  ──▶  result_1
            Cycle 2  ──▶  QLoRA call 2  ──▶  result_2
            Cycle 3  ──▶  QLoRA call 3  ──▶  result_3
                                │
                    ┌───────────▼───────────┐
                    │   Majority Vote        │
                    │   dis_correct          │
                    │   final_disease_label  │
                    │   clinicalNote         │
                    └───────────────────────┘
```

**Advantages of sequential independence:**
- Each call fits within 1 024 tokens — no truncation
- Zero context interference between cycles — each analysis is unbiased
- Majority vote provides robustness against single noisy cycle predictions
- Enables per-cycle attribution in the UI

### 6-Step Clinical Reasoning Output

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1 │ Event Branch Assessment                               │
│         │ Confidence · entropy · margin for event reliability   │
├─────────────────────────────────────────────────────────────────┤
│  Step 2 │ Disease Branch Assessment                             │
│         │ CAM activation bands vs clinical expectations         │
├─────────────────────────────────────────────────────────────────┤
│  Step 3 │ Retrieval Signal Evaluation                           │
│         │ Top class · sim gap · ambiguity threshold 0.05        │
├─────────────────────────────────────────────────────────────────┤
│  Step 4 │ Prototype Cosine Similarity                           │
│         │ Alignment with learned disease cluster prototypes     │
├─────────────────────────────────────────────────────────────────┤
│  Step 5 │ Conflict Identification                               │
│         │ CNN  vs  retrieval  vs  prototype — cross-signal check│
├─────────────────────────────────────────────────────────────────┤
│  Step 6 │ Final Conclusion                                      │
│         │ dis_correct flag · final_label · clinicalNote         │
└─────────────────────────────────────────────────────────────────┘
```

### Majority Vote Aggregation

```python
# dis_correct  → majority among  [True, False, None]
n_correct   = sum(1 for v in votes if v is True)
n_incorrect = sum(1 for v in votes if v is False)
agg = True if n_correct > n_incorrect else (
      False if n_incorrect > n_correct else None)

# final_disease_label  → most common label
from collections import Counter
agg_label = Counter(final_labels).most_common(1)[0][0]
```

> **Important** — The CNN disease prediction is **never overridden** by QLoRA. The LLM acts as a second-opinion layer. If it detects a discrepancy, it raises a `qlora_alt_diagnosis` flag for clinical review.

---

## 🔍 Dual-Target Grad-CAM

### Motivation

Standard Grad-CAM highlights regions important for a single class. Dual-target extends this with a **contrast map** for diagnostic specificity:

```
  cam_pred  =  Grad-CAM( target = predicted_disease )
  cam_alt   =  Grad-CAM( target = alt_disease       )
  cam_diff  =  cam_pred − cam_alt

  High cam_diff  →  region is selectively important for the prediction
                    (not just generally activated)
```

### CAM Feature Extraction

10 scalar features are extracted per CAM map and fed to the LLM:

| Feature | Description | Clinical Signal |
|---------|-------------|-----------------|
| `freq_high` | Mean activation — top ⅓ Mel bins | Broadband crackle bursts |
| `freq_mid` | Mean activation — mid ⅓ Mel bins | Narrow-band wheeze obstruction |
| `freq_low` | Mean activation — bottom ⅓ Mel bins | Low-frequency rumbles |
| `time_early / mid / late` | Temporal activation thirds | Event timing within cycle |
| `peak` | Maximum CAM value | Prediction confidence anchor |
| `std` | Standard deviation | Spatial spread |
| `entropy` | Shannon entropy of normalized CAM | Focal vs diffuse activation |
| `hot_ratio` | Fraction of pixels > 0.6 | Localization sharpness |

```
  High freq_mid  +  low freq_high   →  wheeze pattern
  High freq_high +  scattered hot   →  crackle pattern
  High diff_peak                    →  strong discriminative localization
  Low diff_abs   +  high diff_ent   →  diffuse / ambiguous prediction
```

---

## ⚙️ Inference Flow (v5.2)

```
  1  Receive audio bytes
     │
  2  load_wav_bytes()
     └─▶  N × [1,1,128,188]  Log-Mel tensors
     │
  3  Per-cycle CNN forward pass
     └─▶  emb_e_i, emb_d_i, ev_logits_i  for each i
     │
  4  Patient-level disease
     └─▶  stacked [N,256]  →  patient_disease()  →  pred_disease
     │
  5  Dual-target Grad-CAM per cycle
     ├─▶  cam_event_pred
     ├─▶  cam_dis_pred, cam_dis_alt
     └─▶  cam_diff = cam_dis_pred − cam_dis_alt
     │
  6  Top-3 cycle selection
     ├─  Sort abnormal cycles  (event ≠ Normal)  by cam_disease.peak ↓
     └─  Fill slots with normal cycles  by cam_disease.peak ↓
     │
  7  QLoRA sequential (for each cycle in top-3, independently)
     ├─▶  build_input_text_single_cycle()
     ├─▶  tokenize  →  generate (max_new_tokens=800, greedy)
     └─▶  parse_qlora_output_v2()  →  parsed_i
     │
  8  Majority vote aggregation
     └─▶  aggregate_cycle_qlora_results([p1, p2, p3])
     │
  9  Return JSON response
```

---

## 📦 Weights & Checkpoints

### CNN DualBranch Checkpoints

| Checkpoint | Training Stage | Val F1 | Size | Download |
|------------|---------------|--------|------|---------|
| `best_stage3_f10.6032.pth` | Stage 3 — full fine-tune | **0.6032** | ~75 MB | [Google Drive](#) · [HuggingFace](#) |
| `best_stage2_f10.5891.pth` | Stage 2 — branch heads | 0.5891 | ~75 MB | [Google Drive](#) · [HuggingFace](#) |
| `best_stage1_f10.5412.pth` | Stage 1 — heads only | 0.5412 | ~75 MB | [Google Drive](#) · [HuggingFace](#) |

```bash
mkdir -p weights/
wget -O weights/best_stage3_f10.6032.pth "https://YOUR_DOWNLOAD_URL/..."
```

### QLoRA Adapter

| Adapter | Base Model | Epochs | Download |
|---------|-----------|--------|---------|
| `lora_adapter_20260507_1431` | Qwen2.5-7B-Instruct | 3 | [Google Drive](#) · [HuggingFace](#) |

```
weights/lora_adapter_20260507_1431/
├── adapter_config.json           ← LoRA hyperparameters
├── adapter_model.safetensors     ← weight deltas (~120 MB)
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.model
```

### Override Paths

```bash
export CHECKPOINT_PATH="./weights/best_stage3_f10.6032.pth"
export QLORA_ADAPTER_PATH="./weights/lora_adapter_20260507_1431"
```

---

## 🔧 Installation

### Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10 |
| CUDA | 11.8 | 12.x |
| GPU VRAM | 16 GB | 24 GB |
| System RAM | 16 GB | 32 GB |
| Disk space | 20 GB | — |

### Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/pneumoai.git
cd pneumoai

# 2. Virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. PyTorch with CUDA
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# 4. All dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```text
# Deep learning
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# QLoRA / LLM
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.28.0

# Audio
librosa>=0.10.0
soundfile>=0.12.0
scipy>=1.11.0

# API
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9

# Utilities
numpy>=1.24.0
matplotlib>=3.8.0
Pillow>=10.0.0
nest-asyncio>=1.6.0
pyngrok>=7.0.0          # optional, for Colab tunnel
```

### Google Colab

```python
!pip install -q \
  torch torchvision torchaudio \
  transformers peft bitsandbytes accelerate \
  librosa soundfile scipy \
  fastapi uvicorn python-multipart \
  numpy matplotlib Pillow nest-asyncio pyngrok

from google.colab import drive
drive.mount('/content/drive')

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🚀 Quick Start

### Start the Server

```bash
export CHECKPOINT_PATH="./weights/best_stage3_f10.6032.pth"
export QLORA_ADAPTER_PATH="./weights/lora_adapter_20260507_1431"

python main_v5_2.py
# → Ready at http://0.0.0.0:8000  (init takes ~30–60 s on GPU)
```

### Python Client

```python
import requests

with open("patient_recording.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/analyze",
        files={"file": ("recording.wav", f, "audio/wav")}
    )

r = response.json()["result"]
print(f"Dominant event   : {r['dominantEvent']}")
print(f"Disease (CNN)    : {r['cnn_pred_disease']}  ({r['cnn_confidence']} %)")
print(f"QLoRA verdict    : {r['qlora_dis_correct']}")
print(f"Clinical note    : {r['clinicalNote']}")
print(f"Processing time  : {r['processing_time_ms']} ms")
```

### cURL

```bash
# Full pipeline (CNN + QLoRA)
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -F "file=@patient_recording.wav;type=audio/wav"

# Lightweight CNN-only (faster)
curl -X POST "http://localhost:8000/analyze/top3" \
     -F "file=@recording.wav"
```

---

## 📡 API Reference

### `POST /analyze` — Full Pipeline

```
Request   multipart/form-data
          file  UploadFile  WAV/MP3/FLAC/OGG/M4A/WEBM  max 100 MB

Response  application/json
```

<details>
<summary><b>Response schema (click to expand)</b></summary>

```jsonc
{
  "result": {
    // ── Event ──────────────────────────────────────────────────────────
    "dominantEvent":     "Wheeze",
    "eventCounts":       { "Normal": 2, "Wheeze": 5 },
    "soundType":         "wheeze",

    // ── CNN Disease ────────────────────────────────────────────────────
    "cnn_pred_disease":  "Obstructive",
    "cnn_confidence":    82,
    "cnn_alt_disease":   "Infectious",
    "primaryDiagnosis": {
      "name":        "Obstructive Lung Disease",
      "probability": 82,
      "severity":    "high"
    },
    "differentials": [
      { "name": "Respiratory Infection", "probability": 13 },
      { "name": "Normal",               "probability":  5 }
    ],

    // ── QLoRA aggregated ───────────────────────────────────────────────
    "qlora_dis_correct":         true,
    "clinicalNote":              "Lung sounds suggest chronic obstructive...",
    "llm_source":                "qlora",
    "qloraAlternativeDiagnosis": null,

    // ── Per-cycle QLoRA details ────────────────────────────────────────
    "qlora_per_cycle": [
      {
        "cycle_index":  3,
        "cycle_rank":   1,
        "start_sec":    6.0,
        "end_sec":     12.0,
        "dis_correct":  true,
        "final_label":  "Obstructive",
        "qlora_steps": {
          "step1_event":      "...",
          "step2_disease":    "...",
          "step3_retrieval":  "...",
          "step4_prototype":  "...",
          "step5_conflict":   "...",
          "step6_conclusion": "..."
        }
      }
      // ... cycles 2 and 3
    ],

    // ── Top-3 cycles with CAM data ─────────────────────────────────────
    "top3_cycles": [
      {
        "cycle_index": 3, "rank": 1,
        "event": "Wheeze", "event_confidence": 0.8712,
        "cam_event":       { "freq_high": 0.23, "freq_mid": 0.51, "peak": 0.87 },
        "cam_disease":     { "peak": 0.92, "entropy": 1.84 },
        "cam_diff":        { "diff_peak": 0.51 },
        "gradcam_image_url": "/static/gradcam/abc123_c02.png"
      }
    ],

    // ── Metadata ───────────────────────────────────────────────────────
    "processing_time_ms": 8420,
    "totalCycles":        7,
    "audioDuration":      21.3,
    "model_version":      "DualBranch-v5.2",
    "qlora_mode":         "sequential_per_cycle_independent"
  }
}
```

</details>

### `POST /analyze/top3` — CNN Only (Lightweight)

```jsonc
{
  "disease": {
    "name":       "Obstructive",
    "confidence": 82,
    "severity":   "high",
    "probs": { "Healthy": 5.0, "Obstructive": 82.0, "Infectious": 13.0 }
  },
  "top3_cycles":          [ /* ... */ ],
  "processing_time_ms":   1240
}
```

### `GET /health`

```jsonc
{
  "status":        "ok",
  "model_loaded":  true,
  "qlora_loaded":  true,
  "qlora_mode":    "sequential_per_cycle_independent",
  "version":       "5.2.0"
}
```

---

## 🗂 Project Structure

```
pneumoai/
│
├── main_v5_2.py                      ← main application (all logic)
├── requirements.txt
├── .env.example
├── README.md
│
├── static/
│   └── gradcam/
│       └── {request_id}_c{N}.png     ← auto-generated CAM images
│
├── weights/
│   ├── best_stage3_f10.6032.pth
│   └── lora_adapter_20260507_1431/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer.*
│
├── data/
│   └── ICBHI_2017/
│       ├── audio/
│       └── labels/
│
├── training/
│   ├── train_cnn.py
│   ├── train_qlora.py
│   └── evaluate.py
│
└── notebooks/
    └── demo_inference.ipynb
```

---

## ⚙️ Configuration Reference

All constants are defined at the top of `main_v5_2.py` and overridable via environment variables:

| Constant | Default | Env Variable | Description |
|----------|---------|-------------|-------------|
| `PORT` | `8000` | `PORT` | FastAPI server port |
| `TARGET_SR` | `16000` | — | Audio sample rate (Hz) |
| `TARGET_LENGTH_SEC` | `6` | — | Window duration (s) |
| `N_MELS` | `128` | — | Mel filterbank bins |
| `HOP_LENGTH` | `512` | — | STFT hop length (samples) |
| `N_FFT` | `1024` | — | STFT window size (samples) |
| `FMIN` / `FMAX` | `50` / `4000` | — | Mel frequency bounds (Hz) |
| `QLORA_MAX_SEQ_LEN` | `1024` | — | Max tokenized input length |
| `QLORA_MAX_NEW_TOKENS` | `800` | — | Max new tokens per LLM call |
| `TOP_CYCLES_FOR_DISEASE` | `3` | — | Cycles fed to QLoRA |
| `TOP_K` | `5` | — | Soft retrieval neighbors |
| `RETRIEVAL_GAP_AMBIGUOUS` | `0.05` | — | Ambiguity threshold |

---

## 🏋️ Training Guide

### CNN — 3-Stage Progressive Unfreezing

```
Stage 1  (5–10 epochs)   Freeze backbone → train heads only          lr = 1e-3
Stage 2  (10–20 epochs)  Unfreeze branches + fusion + attention      lr = 1e-4
Stage 3  (20–50 epochs)  Unfreeze all layers including SharedStem    lr = 5e-5
                                                                  CosineAnnealingLR
```

**Multi-task loss:**

```python
loss = 0.5 * CrossEntropyLoss(event_logits,   event_labels  ) \
     + 0.5 * CrossEntropyLoss(disease_logits, disease_labels)
```

### QLoRA Fine-Tuning

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","v_proj","k_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

# trainable params: 83,886,080 / 7,699,431,424  (1.09 %)
```

**Training data format:**

```jsonc
{
  "instruction": "<alpaca system instruction>",
  "input":       "<build_input_text_single_cycle() output>",
  "output":      "**Step 1: ...\n\n**Step 2: ...\n\n...\n\n**Step 6: ...**"
}
```

**Recommended training config:**

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
)
```

### Recommended Datasets

| Dataset | Cycles / Recordings | Labels | Source |
|---------|-------------------|--------|--------|
| ICBHI 2017 | 6 898 cycles | Events + Diseases | [bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr/) |
| SPRSound | 2 683 recordings | Events | [GitHub](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) |
| HF Lung V1 | 9 765 recordings | Events | [HuggingFace](https://huggingface.co/datasets/hf-lung) |

---

## 📊 Evaluation & Metrics

### Benchmark Results — Stage 3 Checkpoint

| Task | Macro F1 | Weighted F1 | ICBHI Score |
|------|----------|------------|------------|
| Event (4 classes) | **0.603** | 0.641 | 0.588 |
| Disease (3 classes) | **0.612** | 0.659 | — |

### Metric Definitions

| Metric | Formula | Use Case |
|--------|---------|---------|
| Macro F1 | `Σ F1_i / N` | Primary — treats all classes equally |
| Weighted F1 | `Σ (support_i × F1_i) / total` | Imbalanced class distributions |
| Sensitivity (SE) | `TP / (TP + FN)` | Per-class recall |
| Specificity (SP) | `TN / (TN + FP)` | Per-class specificity |
| ICBHI Score | `(SE + SP) / 2` | Standard challenge metric |

```bash
# Evaluate CNN
python training/evaluate.py \
  --checkpoint weights/best_stage3_f10.6032.pth \
  --test_dir   data/ICBHI_2017/test

# Evaluate QLoRA step parsing
python training/evaluate_qlora.py \
  --adapter   weights/lora_adapter_20260507_1431 \
  --test_file data/qlora_test.jsonl
```

---

## 🚢 Deployment

### Local

```bash
python main_v5_2.py
# → http://localhost:8000
```

### Google Colab + ngrok

```python
import os
os.environ["NGROK_AUTH_TOKEN"]   = "your_token"
os.environ["CHECKPOINT_PATH"]    = "/content/drive/MyDrive/best_stage3_f10.6032.pth"
os.environ["QLORA_ADAPTER_PATH"] = "/content/drive/MyDrive/lora_adapter_20260507_1431"

exec(open("main_v5_2.py").read())
# Public URL printed after ~2 s
```

### Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main_v5_2.py .
COPY static/ ./static/
ENV CHECKPOINT_PATH=/weights/best_stage3_f10.6032.pth
ENV QLORA_ADAPTER_PATH=/weights/lora_adapter_20260507_1431
EXPOSE 8000
CMD ["python", "main_v5_2.py"]
```

```bash
docker build -t pneumoai:v5.2 .
docker run --gpus all -v /your/weights:/weights -p 8000:8000 pneumoai:v5.2
```

### Memory Budget

| Component | VRAM |
|-----------|------|
| QLoRA (4-bit, 7B) | ~6 GB |
| CNN DualBranch | ~500 MB |
| **Total** | **~6.5 GB** (16 GB GPU comfortable) |

> Set `device_map="auto"` to enable CPU offloading on GPUs with less than 12 GB VRAM (increases latency).

---

## 🛠 Troubleshooting

### `RuntimeError: CUDA out of memory`

```bash
# Reduce generation budget
QLORA_MAX_SEQ_LEN=512
QLORA_MAX_NEW_TOKENS=400
```

Or add before each QLoRA call:

```python
torch.cuda.empty_cache()
```

### QLoRA returns `parse_ok: False`

Inspect the raw output files generated in the working directory:

```
qlora_raw_output_cycle{N}.txt
```

If Qwen uses a different step format, update the regex in `extract_steps_v2()`.

### `AudioFileError` on MP3

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Google Colab
!apt-get install -q ffmpeg
```

### Checkpoint not found warning

```
WARNING - Checkpoint not found — using random weights
```

Verify the path or use the glob fallback in `_resolve_checkpoint()`:

```python
CHECKPOINT_PATH = "./weights/best_stage3_*.pth"
```

### ngrok tunnel fails

```python
from pyngrok import ngrok
ngrok.kill()  # kill existing tunnels (free accounts: 1 concurrent)
```

---

## 📅 Changelog

### `v5.2.0` — 2026-05-07

| Type | Change |
|------|--------|
| `NEW` | `call_local_llm_sequential_cycles` — 3 independent sequential QLoRA calls replacing single combined call |
| `NEW` | `build_input_text_single_cycle` — per-cycle input text generation |
| `NEW` | `_call_qlora_single` — single QLoRA call wrapper with output logging |
| `NEW` | `aggregate_cycle_qlora_results` — majority vote over 3 independent results |
| `UI`  | `qlora_per_cycle` in API response — array of individual cycle results |
| `UI`  | Per-cycle tab navigation in QLoRA panel — 6-step accordion per cycle |

### `v5.1.0`

| Type | Change |
|------|--------|
| `FIX` | Device placement bug — tensors moved to correct device before LLM call |
| `FIX` | Removed SYSTEM_PROMPT — instruction embedded in Alpaca `### Instruction:` block |
| `NEW` | `parse_qlora_output_v2` — improved `dis_correct` extraction logic |

### `v5.0.0`

| Type | Change |
|------|--------|
| `NEW` | Initial dual-branch architecture with CrossAttentionFusion |
| `NEW` | First QLoRA integration (single combined call) |
| `NEW` | Three.js 3D lung visualization in frontend |

---

## 📖 Citation

```bibtex
@software{pneumoai2026,
  author  = {YOUR NAME},
  title   = {PneumoAI: Dual-Branch Multi-Task CNN with QLoRA Sequential
             Per-Cycle Analysis for Lung Sound Diagnostics},
  version = {5.2.0},
  year    = {2026},
  url     = {https://github.com/YOUR_USERNAME/pneumoai},
  note    = {DualBranch ResNet18 + FPN + CrossAttentionFusion +
             PatientAttention, QLoRA fine-tuned Qwen2.5-7B-Instruct}
}
```

**Builds upon:**

- Rocha et al. (2019). *A Respiratory Sound Database for Automated Classification Systems.* ICBHI 2017.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
- Qwen Team. (2024). *Qwen2.5 Technical Report.* Alibaba Group.
- Lin et al. (2017). *Feature Pyramid Networks for Object Detection.* CVPR.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full terms.

---

> **⚕ Medical Disclaimer** — PneumoAI is a research and clinical decision-support tool. It is **not** a certified medical device and must **not** be used as the sole basis for clinical diagnosis or treatment. All outputs require review by a qualified healthcare professional.

---

<div align="center">

*Built for respiratory health · PneumoAI v5.2.0*

</div>
