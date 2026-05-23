# ─────────────────────────── STDLIB ──────────────────────────────────────────
import os, io, uuid, json, time, logging, warnings, traceback, threading, sys, re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────── LOGGING ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pneumoai")

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NGROK_AUTH_TOKEN   = os.getenv("NGROK_AUTH_TOKEN", "..........")
PORT               = int(os.getenv("PORT", 8000))

CHECKPOINT_PATH    = os.getenv(
    "CHECKPOINT_PATH",
    "........"
)
QLORA_ADAPTER_PATH = os.getenv(
    "QLORA_ADAPTER_PATH",
    "........"
)

# ─────────────────────────── Audio ───────────────────────────────────────────
TARGET_SR         = 16_000
TARGET_LENGTH_SEC = 6
N_MELS            = 128
HOP_LENGTH        = 512
N_FFT             = 1_024
FMIN, FMAX        = 50, 4_000
EPS               = 1e-6

# ─────────────────────────── QLoRA ───────────────────────────────────────────
QLORA_MAX_SEQ_LEN    = 1024
# Each call analyzes only 1 cycle -> shorter input -> can increase new tokens
QLORA_MAX_NEW_TOKENS = 800

# ─────────────────────────── Labels ──────────────────────────────────────────
NUM_EVENT   = 4;  EVENT_NAMES   = ["Normal", "Crackle", "Wheeze", "Both"]
NUM_DISEASE = 3;  DISEASE_NAMES = ["Healthy", "Obstructive", "Infectious"]

DISEASE_VI = {
    "Healthy":     ("Normal",                    "No pathology detected"),
    "Infectious":  ("Respiratory Infection",      "Respiratory Infection"),
    "Obstructive": ("Obstructive Lung Disease",   "Obstructive Lung Disease"),
}
EVENT_CHIP = {
    "Normal":  ("normal",  "No abnormal sounds"),
    "Crackle": ("crackle", "Crackle sound"),
    "Wheeze":  ("wheeze",  "Wheeze sound"),
    "Both":    ("rhonchi", "Both Crackle & Wheeze"),
}
SEVERITY_MAP = {
    "Healthy":     "low",
    "Infectious":  "medium",
    "Obstructive": "high",
}

# ─────────────────────────── Top-K cycles ────────────────────────────────────
TOP_CYCLES_FOR_DISEASE = 3

# ─────────────────────────── Soft retrieval ──────────────────────────────────
TOP_K                   = 5
RETRIEVAL_GAP_AMBIGUOUS = 0.05

OUT_DIR = Path("./static/gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
#  ALPACA INSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

ALPACA_INSTRUCTION = (
    "You are a clinical AI assistant analyzing lung sound recordings. "
    "A dual-branch deep learning model has processed the audio segment and produced: "
    "(1) event predictions at segment-level (Normal/Crackle/Wheeze/Both), "
    "(2) disease predictions at patient-level aggregated across all segments "
    "(Healthy/Infectious/Obstructive). "
    "You are given raw numerical evidence only — no pre-computed interpretations. "
    "Important: event and disease are independent tasks. "
    "A Normal acoustic event does NOT imply a Healthy disease label. "
    "The following reasoning steps may be helpful if applicable: "
    "Step 1: Assess event branch reliability from entropy and margin values. "
    "Step 2: Assess disease branch reliability from entropy, margin, Grad-CAM "
    "activation patterns, and the contrast map between predicted and alternative classes. "
    "Step 3: Evaluate the soft retrieval signal — compare the avg_sim per disease class "
    "and the sim_gap_top2. A small gap means the retrieval is ambiguous and should be "
    "weighted less. Do not simply copy the highest-count class as the answer. "
    "Step 4: Identify which disease class has the highest prototype cosine similarity. "
    "Step 5: Identify all conflicts between model prediction, retrieval signal, "
    "and prototype signal. "
    "Step 6: Resolve conflicts and state the most likely true disease label."
)

# ═════════════════════════════════════════════════════════════════════════════
#  NGROK
# ═════════════════════════════════════════════════════════════════════════════

_ngrok_public_url: str = ""


def start_ngrok(port: int) -> None:
    global _ngrok_public_url
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        tunnel = ngrok.connect(addr=port, proto="http")
        url    = tunnel.public_url
        if url.startswith("http://"):
            url = "https://" + url[7:]
        _ngrok_public_url = url
        sep = "═" * 60
        print(sep)
        print("🌐 NGROK URL:")
        print(f"    {_ngrok_public_url}")
        print(sep)
    except Exception as exc:
        log.warning(f"ngrok error: {exc}")


def _delayed_ngrok(port: int, delay: float = 1.8) -> None:
    time.sleep(delay)
    start_ngrok(port)

# ═════════════════════════════════════════════════════════════════════════════
#  THIRD-PARTY IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

import glob as _glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import librosa
from scipy.signal import butter, filtfilt

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device        : {DEVICE}")
log.info(f"Checkpoint    : {CHECKPOINT_PATH}")
log.info(f"QLoRA path    : {QLORA_ADAPTER_PATH}")

# ═════════════════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _make_butter():
    nyq  = 0.5 * TARGET_SR
    b, a = butter(5, [100 / nyq, 2_000 / nyq], btype="band")
    return b, a

_B, _A = _make_butter()


def audio_to_mel(segment: np.ndarray) -> np.ndarray:
    target_len = TARGET_SR * TARGET_LENGTH_SEC
    if len(segment) < target_len:
        reps    = (target_len // len(segment)) + 1
        segment = np.tile(segment, reps)[:target_len]
    else:
        segment = segment[:target_len]
    mel = librosa.feature.melspectrogram(
        y=segment, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    log_mel = np.log(mel + EPS)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


def load_wav_bytes(wav_bytes: bytes):
    y, _   = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
    y_filt = filtfilt(_B, _A, y).astype(np.float32)
    y_filt = (y_filt - y_filt.mean()) / (y_filt.std() + 1e-8)

    win, hop = TARGET_LENGTH_SEC * TARGET_SR, 3 * TARGET_SR
    cycles, start = [], 0

    while start + win <= len(y_filt):
        mel = audio_to_mel(y_filt[start: start + win])
        cycles.append({
            "start_sec":  round(start / TARGET_SR, 2),
            "end_sec":    round((start + win) / TARGET_SR, 2),
            "mel_tensor": torch.tensor(mel).unsqueeze(0).unsqueeze(0),
        })
        start += hop

    if not cycles:
        mel = audio_to_mel(y_filt)
        cycles.append({
            "start_sec":  0.0,
            "end_sec":    round(len(y_filt) / TARGET_SR, 2),
            "mel_tensor": torch.tensor(mel).unsqueeze(0).unsqueeze(0),
        })

    return cycles, round(len(y) / TARGET_SR, 2), y_filt

# ═════════════════════════════════════════════════════════════════════════════
#  DUAL-BRANCH MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class _BranchFPN(nn.Module):
    def __init__(self, in_ch=(64, 128, 256, 512), out_ch=256):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_ch])
        self.pool     = nn.AdaptiveAvgPool2d(1)

    def forward(self, feats):
        out = None
        for i, f in enumerate(feats):
            lat = self.laterals[i](f)
            if out is not None:
                lat = lat + F.interpolate(out, size=lat.shape[-2:], mode="nearest")
            out = lat
        return self.pool(out).flatten(1)


class SharedStem(nn.Module):
    def __init__(self):
        super().__init__()
        base       = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        old_conv   = base.conv1
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2

    def forward(self, x):
        x  = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        return c2, c3


class BranchUpper(nn.Module):
    def __init__(self):
        super().__init__()
        base        = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.fpn    = _BranchFPN(in_ch=(64, 128, 256, 512), out_ch=256)

    def forward(self, c2, c3):
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.ca_e2d = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.ca_d2e = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm_e = nn.LayerNorm(dim)
        self.norm_d = nn.LayerNorm(dim)
        self.gate_e = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate_d = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, emb_e, emb_d):
        e = emb_e.unsqueeze(1); d = emb_d.unsqueeze(1)
        e_ctx, _ = self.ca_d2e(query=e, key=d, value=d)
        g_e      = self.gate_e(torch.cat([emb_e, e_ctx.squeeze(1)], dim=-1))
        emb_e_   = self.norm_e(emb_e + g_e * e_ctx.squeeze(1))
        d_ctx, _ = self.ca_e2d(query=d, key=e, value=e)
        g_d      = self.gate_d(torch.cat([emb_d, d_ctx.squeeze(1)], dim=-1))
        emb_d_   = self.norm_d(emb_d + g_d * d_ctx.squeeze(1))
        return emb_e_, emb_d_


class PatientAttention(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=0)
        return (x * w).sum(dim=0, keepdim=True)


def _make_head(in_dim: int, hidden: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.4),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(hidden // 2, n_classes),
    )


class DualBranchModel(nn.Module):
    def __init__(self, use_cross: bool = True):
        super().__init__()
        self.shared        = SharedStem()
        self.event_upper   = BranchUpper()
        self.disease_upper = BranchUpper()
        self.cross_attn    = CrossAttentionFusion(dim=256, num_heads=4)
        self.patient_attn  = PatientAttention(256)
        self.event_head    = _make_head(256, 256, NUM_EVENT)
        self.disease_head  = _make_head(256, 256, NUM_DISEASE)
        self._use_cross    = use_cross

    def get_embeddings(self, x):
        c2, c3  = self.shared(x)
        emb_e   = self.event_upper(c2, c3)
        emb_d   = self.disease_upper(c2, c3)
        if self._use_cross:
            emb_e, emb_d = self.cross_attn(emb_e, emb_d)
        return emb_e, emb_d

    def forward(self, x):
        emb_e, emb_d = self.get_embeddings(x)
        return emb_e, emb_d, self.event_head(emb_e)

    def patient_disease(self, emb_d_segments: torch.Tensor) -> torch.Tensor:
        return self.disease_head(self.patient_attn(emb_d_segments))

# ═════════════════════════════════════════════════════════════════════════════
#  DUAL-BRANCH GRAD-CAM
# ═════════════════════════════════════════════════════════════════════════════

class DualBranchGradCAM:
    def __init__(self, model: DualBranchModel, task: str = "event"):
        self.model = model
        self.task  = task
        self.acts  = None
        self.grads = None
        target   = (model.event_upper.layer4[-1] if task == "event"
                    else model.disease_upper.layer4[-1])
        self._fh = target.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o.detach()))
        self._bh = target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def compute(self, inp: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        inp_g = inp.clone().requires_grad_(True).to(DEVICE)
        emb_e, emb_d, ev_logits = self.model(inp_g)
        if self.task == "event":
            score = ev_logits[0, target_class]
        else:
            score = self.model.patient_disease(emb_d)[0, target_class]
        score.backward()
        w      = self.grads.mean(dim=(2, 3), keepdim=True)
        cam    = torch.relu((w * self.acts).sum(dim=1, keepdim=True))
        H_, W_ = inp.shape[-2], inp.shape[-1]
        cam_up = F.interpolate(cam.float(), size=(H_, W_), mode="bilinear", align_corners=False)
        c      = cam_up.squeeze().cpu().numpy()
        lo, hi = c.min(), c.max()
        return (c - lo) / (hi - lo + 1e-8)

    def dual_target(self, inp: torch.Tensor,
                    pred_class: int, alt_class: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        cam_pred = self.compute(inp, pred_class)
        if pred_class == alt_class:
            cam_alt = cam_pred
        else:
            cam_alt = self.compute(inp, alt_class)
        return cam_pred, cam_alt


def extract_cam_raw(cam_np: np.ndarray) -> Dict:
    H, W   = cam_np.shape
    h3, w3 = H // 3, W // 3
    flat   = cam_np.flatten()
    fn     = flat / (flat.sum() + 1e-8)
    ent    = float(-np.sum(fn * np.log(fn + 1e-8)))
    return {
        "freq_high" : round(float(cam_np[:h3].mean()),        4),
        "freq_mid"  : round(float(cam_np[h3:2*h3].mean()),    4),
        "freq_low"  : round(float(cam_np[2*h3:].mean()),      4),
        "time_early": round(float(cam_np[:, :w3].mean()),     4),
        "time_mid"  : round(float(cam_np[:, w3:2*w3].mean()), 4),
        "time_late" : round(float(cam_np[:, 2*w3:].mean()),   4),
        "peak"      : round(float(cam_np.max()),               4),
        "std"       : round(float(cam_np.std()),               4),
        "entropy"   : round(ent,                               4),
        "hot_ratio" : round(float((cam_np > 0.6).mean()),     4),
    }


def extract_cam_diff(cam_pred: np.ndarray, cam_alt: np.ndarray) -> Dict:
    diff     = cam_pred - cam_alt
    abs_diff = np.abs(diff)
    flat_abs = abs_diff.flatten()
    fn       = flat_abs / (flat_abs.sum() + 1e-8)
    ent      = float(-np.sum(fn * np.log(fn + 1e-8)))
    return {
        "diff_peak"    : round(float(diff.max()),     4),
        "diff_min"     : round(float(diff.min()),      4),
        "diff_abs_mean": round(float(abs_diff.mean()), 4),
        "diff_entropy" : round(ent,                    4),
    }


def save_cam_image(mel_np, cam_event_pred, cam_event_alt,
                   cam_dis_pred, cam_dis_alt, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    fig.patch.set_facecolor("#0f0f14")
    for ax in axes:
        ax.set_facecolor("#0f0f14")

    data = [
        (mel_np,         "magma", "Log-Mel"),
        (cam_event_pred, "jet",   "CAM Event (pred)"),
        (cam_dis_pred,   "jet",   "CAM Disease (pred)"),
        (cam_dis_alt,    "hot",   "CAM Disease (alt)"),
        (cam_event_pred, "gray",  "Overlay"),
    ]

    for i, (ax, (d, cmap, ttl)) in enumerate(zip(axes, data)):
        if i == 4:
            ax.imshow(mel_np,         origin="lower", aspect="auto", cmap="gray", alpha=0.5)
            ax.imshow(cam_event_pred, origin="lower", aspect="auto", cmap="jet",  alpha=0.5)
        else:
            ax.imshow(d, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(ttl, color="white", fontsize=7)
        ax.axis("off")

    fig.suptitle(title, color="white", fontsize=8, fontweight="bold")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
#  TOP-3 CYCLES SELECTOR
# ═════════════════════════════════════════════════════════════════════════════

def select_top_cycles(cycle_results: List[dict],
                      n: int = TOP_CYCLES_FOR_DISEASE) -> List[dict]:
    if not cycle_results:
        return []

    abnormal = [c for c in cycle_results if c["event"] != "Normal"]
    normal   = [c for c in cycle_results if c["event"] == "Normal"]

    abnormal_sorted = sorted(
        abnormal,
        key=lambda x: x.get("cam_disease", {}).get("peak", 0),
        reverse=True
    )
    normal_sorted = sorted(
        normal,
        key=lambda x: x.get("cam_disease", {}).get("peak", 0),
        reverse=True
    )

    selected = abnormal_sorted[:n]
    if len(selected) < n:
        selected += normal_sorted[:n - len(selected)]

    selected = sorted(selected, key=lambda x: x["start_sec"])

    rank_map = {}
    for rank_i, cyc in enumerate(
        sorted(selected, key=lambda x: x.get("cam_disease", {}).get("peak", 0), reverse=True),
        start=1
    ):
        rank_map[cyc["cycle_index"]] = rank_i

    result = []
    for cyc in selected:
        c = dict(cyc)
        c["is_top_cycle"]   = True
        c["top_cycle_rank"] = rank_map.get(cyc["cycle_index"], 99)
        result.append(c)

    return result

# ═════════════════════════════════════════════════════════════════════════════
#  UNCERTAINTY
# ═════════════════════════════════════════════════════════════════════════════

def compute_uncertainty(logits_tensor: torch.Tensor, class_names: List[str]) -> Dict:
    p        = torch.softmax(logits_tensor.float(), dim=0).cpu().numpy()
    sorted_p = np.sort(p)[::-1]
    return {
        "probs"  : {n: round(float(p[i]), 4) for i, n in enumerate(class_names)},
        "entropy": round(float(-np.sum(p * np.log(p + 1e-8))), 4),
        "margin" : round(float(sorted_p[0] - sorted_p[1]) if len(sorted_p) >= 2 else 1.0, 4),
    }

# ═════════════════════════════════════════════════════════════════════════════
#  SOFT RETRIEVAL STATS
# ═════════════════════════════════════════════════════════════════════════════

def soft_retrieval_stats(topk: List[Dict]) -> Dict:
    if not topk:
        return {
            "avg_sim"     : {n: 0.0 for n in DISEASE_NAMES},
            "count"       : {n: 0   for n in DISEASE_NAMES},
            "top_class"   : DISEASE_NAMES[0],
            "sim_gap_top2": 0.0,
            "is_ambiguous": True,
        }

    class_sims: Dict[str, List[float]] = {n: [] for n in DISEASE_NAMES}
    for r in topk:
        cls_name = (DISEASE_NAMES[r["pred_dis_lab"]]
                    if isinstance(r.get("pred_dis_lab"), int)
                    else r.get("pred_disease", DISEASE_NAMES[0]))
        if cls_name in class_sims:
            class_sims[cls_name].append(r.get("sim", r.get("similarity", 0.0)))

    avg_sim = {
        n: round(float(np.mean(v)), 4) if v else 0.0
        for n, v in class_sims.items()
    }
    count    = {n: len(v) for n, v in class_sims.items()}
    ranked   = sorted(avg_sim.items(), key=lambda x: x[1], reverse=True)
    top_class   = ranked[0][0]
    top_sim     = ranked[0][1]
    second_sim  = ranked[1][1] if len(ranked) > 1 else top_sim
    gap         = round(top_sim - second_sim, 4)

    return {
        "avg_sim"     : avg_sim,
        "count"       : count,
        "top_class"   : top_class,
        "sim_gap_top2": gap,
        "is_ambiguous": gap < RETRIEVAL_GAP_AMBIGUOUS,
    }


def build_pseudo_topk_v2(pred_disease: str, pred_event: str,
                         dis_probs: List[float], n: int = TOP_K) -> List[dict]:
    cases        = []
    pred_dis_idx = DISEASE_NAMES.index(pred_disease) if pred_disease in DISEASE_NAMES else 0
    base_prob    = dis_probs[pred_dis_idx]
    alt_indices  = [i for i in range(NUM_DISEASE) if i != pred_dis_idx]

    for i in range(n):
        if i < max(1, round(n * base_prob)):
            dis_idx = pred_dis_idx
            sim     = round(max(0.0, 0.92 - i * 0.05), 4)
        else:
            alt_i   = alt_indices[i % len(alt_indices)]
            dis_idx = alt_i
            sim     = round(max(0.0, 0.62 - i * 0.04), 4)

        ev_idx = EVENT_NAMES.index(pred_event) if pred_event in EVENT_NAMES else 0
        cases.append({
            "sim"         : sim,
            "similarity"  : sim,
            "pred_dis_lab": dis_idx,
            "pred_disease": DISEASE_NAMES[dis_idx],
            "ev_lab"      : ev_idx,
            "event"       : EVENT_NAMES[ev_idx],
            "ev_conf"     : round(base_prob, 4),
            "dis_conf"    : round(dis_probs[dis_idx], 4),
            "pid"         : f"pseudo_{i:03d}",
        })

    return cases


def build_prototype_scores(dis_probs: List[float]) -> List[float]:
    total = sum(dis_probs) + 1e-8
    return [round(p / total, 4) for p in dis_probs]

# ═════════════════════════════════════════════════════════════════════════════
#  BUILD INPUT TEXT — FOR A SINGLE CYCLE  [SEQ-2]
#
#  Input contains only data for that cycle: cam_event, cam_disease,
#  cam_disease_alt, cam_diff of that cycle; with patient-level
#  uncertainty + retrieval + prototype for sufficient context.
# ═════════════════════════════════════════════════════════════════════════════

def build_input_text_single_cycle(
    cycle_index     : int,
    cycle_rank      : int,
    start_sec       : float,
    end_sec         : float,
    pred_event      : str,
    event_confidence: float,
    pred_disease    : str,
    alt_disease     : str,
    n_segments      : int,
    ev_unc          : Dict,
    dis_unc         : Dict,
    cam_ev_pred_raw : Dict,
    cam_dis_pred_raw: Dict,
    cam_dis_alt_raw : Dict,
    cam_diff_raw    : Dict,
    topk            : List[Dict],
    proto_sim       : List[float],
) -> str:

    ev_prob_str  = " | ".join(f"{k}={v:.4f}" for k, v in ev_unc["probs"].items())
    dis_prob_str = " | ".join(f"{k}={v:.4f}" for k, v in dis_unc["probs"].items())
    proto_str    = " | ".join(
        f"{DISEASE_NAMES[i]}={proto_sim[i]:.4f}" for i in range(NUM_DISEASE)
    )

    ret_stats   = soft_retrieval_stats(topk)
    avg_sim_str = " | ".join(
        f"{n}={ret_stats['avg_sim'][n]:.4f}(n={ret_stats['count'][n]})"
        for n in DISEASE_NAMES
    )

    ret_rows = "\n".join(
        f"  Rank{i+1}: sim={r.get('sim', 0):.4f} | "
        f"event={EVENT_NAMES[r['ev_lab']] if isinstance(r.get('ev_lab'), int) else '?'} | "
        f"pred_disease={DISEASE_NAMES[r['pred_dis_lab']] if isinstance(r.get('pred_dis_lab'), int) else '?'} | "
        f"ev_conf={r.get('ev_conf', 0):.3f} | dis_conf={r.get('dis_conf', 0):.3f}"
        for i, r in enumerate(topk)
    )

    def _cam_line(c: Dict) -> str:
        return (
            f"  freq_high={c.get('freq_high',0):.4f} | freq_mid={c.get('freq_mid',0):.4f} "
            f"| freq_low={c.get('freq_low',0):.4f}\n"
            f"  time_early={c.get('time_early',0):.4f} | time_mid={c.get('time_mid',0):.4f} "
            f"| time_late={c.get('time_late',0):.4f}\n"
            f"  peak={c.get('peak',0):.4f} | std={c.get('std',0):.4f} "
            f"| entropy={c.get('entropy',0):.4f} | hot_ratio={c.get('hot_ratio',0):.4f}"
        )

    return f"""=== CYCLE INFORMATION ===
  Cycle index : {cycle_index}
  Time range  : {start_sec:.1f}s — {end_sec:.1f}s
  Rank (disease branch peak CAM priority) : #{cycle_rank}
  Total cycles in recording : {n_segments}

=== MODEL PREDICTIONS (THIS CYCLE) ===
  Event   (segment-level, this cycle) : {pred_event}  [conf={event_confidence:.4f}]
  Disease (patient-level, CNN)         : {pred_disease}
  Alt disease (2nd highest prob)       : {alt_disease}

=== EVENT-DISEASE INDEPENDENCE ===
  Event and disease are independent tasks. A Normal acoustic event does NOT
  imply a Healthy disease label. Disease is determined by the disease branch
  embeddings, retrieval similarity scores, and prototype cosine scores.

=== EVENT BRANCH — UNCERTAINTY (patient-level) ===
  Probabilities : {ev_prob_str}
  Entropy       : {ev_unc['entropy']:.4f}
  Margin        : {ev_unc['margin']:.4f}

=== DISEASE BRANCH — UNCERTAINTY (patient-level) ===
  Probabilities : {dis_prob_str}
  Entropy       : {dis_unc['entropy']:.4f}
  Margin        : {dis_unc['margin']:.4f}

=== GRAD-CAM — EVENT BRANCH (this cycle, target: predicted event class) ===
{_cam_line(cam_ev_pred_raw)}

=== GRAD-CAM — DISEASE BRANCH (this cycle, target: predicted disease class) ===
{_cam_line(cam_dis_pred_raw)}

=== GRAD-CAM — DISEASE BRANCH (this cycle, target: alternative disease class) ===
{_cam_line(cam_dis_alt_raw)}

=== GRAD-CAM — DISEASE BRANCH CONTRAST (predicted minus alternative, this cycle) ===
  diff_peak={cam_diff_raw.get('diff_peak',0):.4f} | diff_min={cam_diff_raw.get('diff_min',0):.4f}
  diff_abs_mean={cam_diff_raw.get('diff_abs_mean',0):.4f} | diff_entropy={cam_diff_raw.get('diff_entropy',0):.4f}

=== TOP-{TOP_K} RETRIEVAL — RAW RANKED CASES (patient-level) ===
  Note: pred_disease = model's prediction for that case, NOT ground truth.
{ret_rows}

=== TOP-{TOP_K} RETRIEVAL — SOFT SIMILARITY SIGNAL ===
  avg_sim per class : {avg_sim_str}
  sim_gap_top2      : {ret_stats['sim_gap_top2']:.4f}
  retrieval_ambiguous (gap < {RETRIEVAL_GAP_AMBIGUOUS}) : {ret_stats['is_ambiguous']}

=== PROTOTYPE COSINE SIMILARITY ===
  {proto_str}""".strip()

# ═════════════════════════════════════════════════════════════════════════════
#  PARSE QLORA OUTPUT  (unchanged from v5.1)
# ═════════════════════════════════════════════════════════════════════════════

def _detect_section(line: str) -> Optional[str]:
    stripped = re.sub(r'^[=\s]+|[=\s]+$', '', line).strip()
    low      = stripped.lower()
    patterns = [
        (r'ground\s*truth',                   'gt'),
        (r'correctness',                      'correctness'),
        (r'step\s*1.*event.*branch.*assess',  'step1'),
        (r'step\s*1',                         'step1'),
        (r'step\s*2.*disease.*branch.*assess','step2'),
        (r'step\s*2',                         'step2'),
        (r'step\s*3.*retrieval',              'step3'),
        (r'step\s*3',                         'step3'),
        (r'step\s*4.*prototype',              'step4'),
        (r'step\s*4',                         'step4'),
        (r'step\s*5.*conflict',               'step5'),
        (r'step\s*5',                         'step5'),
        (r'step\s*6.*final.*conclusion',      'step6'),
        (r'step\s*6',                         'step6'),
        (r'final\s*conclusion',               'step6'),
        (r'model\s*predictions',              'model_preds'),
        (r'grad.?cam',                        None),
        (r'retrieval',                        None),
        (r'prototype',                        None),
        (r'uncertainty',                      None),
        (r'event.?disease\s*independence',    None),
        (r'cycle\s*information',              None),
    ]
    for pat, key in patterns:
        if re.search(pat, low):
            return key
    return None


import re
from typing import Dict

def extract_steps_v2(text: str) -> Dict[str, str]:
    """
    [FIX-REGEX] Robust step parser that handles ALL LLM output formats:
      - **Step N: Title**   (bold markdown — what Qwen2.5 actually outputs)
      - **Step N — Title**
      - #### Step N: Title  (hash headers)
      - Step N: Title       (plain)
      - Step N — Title      (dash)

    Root cause of original bug: regex only handled #+\s*Step prefix,
    missed **Step** bold format → all steps empty → nothing shown in UI.
    """
    result = {
        "step1_event_assessment": "",
        "step2_disease_assessment": "",
        "step3_retrieval_signal": "",
        "step4_prototype_signal": "",
        "step5_conflict_analysis": "",
        "step6_final_conclusion": "",
    }

    if not text:
        return result

    # Strip content after Human/Instruction turn boundary
    text = re.split(r'###\s*Human:|Human:|###\s*Instruction:', text, flags=re.I)[0]

    # ── [FIX] Universal regex: handles **, ##, plain, dash/colon separator ──
    # Captures: (step_number, title_text, body_text)
    pattern = re.compile(
        r'(?:^|\n)\s*'
        r'(?:\*{1,2}|#{1,6}\s*)?'              # optional ** or ### prefix
        r'Step\s*(\d+)'                          # "Step N"
        r'\s*(?:[:\-—]+\s*)?'                   # optional : - — separator
        r'\*{0,2}\s*'                            # optional trailing ** (bold close)
        r'(.*?)'                                 # title text (may be empty)
        r'\*{0,2}\s*\n'                          # close bold/whitespace + newline
        r'(.*?)'                                 # body content
        r'(?='
        r'\n\s*(?:\*{1,2}|#{1,6}\s*)?Step\s*\d+\s*[:\-—]'  # next Step header
        r'|\Z'                                   # or end of string
        r')',
        re.I | re.S
    )

    matches = pattern.findall(text)

    for step_num_str, title, content in matches:
        try:
            step_num = int(step_num_str)
        except ValueError:
            continue

        # Clean trailing ** or whitespace from title
        title_clean = re.sub(r'\*+', '', title).strip()
        body_clean  = content.strip()

        full_content = (f"{title_clean}\n{body_clean}").strip() if title_clean else body_clean

        key_map = {
            1: "step1_event_assessment",
            2: "step2_disease_assessment",
            3: "step3_retrieval_signal",
            4: "step4_prototype_signal",
            5: "step5_conflict_analysis",
            6: "step6_final_conclusion",
        }
        dest_key = key_map.get(step_num)
        if dest_key:
            result[dest_key] = full_content

    return result


def parse_qlora_output_v2(text: str, cnn_pred_disease: str) -> dict:
    """
    Parse QLoRA output into structured dict.
    [FIX-PARSE] More robust dis_correct extraction — checks Step 6 text
    for disease label alignment with CNN prediction and phrases like
    'correct', 'incorrect', 'confirms', 'disagrees', 'true label is'.
    """
    if not text or len(text.strip()) < 10:
        return {
            "parse_ok": False,
            "dis_correct": None,
            "final_disease_label": cnn_pred_disease,
            "gt_disease": None,
            "reasoning": [],
            "step6_final_conclusion": "",
            "step1_event_assessment": "",
            "step2_disease_assessment": "",
            "step3_retrieval_signal": "",
            "step4_prototype_signal": "",
            "step5_conflict_analysis": "",
        }

    steps = extract_steps_v2(text)

    # ── Extract final disease label from Step 6 ─────────────────────────────
    final_label = cnn_pred_disease
    conclusion  = steps.get("step6_final_conclusion", "")

    # Search in Step 6 first, then full text as fallback
    search_text = conclusion if conclusion else text
    for disease in DISEASE_NAMES:
        if disease.lower() in search_text.lower():
            final_label = disease
            break

    # ── [FIX-PARSE] Determine dis_correct more robustly ─────────────────────
    # In inference mode (no ground truth), LLM may still say things like:
    # "model prediction is correct", "confirms the CNN prediction",
    # "the prediction aligns", "true label is X" where X == cnn_pred, etc.
    is_correct = None

    step6_lower = conclusion.lower()
    full_lower  = text.lower()

    # Positive signals
    positive_phrases = [
        "model prediction is correct",
        "prediction is correct",
        "correctly identified",
        "confirms the cnn",
        "confirms cnn",
        "aligns with the model",
        "consistent with the model",
        "supports the model prediction",
        "prediction appears correct",
        "prediction seems correct",
        "diagnosis is confirmed",
        "đúng",
        "chính xác",
    ]
    # Negative signals
    negative_phrases = [
        "model prediction is incorrect",
        "prediction is incorrect",
        "incorrectly identified",
        "true label is",          # "true label is X" where X != cnn_pred
        "actual diagnosis is",
        "disagrees with",
        "conflicts with the model",
        "may be incorrect",
        "might be incorrect",
        "sai lệch",
        "không chính xác",
        "sai",
    ]

    check_text = step6_lower if step6_lower else full_lower

    has_positive = any(p in check_text for p in positive_phrases)
    has_negative = any(p in check_text for p in negative_phrases)

    # Check if final_label matches CNN prediction
    label_matches_cnn = (final_label == cnn_pred_disease)

    if has_positive and not has_negative:
        is_correct = True
    elif has_negative and not has_positive:
        is_correct = False
    elif label_matches_cnn:
        # LLM concluded same disease as CNN → likely correct
        is_correct = True
    elif final_label != cnn_pred_disease and final_label in DISEASE_NAMES:
        # LLM concluded different disease → likely disagreement
        is_correct = False
    # else: leave as None (inference mode, no clear signal)

    # ── Check for ground truth mention (training eval mode only) ────────────
    gt_disease = None
    gt_pattern = re.compile(
        r'true\s+(?:disease\s+)?label\s*[:\-=]\s*(' +
        '|'.join(re.escape(d) for d in DISEASE_NAMES) + r')',
        re.I
    )
    gt_match = gt_pattern.search(text)
    if gt_match:
        gt_disease = gt_match.group(1).capitalize()
        # Standardize capitalization
        for d in DISEASE_NAMES:
            if d.lower() == gt_disease.lower():
                gt_disease = d
                break

    return {
        "parse_ok"               : bool(conclusion or any(steps.values())),
        "dis_correct"            : is_correct,
        "final_disease_label"    : final_label,
        "gt_disease"             : gt_disease,
        "reasoning"              : [v for v in steps.values() if v],
        **steps,
    }

# ═════════════════════════════════════════════════════════════════════════════
#  QLORA MODEL LOADER
# ═════════════════════════════════════════════════════════════════════════════

_qlora_model     = None
_qlora_tokenizer = None


def get_qlora_model():
    global _qlora_model, _qlora_tokenizer
    if _qlora_model is not None:
        return _qlora_model, _qlora_tokenizer
    if not os.path.exists(QLORA_ADAPTER_PATH):
        log.warning(f"QLoRA adapter not found: {QLORA_ADAPTER_PATH}")
        return None, None
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        log.info(f"Loading QLoRA: {QLORA_ADAPTER_PATH}")
        _qlora_model = AutoPeftModelForCausalLM.from_pretrained(
            QLORA_ADAPTER_PATH,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        _qlora_tokenizer = AutoTokenizer.from_pretrained(
            QLORA_ADAPTER_PATH, trust_remote_code=True
        )
        _qlora_tokenizer.pad_token    = _qlora_tokenizer.eos_token
        _qlora_tokenizer.padding_side = "left"
        log.info("QLoRA loaded successfully ✓")
        return _qlora_model, _qlora_tokenizer
    except Exception as exc:
        log.warning(f"Failed to load QLoRA: {exc}\n{traceback.format_exc()}")
        return None, None




# ═════════════════════════════════════════════════════════════════════════════
#  _CALL_QLORA_SINGLE — call QLoRA once for 1 cycle  [SEQ-3]
# ═════════════════════════════════════════════════════════════════════════════

def _call_qlora_single(input_text: str, pred_disease: str) -> Optional[dict]:
    """
    Call QLoRA once for 1 cycle — WITH CLEAR OUTPUT PRINTING
    """
    model, tok = get_qlora_model()
    if model is None:
        log.warning("QLoRA model not loaded")
        return None

    try:
        prompt = (
            f"### Instruction:\n{ALPACA_INSTRUCTION}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )

        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=QLORA_MAX_SEQ_LEN,
            padding=False,
        )

        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # === PRINT INFO BEFORE CALLING ===
        log.info(f"🚀 Calling QLoRA for 1 cycle... (Input tokens: {prompt_len})")

        with torch.no_grad():
            out = model.generate(
                input_ids          = inputs["input_ids"],
                attention_mask     = inputs.get("attention_mask"),
                max_new_tokens     = QLORA_MAX_NEW_TOKENS,
                do_sample          = False,
                repetition_penalty = 1.1,
                pad_token_id       = tok.eos_token_id,
                eos_token_id       = tok.eos_token_id,
            )

        generated_ids = out[0][prompt_len:]
        raw_output = tok.decode(generated_ids, skip_special_tokens=True).strip()

        # ====================== PRINT LLM OUTPUT ======================
        print("\n" + "═"*120)
        print("🔥 QLoRA RAW OUTPUT (Raw text returned from LLM)")
        print("═"*120)
        print(raw_output)
        print("═"*120 + "\n")

        # Save file for later review
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qlora_raw_output_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"QLoRA RAW OUTPUT - {timestamp}\n")
            f.write("="*80 + "\n\n")
            f.write(raw_output)

        log.info(f"💾 Raw output saved to: {filename}")

        # Parse
        parsed = parse_qlora_output_v2(raw_output, pred_disease)

        # Print parsed result
        print("📊 PARSED RESULT:")
        print(json.dumps({
            "dis_correct": parsed.get("dis_correct"),
            "final_disease_label": parsed.get("final_disease_label"),
            "gt_disease": parsed.get("gt_disease"),
            "parse_ok": parsed.get("parse_ok"),
            "step6_final_conclusion": parsed.get("step6_final_conclusion", "")[:300] + "..."
        }, ensure_ascii=False, indent=2))

        return parsed

    except Exception as exc:
        log.error(f"_call_qlora_single exception: {exc}")
        traceback.print_exc()
        return None

# ═════════════════════════════════════════════════════════════════════════════
#  CALL_LOCAL_LLM_SEQUENTIAL_CYCLES  [SEQ-1]
#
#  Main flow:
#    top3_cycles already have full cam data
#    -> Cycle rank#1 -> _call_qlora_single -> wait -> parsed_1
#    -> Cycle rank#2 -> _call_qlora_single -> wait -> parsed_2
#    -> Cycle rank#3 -> _call_qlora_single -> wait -> parsed_3
#    -> return [parsed_1, parsed_2, parsed_3]
# ═════════════════════════════════════════════════════════════════════════════

def call_local_llm_sequential_cycles(
    top3_cycles    : List[dict],
    pred_disease   : str,
    dominant_event : str,
    n_segments     : int,
    ev_unc         : Dict,
    dis_unc        : Dict,
    dis_probs      : List[float],
    alt_disease    : str,
) -> List[Optional[dict]]:
    """
    Call QLoRA sequentially and independently for each cycle in top3_cycles.
    Returns a list of 3 elements (None if QLoRA fails for that cycle).
    """
    # Pseudo retrieval + prototype shared across all 3 calls (patient-level)
    pseudo_topk = build_pseudo_topk_v2(pred_disease, dominant_event, dis_probs, n=TOP_K)
    proto_sim   = build_prototype_scores(dis_probs)

    results: List[Optional[dict]] = []

    for i, cyc in enumerate(top3_cycles):
        rank      = cyc.get("top_cycle_rank", i + 1)
        cycle_idx = cyc.get("cycle_index", i + 1)
        start_sec = cyc.get("start_sec", 0.0)
        end_sec   = cyc.get("end_sec", 0.0)
        cyc_event = cyc.get("event", dominant_event)
        cyc_ev_conf = cyc.get("event_confidence", 0.0)

        log.info(
            f"[SEQ] QLoRA call {i+1}/{len(top3_cycles)} — "
            f"Cycle #{cycle_idx} rank={rank} "
            f"[{start_sec}s–{end_sec}s] event={cyc_event}"
        )

        # Build input text specific to this cycle
        input_text = build_input_text_single_cycle(
            cycle_index      = cycle_idx,
            cycle_rank       = rank,
            start_sec        = start_sec,
            end_sec          = end_sec,
            pred_event       = cyc_event,
            event_confidence = cyc_ev_conf,
            pred_disease     = pred_disease,
            alt_disease      = alt_disease,
            n_segments       = n_segments,
            ev_unc           = ev_unc,
            dis_unc          = dis_unc,
            cam_ev_pred_raw  = cyc.get("cam_event", {}),
            cam_dis_pred_raw = cyc.get("cam_disease", {}),
            cam_dis_alt_raw  = cyc.get("cam_disease_alt", {}),
            cam_diff_raw     = cyc.get("cam_diff", {}),
            topk             = pseudo_topk,
            proto_sim        = proto_sim,
        )

        # Call QLoRA, wait for result before moving to next cycle
        parsed = _call_qlora_single(input_text, pred_disease)

        if parsed is not None:
            # Attach cycle metadata for UI tracing
            parsed["_cycle_index"] = cycle_idx
            parsed["_cycle_rank"]  = rank
            parsed["_cycle_event"] = cyc_event
            parsed["_start_sec"]   = start_sec
            parsed["_end_sec"]     = end_sec
            log.info(
                f"[SEQ] Cycle #{cycle_idx} done — "
                f"dis_correct={parsed.get('dis_correct')} "
                f"parse_ok={parsed.get('parse_ok')} "
                f"final_label={parsed.get('final_disease_label')}"
            )
        else:
            log.warning(f"[SEQ] Cycle #{cycle_idx} — QLoRA returned None, continuing to next cycle")

        results.append(parsed)

    return results

# ═════════════════════════════════════════════════════════════════════════════
#  AGGREGATE QLORA CYCLE RESULTS -> CLINICAL SUMMARY  [SEQ-4]
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_cycle_qlora_results(
    cycle_results_qlora: List[Optional[dict]],
    pred_disease       : str,
    dis_conf           : int,
) -> dict:
    """
    Aggregate QLoRA results from 3 independent cycles into 1 clinical summary.
    CNN disease is ALWAYS preserved — QLoRA does not override.
    Majority vote for dis_correct and final_disease_label.
    """
    dis_vi, dis_en = DISEASE_VI.get(pred_disease, (pred_disease, pred_disease))
    severity       = SEVERITY_MAP.get(pred_disease, "medium")

    valid = [r for r in cycle_results_qlora if r is not None]

    if not valid:
        return _fallback_clinical(pred_disease, dis_conf)

    # Majority vote dis_correct
    correct_votes = [r.get("dis_correct") for r in valid if r.get("dis_correct") is not None]
    if correct_votes:
        n_corr = sum(1 for v in correct_votes if v is True)
        n_incr = sum(1 for v in correct_votes if v is False)
        agg_dis_correct = True if n_corr > n_incr else (False if n_incr > n_corr else None)
    else:
        agg_dis_correct = None

    # Majority vote final_disease_label
    final_labels = [
        r.get("final_disease_label", pred_disease)
        for r in valid
        if r.get("final_disease_label") in DISEASE_NAMES
    ]
    cnt = Counter(final_labels)
    agg_final_label = cnt.most_common(1)[0][0] if cnt else pred_disease

    # GT disease (take first available)
    gt_candidates = [r.get("gt_disease") for r in valid
                     if r.get("gt_disease") in DISEASE_NAMES]
    agg_gt_disease = gt_candidates[0] if gt_candidates else None

    # Clinical note — combine 1 sentence from step6 of each cycle
    note_parts = []
    for r in valid:
        s6 = r.get("step6_final_conclusion", "")
        if s6:
            first = [s.strip() for s in re.split(r'\.\s+', s6) if s.strip()]
            if first:
                note_parts.append(first[0].rstrip("."))

    if note_parts:
        clinical_note = ". ".join(note_parts) + "."
    else:
        clinical_note = f"Lung sound analysis suggests: {dis_vi}."

    if agg_dis_correct is False:
        clinical_note = "⚠ QLoRA detected a possible discrepancy in the prediction. " + clinical_note

    # Recommendations
    all_reasoning = []
    for r in valid:
        all_reasoning.extend(r.get("reasoning", []))
    recommendations = [s for s in all_reasoning if s and len(s) > 10][:4]
    if not recommendations:
        recommendations = _disease_recommendations(pred_disease)

    # Alt diagnosis: majority vote points to a label different from pred
    qlora_alt = None
    if (agg_dis_correct is False
            and agg_final_label in DISEASE_NAMES
            and agg_final_label != pred_disease):
        alt_vi, alt_en = DISEASE_VI.get(agg_final_label, (agg_final_label, agg_final_label))
        qlora_alt = {
            "name"    : alt_vi,
            "nameEN"  : alt_en,
            "disease" : agg_final_label,
            "severity": SEVERITY_MAP.get(agg_final_label, "medium"),
            "source"  : "QLoRA Sequential — majority vote 3 cycles",
            "note"    : (
                f"QLoRA analyzed 3 independent cycles; majority vote suggests '{alt_vi}' ({alt_en}). "
                "CNN diagnosis remains the primary result — clinical correlation required."
            ),
        }

    return {
        "final_disease"       : pred_disease,
        "dis_vi"              : dis_vi,
        "dis_en"              : dis_en,
        "severity"            : severity,
        "confidence"          : dis_conf,
        "clinicalNote"        : clinical_note,
        "recommendations"     : recommendations,
        "is_correct"          : agg_dis_correct,
        "dis_correct"         : agg_dis_correct,
        "correct_flag"        : (f"Correct: {agg_dis_correct}"
                                 if agg_dis_correct is not None else None),
        "parse_ok"            : any(r.get("parse_ok", False) for r in valid),
        "gt_disease"          : agg_gt_disease,
        "qlora_alt_diagnosis" : qlora_alt,
        "final_disease_label" : agg_final_label,
        # qlora_steps not used at this level anymore
        "qlora_steps"         : None,
        # Individual per-cycle results — primary key for UI rendering
        "qlora_per_cycle"     : [
            {
                "cycle_index" : r.get("_cycle_index"),
                "cycle_rank"  : r.get("_cycle_rank"),
                "cycle_event" : r.get("_cycle_event"),
                "start_sec"   : r.get("_start_sec"),
                "end_sec"     : r.get("_end_sec"),
                "dis_correct" : r.get("dis_correct"),
                "ev_correct"  : r.get("ev_correct"),
                "parse_ok"    : r.get("parse_ok", False),
                "gt_disease"  : r.get("gt_disease"),
                "final_label" : r.get("final_disease_label", pred_disease),
                "qlora_steps" : {
                    "step1_event"     : r.get("step1_event_assessment", ""),
                    "step2_disease"   : r.get("step2_disease_assessment", ""),
                    "step3_retrieval" : r.get("step3_retrieval_signal", ""),
                    "step4_prototype" : r.get("step4_prototype_signal", ""),
                    "step5_conflict"  : r.get("step5_conflict_analysis", ""),
                    "step6_conclusion": r.get("step6_final_conclusion", ""),
                },
            }
            for r in valid
        ],
    }


def _fallback_clinical(pred_disease: str, dis_conf: int) -> dict:
    dis_vi, dis_en = DISEASE_VI.get(pred_disease, (pred_disease, pred_disease))
    return {
        "final_disease"       : pred_disease,
        "dis_vi"              : dis_vi,
        "dis_en"              : dis_en,
        "severity"            : SEVERITY_MAP.get(pred_disease, "medium"),
        "confidence"          : dis_conf,
        "clinicalNote"        : f"Lung sound analysis suggests: {dis_vi}.",
        "recommendations"     : _disease_recommendations(pred_disease),
        "is_correct"          : None,
        "dis_correct"         : None,
        "correct_flag"        : None,
        "parse_ok"            : False,
        "gt_disease"          : None,
        "qlora_alt_diagnosis" : None,
        "final_disease_label" : pred_disease,
        "qlora_steps"         : None,
        "qlora_per_cycle"     : [],
    }


def fallback_llm_v2(disease_name: str, dominant_event: str) -> dict:
    sound = (
        "wheeze"  if "Wheeze"  in dominant_event else
        "crackle" if "Crackle" in dominant_event else
        "abnormal sound"
    )
    _db = {
        "Healthy": {
            "final_disease": "Healthy",
            "dis_vi"       : DISEASE_VI["Healthy"][0],
            "dis_en"       : DISEASE_VI["Healthy"][1],
            "severity"     : "low",
            "confidence"   : 88,
            "clinicalNote" : (
                "Lung sounds are within normal limits, no abnormal sounds detected. "
                "Lungs are well-ventilated with no clear signs of obstruction or infection."
            ),
            "recommendations": _disease_recommendations("Healthy"),
        },
        "Infectious": {
            "final_disease": "Infectious",
            "dis_vi"       : DISEASE_VI["Infectious"][0],
            "dis_en"       : DISEASE_VI["Infectious"][1],
            "severity"     : "medium",
            "confidence"   : 78,
            "clinicalNote" : (
                f"Detected {sound} suggesting lower respiratory tract infection. "
                "Clinical evaluation combined with laboratory tests is needed to identify the causative agent."
            ),
            "recommendations": _disease_recommendations("Infectious"),
        },
        "Obstructive": {
            "final_disease": "Obstructive",
            "dis_vi"       : DISEASE_VI["Obstructive"][0],
            "dis_en"       : DISEASE_VI["Obstructive"][1],
            "severity"     : "high",
            "confidence"   : 82,
            "clinicalNote" : (
                "Lung sounds suggest chronic obstructive disease (COPD) or bronchial asthma. "
                "Prolonged expiratory wheeze is a characteristic sign; "
                "comprehensive pulmonary function assessment is required."
            ),
            "recommendations": _disease_recommendations("Obstructive"),
        },
    }
    base = _db.get(disease_name, {
        "final_disease"  : disease_name,
        "dis_vi"         : DISEASE_VI.get(disease_name, (disease_name, ""))[0],
        "dis_en"         : DISEASE_VI.get(disease_name, ("", disease_name))[1],
        "severity"       : "medium",
        "confidence"     : 65,
        "clinicalNote"   : "Insufficient data for clinical analysis.",
        "recommendations": ["Consult a respiratory specialist"],
    })
    base.update({
        "is_correct"          : None,
        "dis_correct"         : None,
        "correct_flag"        : None,
        "gt_disease"          : None,
        "parse_ok"            : False,
        "qlora_alt_diagnosis" : None,
        "final_disease_label" : disease_name,
        "qlora_steps"         : None,
        "qlora_per_cycle"     : [],
    })
    return base


def _disease_recommendations(disease: str) -> List[str]:
    _recs = {
        "Healthy": [
            "Continue regular health check-ups every 6–12 months",
            "Maintain a healthy lifestyle, avoid smoking and air pollution",
            "Engage in moderate aerobic exercise to maintain lung capacity",
            "Seek medical attention immediately if cough or shortness of breath persists beyond 1 week",
        ],
        "Infectious": [
            "Complete blood count and CRP tests to assess the degree of inflammation",
            "Chest X-ray (PA view) to rule out pneumonia or pleural effusion",
            "Consider broad-spectrum antibiotics if bacterial cause is suspected",
            "Monitor SpO₂, respiratory rate and body temperature every 4–6 hours",
        ],
        "Obstructive": [
            "Pulmonary function test (spirometry) to classify the degree of obstruction",
            "Evaluate the need for bronchodilator therapy (SABA and LABA)",
            "Avoid exposure to tobacco smoke, PM2.5 dust, cold and humid air",
            "Follow up with a respiratory specialist within 1–2 weeks",
        ],
    }
    return _recs.get(disease, ["Consult a respiratory specialist"])

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

_model:    Optional[DualBranchModel]    = None
_gcam_ev:  Optional[DualBranchGradCAM] = None
_gcam_dis: Optional[DualBranchGradCAM] = None


def _resolve_checkpoint(pattern: str) -> Optional[str]:
    if os.path.isfile(pattern):
        return pattern
    matches = sorted(_glob.glob(pattern))
    return matches[-1] if matches else None


def get_model() -> DualBranchModel:
    global _model, _gcam_ev, _gcam_dis
    if _model is not None:
        return _model

    log.info("Initializing DualBranchModel ...")
    m = DualBranchModel(use_cross=True).to(DEVICE)

    ckpt_path = _resolve_checkpoint(CHECKPOINT_PATH)
    if ckpt_path:
        log.info(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict):
            state = (state.get("model_state_dict")
                     or state.get("state_dict")
                     or state)
        m.load_state_dict(state, strict=True)
        log.info("Checkpoint loaded successfully ✓")
    else:
        log.warning(f"Checkpoint not found ({CHECKPOINT_PATH}) — using random weights")

    m.eval()
    _model    = m
    _gcam_ev  = DualBranchGradCAM(m, task="event")
    _gcam_dis = DualBranchGradCAM(m, task="disease")
    return m

# ═════════════════════════════════════════════════════════════════════════════
#  HTML — left empty, fill in later
# ═════════════════════════════════════════════════════════════════════════════

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<title>PneumoAI — Lung Sound Diagnostic</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,700;1,700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
:root {
  --bg:        #070b12;
  --bg2:       #0c1220;
  --surface:   rgba(255,255,255,0.032);
  --surface2:  rgba(255,255,255,0.058);
  --border:    rgba(150,210,255,0.10);
  --border2:   rgba(150,210,255,0.22);
  --c1:        #7ed8ff;
  --c2:        #a8edbe;
  --c3:        #c694e7;
  --text:      #e8f0ff;
  --text2:     #d1dce7;
  --text3:     #dae7f9;
  --red:       #ff7b7b;
  --amber:     #ffc46e;
  --green:     #7ef0a8;
  --f-display: 'Playfair Display', serif;
  --f-body: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  --f-mono: Consolas, "Courier New", monospace;
  --r-card:    20px;
  --r-pill:    100px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; -webkit-text-size-adjust: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--f-body);
  font-size: 15px;
  min-height: 100vh;
  overflow-x: hidden;
  line-height: 1.6;
}

/* ═══ LANGUAGE TOGGLE ═══ */
:root[lang="en"] .lang-vi { display: none !important; }
:root[lang="vi"] .lang-en { display: none !important; }
.lang-en, .lang-vi { display: contents; }

.lang-toggle {
  cursor: pointer;
  transition: all 0.3s ease;
  user-select: none;
}
.lang-toggle:hover {
  background: rgba(126,216,255,0.12);
  border-color: rgba(126,216,255,0.3);
  color: var(--c1);
}

#auroraCanvas {
  position: fixed; inset: 0; z-index: 0;
  pointer-events: none; opacity: 0.4;
}
body::after {
  content: ''; position: fixed; inset: 0; z-index: 1;
  pointer-events: none; opacity: 0.022;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size: 180px;
}
.page {
  position: relative; z-index: 10;
  max-width: 1380px; margin: 0 auto;
  padding: 0 clamp(16px, 4vw, 40px) 100px;
}
header {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
  padding: clamp(20px, 4vw, 40px) 0 0;
  animation: fadeDown .9s ease both;
}
.logo { display: flex; align-items: center; gap: 13px; }
.logo-icon {
  width: 40px; height: 40px; flex-shrink: 0;
  background: linear-gradient(135deg,rgba(126,216,255,.14),rgba(168,237,190,.09));
  border: 1px solid var(--border2); border-radius: 11px;
  display: flex; align-items: center; justify-content: center;
  position: relative; overflow: hidden;
}
.logo-icon::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(circle at 30% 30%,rgba(126,216,255,.2),transparent 60%);
}
.logo-icon svg { width: 20px; height: 20px; stroke: var(--c1); fill: none; stroke-width: 1.5; position: relative; z-index: 1; }
.logo-wordmark { line-height: 1.1; }
.logo-name { font-family: var(--f-display); font-size: 21px; font-weight: 600; letter-spacing: .3px; }
.logo-name em { font-style: italic; color: var(--c1); }
.logo-tag { font-family: var(--f-mono); font-size: 9px; color: var(--text3); letter-spacing: 2px; text-transform: uppercase; }
.hdr-pills { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pill {
  font-family: var(--f-mono); font-size: 10px;
  padding: 5px 13px; border-radius: var(--r-pill);
  border: 1px solid var(--border); background: var(--surface); color: var(--text2);
}
.pill-live {
  color: var(--c2); border-color: rgba(168,237,190,.25); background: rgba(168,237,190,.06);
  display: flex; align-items: center; gap: 6px;
}
.live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--c2); animation: pulse 2s ease infinite; }
.hero {
  display: grid;
  grid-template-columns: 1fr 720px;
  gap: clamp(28px, 3vw, 48px);
  align-items: center;
  min-height: 720px;
  padding-top: clamp(18px, 4vw, 26px);
  margin-bottom: clamp(35px, 7vw, 60px);
  overflow: visible;
}
.hero-content { animation: slideL 1s cubic-bezier(.22,1,.36,1) .15s both; }
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--f-mono); font-size: 10px; color: var(--c3);
  letter-spacing: 1.5px; text-transform: uppercase;
  background: rgba(232,197,255,.06); border: 1px solid rgba(232,197,255,.18);
  padding: 5px 14px; border-radius: var(--r-pill); margin-bottom: 20px;
}
.title-wrap { perspective: 700px; margin-bottom: 22px; }
.hero-title {
  font-family: var(--f-display);
  font-size: clamp(44px, 6.5vw, 50px);
  font-weight: 500; line-height: 1.45; letter-spacing: -1px;
}
.hero-title .ln1 { display: block; opacity: 0.85; letter-spacing: 1px; animation: textFloat 6s ease-in-out infinite; }
.hero-title .ln2 {
  display: block; font-weight: 600;
  background: linear-gradient(120deg, #7ed8ff, #a8edbe, #c694e7, #7ed8ff);
  background-size: 300% 100%;
  margin-left: 2%;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: gradientFlow 6s ease infinite;
  filter: drop-shadow(0 0 20px rgba(126,216,255,.25));
}
.hero-title .ln3 {
  display: block; margin-left: 4%;
  background: linear-gradient(120deg, #7ed8ff, #a8edbe, #c694e7, #7ed8ff);
  background-size: 300% 100%;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: breatheText 5.25s ease-in-out infinite;
  animation-delay: .5s;
  font-weight: 500; font-size: .78em; opacity: 0.7; letter-spacing: 1.5px;
}
.hero-desc { font-size: clamp(13px,1.5vw,15px); color: var(--text2); line-height: 1.85; max-width: 440px; margin-bottom: 32px; font-weight: 300; }
.hero-stats { display: flex; gap: clamp(20px,4vw,40px); flex-wrap: wrap; }
.stat-num {
  font-family: var(--f-body); font-size: clamp(30px,3.5vw,40px);
  font-weight: 600; font-style: italic;
  background: linear-gradient(135deg,var(--c1),var(--c2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  line-height: 1;
}
.stat-label { font-size: 11px; color: var(--text3); margin-top: 3px; font-family: var(--f-mono); letter-spacing: .4px; }
.stat-item::after { content: ''; display: block; width: 22px; height: 1.5px; background: linear-gradient(90deg,var(--c1),transparent); margin-top: 7px; }
.lung-visual {
  position: relative; display: flex; align-items: center; justify-content: center;
  width: 920px; height: 720px; flex-shrink: 0;
  animation: slideR 1s cubic-bezier(.22,1,.36,1) .3s both;
}
#lungCanvas { width: 720px; height: 720px; display: block; flex-shrink: 0; }
.lung-label {
  position: absolute;
  font-family: var(--f-mono); font-size: 9px; letter-spacing: 1px;
  color: var(--c1); background: rgba(7,11,18,.85);
  border: 1px solid rgba(126,216,255,.25);
  padding: 4px 10px; border-radius: 4px; backdrop-filter: blur(8px);
  white-space: nowrap; pointer-events: none;
}
.ll1 { top: 9%;  left: 12px; animation: breatheText 2.5s ease-in-out infinite; }
.ll2 { top: 48%; right: 28%; animation: breatheText 2.5s ease-in-out infinite; animation-delay: 1s; }
.ll3 { bottom: 9%; left: 12px; animation: breatheText 2.5s ease-in-out infinite; animation-delay: .5s; }
.sec-divider { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
.sec-divider span { font-family: var(--f-mono); font-size: 10px; color: var(--text3); letter-spacing: 2px; text-transform: uppercase; white-space: nowrap; }
.sec-line { flex: 1; height: 1px; background: linear-gradient(90deg,var(--border2),transparent); }
.models-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: 12px; margin-bottom: clamp(36px,5vw,56px);
}
.model-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card);
  padding: 18px 20px; display: flex; align-items: center; gap: 12px;
  position: relative; overflow: hidden; backdrop-filter: blur(12px);
  transition: all .35s cubic-bezier(.4,0,.2,1);
}
.model-card::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 0 0,rgba(126,216,255,.06),transparent 70%);
  opacity: 0; transition: opacity .35s;
}
.model-card:hover { border-color: var(--border2); transform: translateY(-3px); box-shadow: 0 16px 40px rgba(0,0,0,.4); }
.model-card:hover::before { opacity: 1; }
.model-icon { width: 42px; height: 42px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.mc1 .model-icon { background: rgba(126,216,255,.08); border: 1px solid rgba(126,216,255,.18); }
.mc2 .model-icon { background: rgba(168,237,190,.08); border: 1px solid rgba(168,237,190,.18); }
.model-icon svg { width: 19px; height: 19px; fill: none; stroke-width: 1.5; }
.mc1 .model-icon svg { stroke: var(--c1); }
.mc2 .model-icon svg { stroke: var(--c2); }
.model-info { flex: 1; min-width: 0; }
.model-name { font-size: 13px; font-weight: 500; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-desc { font-size: 11px; color: var(--text3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-tag { font-family: var(--f-mono); font-size: 10px; padding: 3px 10px; border-radius: 6px; flex-shrink: 0; white-space: nowrap; }
.mc1 .model-tag { background: rgba(126,216,255,.08); color: var(--c1); }
.mc2 .model-tag { background: rgba(168,237,190,.08); color: var(--c2); }

/* ═══ UPLOAD ZONE ═══ */
.upload-zone {
  position: relative;
  border: 1.5px dashed rgba(126,216,255,.18); border-radius: 26px;
  padding: clamp(36px,6vw,60px) clamp(20px,4vw,36px);
  text-align: center; cursor: pointer;
  background: var(--surface); overflow: hidden;
  margin-bottom: 12px;
  transition: border-color .4s, box-shadow .4s, background .4s, transform .3s cubic-bezier(.34,1.56,.64,1);
  backdrop-filter: blur(12px);
  user-select: none;
}
.upload-zone::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 110%,rgba(126,216,255,.04),transparent 55%);
  transition: opacity .4s; opacity: 1;
}
.upload-zone::after {
  content: ''; position: absolute; inset: 0; border-radius: 26px;
  background: radial-gradient(ellipse at 50% 60%, rgba(126,216,255,.09), transparent 60%);
  opacity: 0; transition: opacity .45s; pointer-events: none;
}
.upload-zone:hover { border-color: rgba(126,216,255,.45); box-shadow: 0 0 80px rgba(126,216,255,.08); transform: translateY(-2px); }
.upload-zone.drag-over { border-color: rgba(126,216,255,.7); border-style: solid; background: rgba(126,216,255,.06); box-shadow: 0 0 0 3px rgba(126,216,255,.12); transform: scale(1.008) translateY(-3px); }
.upload-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; z-index: 2; }
.upload-orb {
  width: clamp(56px,8vw,68px); height: clamp(56px,8vw,68px);
  border-radius: 50%; margin: 0 auto clamp(16px,3vw,24px);
  background: radial-gradient(circle,rgba(126,216,255,.1),rgba(126,216,255,.03));
  border: 1px solid rgba(126,216,255,.18);
  display: flex; align-items: center; justify-content: center;
  animation: breathe 4s ease-in-out infinite;
  position: relative; z-index: 1;
  transition: transform .3s cubic-bezier(.34,1.56,.64,1), background .3s, border-color .3s, box-shadow .3s;
}
.upload-zone:hover .upload-orb { transform: scale(1.12); border-color: rgba(126,216,255,.38); box-shadow: 0 0 28px rgba(126,216,255,.25); }
.upload-orb svg { width: 26px; height: 26px; stroke: var(--c1); fill: none; stroke-width: 1.5; position: relative; z-index: 1; }
.upload-title { font-family: var(--f-display); font-size: clamp(18px,3vw,22px); font-weight: 600; font-style: italic; margin-bottom: 8px; position: relative; z-index: 1; }
.upload-sub { font-size: clamp(12px,1.5vw,13px); color: var(--text2); margin-bottom: 20px; line-height: 1.7; position: relative; z-index: 1; font-weight: 300; }
.fmt-row { display: inline-flex; gap: 6px; flex-wrap: wrap; justify-content: center; position: relative; z-index: 1; }
.fmt { font-family: var(--f-mono); font-size: 10px; color: var(--text3); background: var(--surface2); border: 1px solid var(--border); border-radius: 5px; padding: 2px 8px; }

.file-preview {
  display: none;
  background: var(--surface2); border: 1px solid var(--border2); border-radius: 18px;
  padding: clamp(16px,3vw,22px) clamp(16px,3vw,24px);
  margin-bottom: 8px; backdrop-filter: blur(12px);
}
.file-preview.show { display: block; animation: fadeUp .4s ease both; }
.file-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.file-icon { width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0; background: rgba(126,216,255,.08); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; }
.file-icon svg { width: 17px; height: 17px; stroke: var(--c1); fill: none; stroke-width: 1.5; }
.file-name { font-size: 14px; font-weight: 500; word-break: break-all; }
.file-meta { font-family: var(--f-mono); font-size: 11px; color: var(--text2); margin-top: 2px; }
.file-rm { margin-left: auto; background: none; border: none; color: var(--text3); cursor: pointer; font-size: 22px; line-height: 1; padding: 2px 6px; transition: color .2s; flex-shrink: 0; }
.file-rm:hover { color: var(--red); }
#waveCanvas { width: 100%; height: 64px; display: block; border-radius: 9px; background: rgba(0,0,0,.2); }
.audio-controls { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
.play-btn { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0; background: linear-gradient(135deg, var(--c1), rgba(126,216,255,.6)); border: none; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 16px rgba(126,216,255,.25); transition: transform .15s, box-shadow .2s; }
.play-btn:hover { transform: scale(1.08); box-shadow: 0 6px 22px rgba(126,216,255,.35); }
.play-btn svg { width: 12px; height: 12px; fill: var(--bg); }
.audio-time { font-family: var(--f-mono); font-size: 11px; color: var(--text2); flex-shrink: 0; min-width: 80px; }
.audio-progress { flex: 1; height: 3px; background: var(--surface2); border-radius: 2px; cursor: pointer; }
.audio-progress-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg,var(--c1),var(--c2)); width: 0%; transition: width .1s; }
.analyze-wrap { text-align: center; margin: 20px 0 clamp(40px,6vw,72px); }
.analyze-btn {
  position: relative; overflow: hidden;
  background: transparent; border: 1px solid rgba(126,216,255,.3); border-radius: 13px;
  color: var(--c1); font-family: var(--f-body); font-size: 14px; font-weight: 500;
  padding: clamp(12px,2vw,14px) clamp(32px,5vw,48px);
  cursor: pointer; transition: all .3s; letter-spacing: .3px; backdrop-filter: blur(8px);
}
.analyze-btn::before { content: ''; position: absolute; inset: 0; background: linear-gradient(135deg,rgba(126,216,255,.1),rgba(168,237,190,.06)); opacity: 0; transition: opacity .3s; }
.analyze-btn:hover:not(:disabled)::before { opacity: 1; }
.analyze-btn:hover:not(:disabled) { border-color: rgba(126,216,255,.6); box-shadow: 0 0 36px rgba(126,216,255,.14); transform: translateY(-1px); }
.analyze-btn:disabled { opacity: .3; cursor: not-allowed; }
.btn-inner { display: flex; align-items: center; gap: 9px; position: relative; z-index: 1; }
.btn-pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--c2); animation: pulse 1.5s ease infinite; }
.results-section { display: none; }
.results-section.show { display: block; animation: fadeUp .7s ease both; }
.results-header { display: flex; align-items: center; gap: 14px; margin-bottom: clamp(20px,4vw,32px); flex-wrap: wrap; }
.results-header h2 { font-family: var(--f-display); font-size: clamp(24px,3.5vw,32px); font-style: italic; font-weight: 300; }
.results-line { flex: 1; height: 1px; background: linear-gradient(90deg,var(--border2),transparent); min-width: 20px; }
.results-ts { font-family: var(--f-mono); font-size: 10px; color: var(--text3); }
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
  gap: 16px; margin-bottom: 20px;
}
.result-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); overflow: hidden; backdrop-filter: blur(12px); }
.panel-head { padding: clamp(13px,2vw,16px) clamp(16px,3vw,22px); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.p-badge { font-family: var(--f-mono); font-size: 10px; padding: 3px 10px; border-radius: 5px; flex-shrink: 0; }
.p1 .p-badge { background: rgba(126,216,255,.1); color: var(--c1); }
.p2 .p-badge { background: rgba(168,237,190,.1); color: var(--c2); }
.p-name { font-size: 12px; font-weight: 500; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.p-conf { font-family: var(--f-mono); font-size: 13px; flex-shrink: 0; }
.panel-body { padding: clamp(16px,3vw,22px); }
.dx-box { background: rgba(0,0,0,.2); border: 1px solid var(--border); border-radius: 12px; padding: clamp(13px,2vw,16px) clamp(14px,2.5vw,18px); margin-bottom: 14px; }
.dx-label { font-family: var(--f-mono); font-size: 9px; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.dx-name { font-family: var(--f-display); font-size: clamp(18px,3vw,22px); font-style: italic; font-weight: 600; margin-bottom: 2px; }
.dx-en { font-size: 11px; color: var(--text2); margin-bottom: 12px; }
.prob-bar { height: 4px; background: rgba(255,255,255,.06); border-radius: 2px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 2px; transition: width 1.4s cubic-bezier(.4,0,.2,1) .3s; }
.fill-red   { background: linear-gradient(90deg,#ff7b7b,#ef4444); }
.fill-amber { background: linear-gradient(90deg,#ffc46e,#ff9800); }
.fill-green { background: linear-gradient(90deg,#7ef0a8,#00e676); }
.prob-meta { display: flex; justify-content: space-between; margin-top: 5px; }
.prob-meta span { font-family: var(--f-mono); font-size: 10px; color: var(--text3); }
.sound-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 500; padding: 5px 12px; border-radius: var(--r-pill); margin-bottom: 13px; }
.chip-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.chip-wheeze  { background: rgba(255,196,110,.1); color: var(--amber); border: 1px solid rgba(255,196,110,.25); }
.chip-crackle { background: rgba(255,123,123,.1); color: var(--red);   border: 1px solid rgba(255,123,123,.25); }
.chip-normal  { background: rgba(126,240,168,.1); color: var(--green);  border: 1px solid rgba(126,240,168,.25); }
.chip-rhonchi { background: rgba(126,216,255,.1); color: var(--c1);     border: 1px solid rgba(126,216,255,.25); }
.dd-list { display: flex; flex-direction: column; gap: 8px; }
.dd-item { display: flex; align-items: center; gap: 8px; }
.dd-name { font-size: 12px; color: var(--text2); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dd-track { flex-shrink: 0; width: 60px; height: 2px; background: rgba(255,255,255,.05); border-radius: 2px; }
.dd-bar { height: 100%; border-radius: 2px; background: var(--border2); transition: width 1.1s ease .5s; }
.dd-num { font-family: var(--f-mono); font-size: 10px; color: var(--text3); min-width: 28px; text-align: right; }
.dd-more-btn {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--f-mono); font-size: 11px;
  color: var(--c1); background: rgba(126,216,255,.06);
  border: 1px solid rgba(126,216,255,.2); border-radius: 8px;
  padding: 5px 13px; cursor: pointer;
  transition: background .25s, border-color .25s, transform .2s cubic-bezier(.34,1.56,.64,1);
  margin-top: 4px;
}
.dd-more-btn:hover { background: rgba(126,216,255,.12); border-color: rgba(126,216,255,.45); transform: translateY(-1px); }
.dd-more-btn .arr { display: inline-block; transition: transform .3s; }
.dd-more-btn.open .arr { transform: rotate(180deg); }
.dd-extra { max-height: 0; overflow: hidden; transition: max-height .45s cubic-bezier(.4,0,.2,1), opacity .35s; opacity: 0; margin-top: 0; }
.dd-extra.open { opacity: 1; margin-top: 8px; }
.sev-badge { display: inline-flex; align-items: center; gap: 5px; font-size: 10px; font-weight: 600; padding: 4px 11px; border-radius: 7px; text-transform: uppercase; letter-spacing: .5px; margin-top: 12px; }
.sev-high  { background: rgba(255,123,123,.1); color: var(--red);   border: 1px solid rgba(255,123,123,.2); }
.sev-med   { background: rgba(255,196,110,.1); color: var(--amber);  border: 1px solid rgba(255,196,110,.2); }
.sev-low   { background: rgba(126,240,168,.1); color: var(--green);  border: 1px solid rgba(126,240,168,.2); }

/* ═══ TOP-3 CYCLES DISEASE BLOCK ═══ */
.top3-disease-block {
  margin: 0 0 20px;
  background: var(--surface);
  border: 1px solid rgba(168,237,190,.18);
  border-radius: 22px;
  overflow: hidden;
  backdrop-filter: blur(12px);
  position: relative;
}
.top3-disease-block::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--c2), transparent);
  opacity: .45;
}
.t3d-head {
  padding: clamp(14px,2.5vw,18px) clamp(16px,3vw,24px);
  border-bottom: 1px solid rgba(168,237,190,.1);
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.t3d-head h3 { font-family: var(--f-display); font-size: clamp(16px,2.5vw,20px); font-style: italic; font-weight: 300; flex: 1; }
.t3d-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--c2); flex-shrink: 0; box-shadow: 0 0 10px rgba(168,237,190,.6); animation: conPulse 2s ease infinite; }
.t3d-tag { font-family: var(--f-mono); font-size: 9px; color: var(--c2); background: rgba(168,237,190,.07); border: 1px solid rgba(168,237,190,.2); padding: 4px 11px; border-radius: 5px; letter-spacing: .5px; flex-shrink: 0; }
.t3d-strategy { padding: 10px clamp(16px,3vw,24px); font-family: var(--f-mono); font-size: 10px; color: var(--text3); background: rgba(168,237,190,.03); border-bottom: 1px solid rgba(168,237,190,.07); line-height: 1.6; }
.t3d-body { padding: clamp(14px,2.5vw,18px) clamp(16px,3vw,24px); display: flex; flex-direction: column; gap: 16px; }
.t3d-cycle-card { background: rgba(0,0,0,.22); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; transition: border-color .3s, box-shadow .3s; position: relative; }
.t3d-cycle-card:hover { border-color: rgba(168,237,190,.3); box-shadow: 0 4px 20px rgba(168,237,190,.06); }
.t3d-cycle-card.rank-1 { border-color: rgba(168,237,190,.35) !important; box-shadow: 0 0 0 1px rgba(168,237,190,.12), 0 6px 22px rgba(168,237,190,.08) !important; }
.t3d-cycle-card.rank-1::before { content: '★ RANK #1'; position: absolute; top: 10px; right: 10px; z-index: 2; font-family: var(--f-mono); font-size: 8px; font-weight: 700; color: var(--c2); background: rgba(7,11,18,.92); border: 1px solid rgba(168,237,190,.45); border-radius: 5px; padding: 3px 8px; letter-spacing: .5px; }
.t3d-cycle-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 11px 14px; border-bottom: 1px solid var(--border); background: rgba(0,0,0,.15); }
.t3d-cycle-rank-badge { font-family: var(--f-mono); font-size: 9px; font-weight: 700; width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.t3d-cycle-rank-badge.r1 { background: rgba(168,237,190,.15); color: var(--c2); border: 1px solid rgba(168,237,190,.3); }
.t3d-cycle-rank-badge.r2 { background: rgba(126,216,255,.1); color: var(--c1); border: 1px solid rgba(126,216,255,.22); }
.t3d-cycle-rank-badge.r3 { background: rgba(218,231,249,.07); color: var(--text3); border: 1px solid rgba(218,231,249,.15); }
.t3d-cycle-time { font-family: var(--f-mono); font-size: 11px; color: var(--c1); flex-shrink: 0; }
.t3d-cycle-event-chip { display: inline-flex; align-items: center; gap: 5px; font-size: 10px; font-weight: 500; padding: 3px 10px; border-radius: var(--r-pill); }
.t3d-cycle-event-chip.abnormal { background: rgba(255,196,110,.08); color: var(--amber); border: 1px solid rgba(255,196,110,.22); }
.t3d-cycle-event-chip.normal   { background: rgba(126,240,168,.08); color: var(--green); border: 1px solid rgba(126,240,168,.22); }
.t3d-cycle-event-chip .t3d-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; }
.t3d-cycle-ev-conf { font-family: var(--f-mono); font-size: 10px; color: var(--text3); margin-left: auto; }
.t3d-cycle-content { display: grid; grid-template-columns: auto 1fr; gap: 14px; padding: 13px 14px; }
.t3d-gradcam-img-wrap { width: 140px; flex-shrink: 0; align-self: start; }
.t3d-gradcam-img-wrap img { width: 100%; border-radius: 8px; display: block; border: 1px solid var(--border); transition: border-color .2s; }
.t3d-gradcam-label { font-family: var(--f-mono); font-size: 9px; color: var(--text3); text-align: center; margin-top: 5px; letter-spacing: .5px; }
.t3d-metrics-col { display: flex; flex-direction: column; gap: 10px; }
.t3d-disease-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.t3d-disease-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 5px 13px; border-radius: 9px; }
.t3d-disease-badge.pred { background: rgba(168,237,190,.09); color: var(--c2); border: 1px solid rgba(168,237,190,.25); }
.t3d-disease-badge.alt  { background: rgba(255,196,110,.06); color: var(--amber); border: 1px solid rgba(255,196,110,.18); }
.t3d-disease-badge-key { font-family: var(--f-mono); font-size: 8px; color: var(--text3); text-transform: uppercase; letter-spacing: .5px; }
.t3d-cam-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
.t3d-cam-item { background: rgba(0,0,0,.2); border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; }
.t3d-cam-key { font-family: var(--f-mono); font-size: 8px; color: var(--text3); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
.t3d-cam-val { font-family: var(--f-mono); font-size: 11px; font-weight: 500; color: var(--c1); }
.t3d-cam-val.green  { color: var(--c2); }
.t3d-cam-val.amber  { color: var(--amber); }
.t3d-cam-val.purple { color: var(--c3); }
.t3d-cam-bar-wrap { background: rgba(0,0,0,.18); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.t3d-cam-bar-row { display: flex; justify-content: space-between; align-items: center; font-family: var(--f-mono); font-size: 9px; color: var(--text3); margin-bottom: 5px; }
.t3d-cam-bar-row span { color: var(--c2); font-size: 10px; }
.t3d-cam-bar { height: 4px; background: rgba(255,255,255,.06); border-radius: 2px; overflow: hidden; margin-bottom: 5px; }
.t3d-cam-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--c1), var(--c2)); width: 0%; transition: width 1.4s cubic-bezier(.4,0,.2,1) .4s; }
.t3d-cam-fill.alt-fill { background: linear-gradient(90deg, var(--amber), var(--red)); }
.t3d-diff-row { background: rgba(198,148,231,.04); border: 1px solid rgba(198,148,231,.15); border-radius: 9px; padding: 9px 11px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.t3d-diff-label { font-family: var(--f-mono); font-size: 9px; color: var(--c3); text-transform: uppercase; letter-spacing: .8px; flex-shrink: 0; }
.t3d-diff-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.t3d-diff-chip { font-family: var(--f-mono); font-size: 10px; padding: 2px 9px; border-radius: 5px; background: rgba(0,0,0,.2); border: 1px solid var(--border); color: var(--text2); display: inline-flex; align-items: center; gap: 5px; }
.t3d-diff-chip span { color: var(--c3); }

/* ═══ CONSENSUS ═══ */
.consensus {
  background: var(--surface); border: 1px solid var(--border2); border-radius: 24px;
  overflow: hidden; backdrop-filter: blur(16px); position: relative; margin-bottom: 20px;
}
.consensus::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg,transparent,var(--c1),var(--c2),transparent); opacity: .35; }
.con-head { padding: clamp(16px,3vw,22px) clamp(20px,3vw,28px); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.con-head h3 { font-family: var(--f-display); font-size: clamp(18px,3vw,22px); font-style: italic; font-weight: 300; flex: 1; }
.con-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--c1); flex-shrink: 0; box-shadow: 0 0 10px rgba(126,216,255,.6); animation: conPulse 2s ease infinite; }
.con-tag { font-family: var(--f-mono); font-size: 9px; color: var(--c1); background: rgba(126,216,255,.07); border: 1px solid var(--border); padding: 4px 10px; border-radius: 5px; letter-spacing: .5px; flex-shrink: 0; }
.con-body { padding: clamp(20px,4vw,28px); }
.con-grid { display: grid; grid-template-columns: 1fr auto; gap: clamp(16px,3vw,28px); align-items: start; }
.con-dx-label { font-family: var(--f-mono); font-size: 10px; color: var(--c1); letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
.con-dx-name { font-family: var(--f-display); font-size: clamp(28px,5vw,38px); font-style: italic; font-weight: 600; line-height: 1.05; margin-bottom: 4px; }
.con-dx-en { font-size: 13px; color: var(--text2); margin-bottom: 18px; font-weight: 300; }
.con-note { font-size: 13px; color: var(--text2); line-height: 1.8; border-left: 2px solid rgba(126,216,255,.3); padding-left: 14px; margin-bottom: 20px; font-weight: 300; }
.rec-title { font-size: 12px; font-weight: 500; margin-bottom: 11px; letter-spacing: .5px; }
.rec-list { display: flex; flex-direction: column; gap: 9px; }
.rec-item { display: flex; align-items: flex-start; gap: 8px; font-size: 13px; color: var(--text2); line-height: 1.5; font-weight: 300; }
.rec-bullet { width: 18px; height: 18px; border-radius: 5px; flex-shrink: 0; margin-top: 2px; background: rgba(126,216,255,.07); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; }
.rec-bullet svg { width: 10px; height: 10px; stroke: var(--c1); fill: none; stroke-width: 2; }
.con-stats { display: flex; flex-direction: column; gap: 11px; background: rgba(0,0,0,.2); border: 1px solid var(--border); border-radius: 14px; padding: clamp(14px,2.5vw,20px); text-align: center; min-width: 120px; }
.con-stat-num { font-family: var(--f-display); font-size: clamp(26px,4vw,34px); font-style: italic; font-weight: 600; color: var(--c1); line-height: 1; }
.con-stat-label { font-family: var(--f-mono); font-size: 9px; color: var(--text3); margin-top: 2px; }
.con-stat-div { height: 1px; background: var(--border); }
.card-section { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-card); overflow: hidden; backdrop-filter: blur(12px); margin-bottom: 20px; }
.card-head { padding: clamp(14px,2.5vw,18px) clamp(16px,3vw,24px); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.card-head h3 { font-family: var(--f-display); font-size: clamp(16px,2.5vw,20px); font-style: italic; font-weight: 300; flex: 1; }
.cycles-body { padding: clamp(14px,3vw,18px) clamp(14px,3vw,20px); display: grid; grid-template-columns: repeat(auto-fill, minmax(clamp(140px, 22%, 210px), 1fr)); gap: 11px; }
.cycle-card { background: rgba(0,0,0,.2); border: 1px solid var(--border); border-radius: 11px; overflow: hidden; transition: border-color .25s, box-shadow .25s; }
.cycle-card:hover { border-color: var(--border2); box-shadow: 0 4px 16px rgba(0,0,0,.3); }
.cycle-card.top3-highlight { border-color: rgba(168,237,190,.4) !important; box-shadow: 0 0 0 1px rgba(168,237,190,.14), 0 5px 18px rgba(168,237,190,.1) !important; position: relative; }
.cycle-card.top3-highlight::before { content: 'TOP-3'; position: absolute; top: 6px; right: 6px; z-index: 2; font-family: var(--f-mono); font-size: 8px; font-weight: 700; color: var(--c2); background: rgba(7,11,18,.9); border: 1px solid rgba(168,237,190,.4); border-radius: 4px; padding: 2px 6px; letter-spacing: .5px; }
.cycle-img { width: 100%; height: 90px; object-fit: cover; display: block; background: rgba(0,0,0,.3); }
.cycle-info { padding: 9px 11px; }
.cycle-time { font-family: var(--f-mono); font-size: 10px; color: var(--text3); }
.cycle-event { font-size: 12px; font-weight: 500; margin: 4px 0 2px; }
.cycle-conf { font-family: var(--f-mono); font-size: 10px; color: var(--c1); }
.cycle-peak { font-family: var(--f-mono); font-size: 10px; color: var(--text3); margin-top: 2px; }
.gradcam-body { padding: clamp(14px,3vw,18px) clamp(14px,3vw,20px); display: flex; flex-wrap: wrap; gap: 10px; }
.gradcam-img-wrap { border-radius: 9px; overflow: hidden; border: 1px solid var(--border); transition: border-color .25s; }
.gradcam-img-wrap:hover { border-color: var(--border2); }
.gradcam-img-wrap img { display: block; height: clamp(90px,14vw,120px); width: auto; max-width: 100%; }

/* ═══ QLORA PANEL v5.2 ═══ */
.qlora-panel {
  background: var(--surface);
  border: 1px solid rgba(198,148,231,.2);
  border-radius: var(--r-card);
  overflow: hidden;
  backdrop-filter: blur(12px);
  margin-bottom: 20px;
  position: relative;
}
.qlora-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--c3), transparent);
  opacity: .4;
}
.qlora-head {
  padding: clamp(14px,2.5vw,18px) clamp(16px,3vw,24px);
  border-bottom: 1px solid rgba(198,148,231,.12);
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  cursor: pointer; user-select: none;
  transition: background .25s;
}
.qlora-head:hover { background: rgba(198,148,231,.04); }
.qlora-head h3 { font-family: var(--f-display); font-size: clamp(16px,2.5vw,20px); font-style: italic; font-weight: 300; flex: 1; }
.qlora-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--c3); flex-shrink: 0; box-shadow: 0 0 10px rgba(198,148,231,.6); animation: conPulse 2s ease infinite; }
.qlora-header-badges { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.qlora-source-badge { font-family: var(--f-mono); font-size: 9px; padding: 4px 11px; border-radius: 5px; letter-spacing: .5px; display: inline-flex; align-items: center; gap: 5px; }
.qsrc-qlora    { background: rgba(198,148,231,.1); color: var(--c3); border: 1px solid rgba(198,148,231,.25); }
.qsrc-fallback { background: rgba(255,196,110,.08); color: var(--amber); border: 1px solid rgba(255,196,110,.2); }
.qlora-verdict-badge { font-family: var(--f-mono); font-size: 9px; font-weight: 600; padding: 4px 11px; border-radius: 5px; letter-spacing: .5px; display: inline-flex; align-items: center; gap: 5px; }
.qverdict-correct   { background: rgba(126,240,168,.08); color: var(--green); border: 1px solid rgba(126,240,168,.2); }
.qverdict-incorrect { background: rgba(255,123,123,.08); color: var(--red);   border: 1px solid rgba(255,123,123,.2); }
.qverdict-unknown   { background: rgba(218,231,249,.05); color: var(--text3); border: 1px solid rgba(218,231,249,.1); }
.qlora-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.qlora-toggle-btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--f-mono); font-size: 10px; color: var(--c3);
  background: rgba(198,148,231,.06); border: 1px solid rgba(198,148,231,.2);
  border-radius: 7px; padding: 4px 12px; cursor: pointer; flex-shrink: 0;
  transition: background .25s, border-color .25s, transform .2s cubic-bezier(.34,1.56,.64,1);
}
.qlora-toggle-btn:hover { background: rgba(198,148,231,.14); border-color: rgba(198,148,231,.45); transform: translateY(-1px); }
.qlora-toggle-btn .qarr { display: inline-block; transition: transform .3s; }
.qlora-toggle-btn.open .qarr { transform: rotate(180deg); }
.qlora-body-wrap { max-height: 0; overflow: hidden; transition: max-height .5s cubic-bezier(.4,0,.2,1), opacity .4s ease; opacity: 0; }
.qlora-body-wrap.open { opacity: 1; }
.qlora-body { padding: clamp(16px,3vw,22px) clamp(16px,3vw,24px); display: flex; flex-direction: column; gap: 24px; }

/* Verdict banner */
.qlora-verdict-banner { border-radius: 12px; padding: 14px 18px; display: flex; align-items: flex-start; gap: 12px; }
.qlora-verdict-banner.correct   { background: rgba(126,240,168,.05); border: 1px solid rgba(126,240,168,.18); }
.qlora-verdict-banner.incorrect { background: rgba(255,123,123,.05); border: 1px solid rgba(255,123,123,.18); }
.qlora-verdict-banner.unknown   { background: rgba(218,231,249,.03); border: 1px solid rgba(218,231,249,.08); }
.qlora-verdict-icon { width: 34px; height: 34px; border-radius: 9px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.correct   .qlora-verdict-icon { background: rgba(126,240,168,.1); }
.incorrect .qlora-verdict-icon { background: rgba(255,123,123,.1); }
.unknown   .qlora-verdict-icon { background: rgba(218,231,249,.06); }
.qlora-verdict-icon svg { width: 16px; height: 16px; fill: none; stroke-width: 2; }
.correct   .qlora-verdict-icon svg { stroke: var(--green); }
.incorrect .qlora-verdict-icon svg { stroke: var(--red); }
.unknown   .qlora-verdict-icon svg { stroke: var(--text3); }
.qlora-verdict-label { font-family: var(--f-mono); font-size: 10px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.correct   .qlora-verdict-label { color: var(--green); }
.incorrect .qlora-verdict-label { color: var(--red); }
.unknown   .qlora-verdict-label { color: var(--text3); }
.qlora-verdict-desc { font-size: 12px; color: var(--text2); line-height: 1.6; }

/* Section title */
.qlora-section-title {
  font-family: var(--f-mono); font-size: 9px; color: var(--text3);
  text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 11px;
  display: flex; align-items: center; gap: 8px;
}
.qlora-section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(198,148,231,.2), transparent); }

/* Aggregated pred grid */
.qlora-pred-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
.qlora-pred-card { background: rgba(0,0,0,.25); border: 1px solid var(--border); border-radius: 11px; padding: 12px 15px; transition: border-color .25s; }
.qlora-pred-card:hover { border-color: rgba(198,148,231,.3); }
.qlora-pred-key { font-family: var(--f-mono); font-size: 9px; color: var(--c3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.qlora-pred-val { font-size: 13px; font-weight: 500; color: var(--text); line-height: 1.4; }
.qlora-pred-val.mono  { font-family: var(--f-mono); font-size: 12px; color: var(--amber); }
.qlora-pred-val.green { color: var(--green); }
.qlora-pred-val.red   { color: var(--red); }
.qlora-pred-val.dim   { color: var(--text3); font-style: italic; }
.qlora-pred-val.amber-c { color: var(--amber); }

/* ── PER-CYCLE TABS ── */
.qlora-cycle-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
.qlora-cycle-tab {
  font-family: var(--f-mono); font-size: 10px;
  padding: 5px 14px; border-radius: 8px; cursor: pointer;
  border: 1px solid var(--border); background: rgba(0,0,0,.2);
  color: var(--text3); display: inline-flex; align-items: center; gap: 6px;
  transition: all .2s cubic-bezier(.34,1.56,.64,1);
  user-select: none;
}
.qlora-cycle-tab:hover { border-color: rgba(198,148,231,.35); color: var(--c3); }
.qlora-cycle-tab.active {
  background: rgba(198,148,231,.12); border-color: rgba(198,148,231,.4);
  color: var(--c3); box-shadow: 0 0 14px rgba(198,148,231,.12);
}
.qlora-cycle-tab .tab-verdict-dot { width: 6px; height: 6px; border-radius: 50%; }
.qlora-cycle-tab .tab-verdict-dot.correct   { background: var(--green); }
.qlora-cycle-tab .tab-verdict-dot.incorrect { background: var(--red); }
.qlora-cycle-tab .tab-verdict-dot.unknown   { background: var(--text3); }

/* Per-cycle panel */
.qlora-cycle-panel { display: none; animation: fadeUp .3s ease both; }
.qlora-cycle-panel.active { display: block; }

/* Cycle summary row */
.qlora-cycle-summary {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  padding: 10px 14px; background: rgba(0,0,0,.2);
  border: 1px solid var(--border); border-radius: 10px; margin-bottom: 12px;
}
.qlora-cycle-summary-item { font-family: var(--f-mono); font-size: 10px; color: var(--text2); display: inline-flex; align-items: center; gap: 5px; }
.qlora-cycle-summary-item span { color: var(--c3); }
.qlora-cycle-summary-item.vc { color: var(--green); }
.qlora-cycle-summary-item.vi { color: var(--red); }
.qlora-cycle-summary-item.vu { color: var(--text3); }

/* 6-Step cards */
.qlora-steps-grid { display: flex; flex-direction: column; gap: 8px; }
.qlora-step-card {
  background: rgba(0,0,0,.2); border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden;
  transition: border-color .25s;
}
.qlora-step-card:hover { border-color: rgba(198,148,231,.25); }
.qlora-step-card.has-content { border-color: rgba(198,148,231,.18); }
.qlora-step-head {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; background: rgba(0,0,0,.18);
  cursor: pointer; user-select: none;
  transition: background .2s;
}
.qlora-step-head:hover { background: rgba(198,148,231,.06); }
.qlora-step-num {
  font-family: var(--f-mono); font-size: 9px; font-weight: 700;
  padding: 3px 9px; border-radius: 5px; flex-shrink: 0;
  background: rgba(198,148,231,.12); color: var(--c3);
  border: 1px solid rgba(198,148,231,.25); letter-spacing: .5px;
}
.qlora-step-label { font-family: var(--f-mono); font-size: 10px; color: var(--text2); flex: 1; }
.qlora-step-status { font-family: var(--f-mono); font-size: 9px; flex-shrink: 0; }
.qlora-step-status.ok  { color: var(--green); }
.qlora-step-status.empty { color: var(--text3); }
.qlora-step-arr { font-size: 10px; color: var(--c3); flex-shrink: 0; transition: transform .3s; }
.qlora-step-card.expanded .qlora-step-arr { transform: rotate(180deg); }
.qlora-step-body-wrap { max-height: 0; overflow: hidden; transition: max-height .4s cubic-bezier(.4,0,.2,1), opacity .3s; opacity: 0; }
.qlora-step-body-wrap.open { opacity: 1; }
.qlora-step-body {
  padding: 12px 14px;
  font-size: 12.5px; color: var(--text2); line-height: 1.75; font-weight: 300;
  white-space: pre-wrap; word-break: break-word;
  max-height: 320px; overflow-y: auto;
  scrollbar-width: thin; scrollbar-color: rgba(198,148,231,.3) transparent;
  border-top: 1px solid var(--border);
}
.qlora-step-body::-webkit-scrollbar { width: 3px; }
.qlora-step-body::-webkit-scrollbar-thumb { background: rgba(198,148,231,.3); border-radius: 2px; }
.qlora-step-body.empty-body { color: var(--text3); font-style: italic; font-size: 11px; }

/* Input signals */
.qlora-signals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 9px; }
.qlora-signal-card { background: rgba(0,0,0,.2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 13px; transition: border-color .2s; }
.qlora-signal-card:hover { border-color: rgba(198,148,231,.25); }
.qlora-signal-key { font-family: var(--f-mono); font-size: 8px; color: var(--c3); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 5px; }
.qlora-signal-val { font-family: var(--f-mono); font-size: 11px; color: var(--text2); line-height: 1.4; }
.qlora-signal-val.green { color: var(--c2); }
.qlora-signal-val.amber { color: var(--amber); }

/* Alt diagnosis alert */
.qlora-alt-alert { background: rgba(255,123,123,.04); border: 1px solid rgba(255,123,123,.18); border-radius: 12px; padding: 14px 18px; display: flex; align-items: flex-start; gap: 11px; }
.qlora-alt-alert-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
.qlora-alt-alert-body { flex: 1; }
.qlora-alt-alert-title { font-family: var(--f-mono); font-size: 10px; font-weight: 600; color: var(--amber); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.qlora-alt-alert-name { font-family: var(--f-display); font-size: 17px; font-style: italic; font-weight: 600; margin-bottom: 4px; color: var(--red); }
.qlora-alt-alert-note { font-size: 12px; color: var(--text2); line-height: 1.6; }

/* Uncertainty */
.qlora-unc-row { display: flex; gap: 10px; flex-wrap: wrap; }
.qlora-unc-chip { font-family: var(--f-mono); font-size: 10px; padding: 4px 11px; border-radius: 7px; background: rgba(0,0,0,.2); border: 1px solid var(--border); color: var(--text2); display: inline-flex; align-items: center; gap: 6px; }
.qlora-unc-chip span { color: var(--amber); }

/* Majority vote summary */
.qlora-vote-summary {
  background: rgba(198,148,231,.04); border: 1px solid rgba(198,148,231,.15);
  border-radius: 12px; padding: 14px 18px; margin-bottom: 4px;
}
.qlora-vote-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
.qlora-vote-label { font-family: var(--f-mono); font-size: 9px; color: var(--c3); text-transform: uppercase; letter-spacing: .8px; flex-shrink: 0; }
.qlora-vote-chips { display: flex; gap: 6px; flex-wrap: wrap; }
.qlora-vote-chip { font-family: var(--f-mono); font-size: 10px; padding: 3px 10px; border-radius: 5px; background: rgba(0,0,0,.2); border: 1px solid var(--border); color: var(--text2); }
.qlora-vote-chip.winner { background: rgba(168,237,190,.08); color: var(--c2); border-color: rgba(168,237,190,.3); }

/* Fallback / empty */
.qlora-empty { text-align: center; padding: 32px 20px; font-family: var(--f-mono); font-size: 11px; color: var(--text3); }
.qlora-empty svg { width: 30px; height: 30px; stroke: rgba(198,148,231,.35); fill: none; stroke-width: 1.5; margin: 0 auto 12px; display: block; }

/* ═══ LOADING & TOAST ═══ */
.loading-overlay { display: none; position: fixed; inset: 0; background: rgba(7,11,18,.88); backdrop-filter: blur(16px); z-index: 1000; flex-direction: column; align-items: center; justify-content: center; gap: 22px; }
.loading-overlay.show { display: flex; }
.load-rings { position: relative; width: 68px; height: 68px; }
.lr1 { width: 68px; height: 68px; border: 1.5px solid var(--border); border-top-color: var(--c1); border-radius: 50%; animation: spin 1s linear infinite; position: absolute; }
.lr2 { position: absolute; inset: 11px; border: 1px solid transparent; border-top-color: var(--c2); border-radius: 50%; animation: spin .65s linear infinite reverse; }
.load-txt { font-family: var(--f-display); font-size: clamp(15px,3vw,18px); font-style: italic; color: var(--text2); text-align: center; padding: 0 20px; }
.load-sub { font-family: var(--f-mono); font-size: 10px; color: var(--text3); letter-spacing: .5px; text-align: center; }
.load-prog { width: 160px; height: 1.5px; background: var(--surface2); border-radius: 1px; overflow: hidden; }
.load-prog-fill { height: 100%; background: linear-gradient(90deg,var(--c1),var(--c2)); animation: loadAnim 2s ease-in-out infinite; }
.toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%) translateY(60px); background: var(--surface2); border: 1px solid rgba(255,123,123,.25); border-radius: 10px; padding: 10px 20px; font-size: 13px; color: var(--red); transition: transform .3s cubic-bezier(.4,0,.2,1); z-index: 2000; backdrop-filter: blur(12px); max-width: min(90vw, 440px); text-align: center; }
.toast.show { transform: translateX(-50%) translateY(0); }
.toast.ok { color: var(--green); border-color: rgba(126,240,168,.25); }

/* ═══ ANIMATIONS ═══ */
@keyframes pulse    { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(168,237,190,.5);}50%{opacity:.6;box-shadow:0 0 0 5px rgba(168,237,190,0);} }
@keyframes breathe  { 0%,100%{transform:scale(1);}50%{transform:scale(1.05);} }
@keyframes fadeDown { from{opacity:0;transform:translateY(-14px);}to{opacity:1;transform:none;} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:none;} }
@keyframes slideL   { from{opacity:0;transform:translateX(-24px);}to{opacity:1;transform:none;} }
@keyframes slideR   { from{opacity:0;transform:translateX(24px);}to{opacity:1;transform:none;} }
@keyframes spin     { to{transform:rotate(360deg);} }
@keyframes loadAnim { 0%{width:0%;margin-left:0;}60%{width:75%;margin-left:0;}100%{width:0%;margin-left:100%;} }
@keyframes conPulse { 0%,100%{box-shadow:0 0 7px rgba(126,216,255,.5);}50%{box-shadow:0 0 20px rgba(126,216,255,.8);} }
@keyframes breatheText { 0%,100%{opacity:.6;letter-spacing:1px;}50%{opacity:1;letter-spacing:2px;} }
@keyframes gradientFlow { 0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;} }
@keyframes textFloat { 0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);} }

/* ═══ RESPONSIVE ═══ */
@media (max-width: 1100px) {
  .hero { grid-template-columns: 1fr; gap: 28px; min-height: unset; }
  .lung-visual { width: min(720px, 100vw - 32px); height: min(720px, 100vw - 32px); margin: 0 auto; order: -1; }
  #lungCanvas { width: 100%; height: 100%; }
}
@media (max-width: 700px) {
  .lung-visual { width: min(480px, 100vw - 32px); height: min(480px, 100vw - 32px); }
  .con-grid { grid-template-columns: 1fr; }
  .con-stats { flex-direction: row; flex-wrap: wrap; justify-content: space-around; min-width: unset; }
  .con-stat-div { display: none; }
  .cycles-body { grid-template-columns: 1fr 1fr; }
  .qlora-pred-grid { grid-template-columns: 1fr 1fr; }
  .t3d-cycle-content { grid-template-columns: 1fr; }
  .t3d-gradcam-img-wrap { width: 100%; }
  .t3d-cam-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 420px) {
  .cycles-body { grid-template-columns: 1fr; }
  .models-row { grid-template-columns: 1fr; }
  .hdr-pills .pill:not(.pill-live):not(.lang-toggle) { display: none; }
  .qlora-pred-grid { grid-template-columns: 1fr; }
  .t3d-cam-grid { grid-template-columns: 1fr; }
  .qlora-signals-grid { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<canvas id="auroraCanvas"></canvas>

<div class="loading-overlay" id="loadingOverlay">
  <div class="load-rings"><div class="lr1"></div><div class="lr2"></div></div>
  <div class="load-txt" id="loadingText">
    <span class="lang-en">Analyzing lung sounds...</span>
    <span class="lang-vi">Đang phân tích âm thanh phổi…</span>
  </div>
  <div class="load-sub" id="loadingSub">
    <span class="lang-en">Connecting to AI server...</span>
    <span class="lang-vi">Kết nối tới server AI...</span>
  </div>
  <div class="load-prog"><div class="load-prog-fill"></div></div>
</div>

<div class="toast" id="toast"></div>

<div class="page">
  <header>
    <div class="logo">
      <div class="logo-icon">
        <svg viewBox="0 0 24 24"><path d="M12 2L4 6v6c0 5.25 3.75 10.15 8 11.5C16.25 22.15 20 17.25 20 12V6L12 2z"/><path d="M9 12l2 2 4-4"/></svg>
      </div>
      <div class="logo-wordmark">
        <div class="logo-name">Pneumo<em>AI</em></div>
        <div class="logo-tag">Lung Diagnostic System</div>
      </div>
    </div>
    <div class="hdr-pills">
      <div class="pill pill-live"><div class="live-dot"></div>SYSTEM ONLINE</div>
      <div class="pill" id="modelStatusPill">Model: —</div>
      <div class="pill lang-toggle" onclick="toggleLanguage()" title="Toggle Language">
        <span class="lang-en">EN / <b style="color:var(--text3); font-weight:400">VI</b></span>
        <span class="lang-vi"><b style="color:var(--text3); font-weight:400">EN</b> / VI</span>
      </div>
    </div>
  </header>

  <div class="hero">
    <div class="hero-content">
      <div class="hero-eyebrow">
        <span class="lang-en">⬡ AI-Powered Respiratory Analysis</span>
        <span class="lang-vi">⬡ Phân tích hô hấp bằng AI</span>
      </div>
      <div class="title-wrap">
        <h1 class="hero-title">
          <span class="ln1"><span class="lang-en">Listen to</span><span class="lang-vi">Lắng nghe</span></span>
          <span class="ln2"><span class="lang-en">the lung's rhythm</span><span class="lang-vi">nhịp thở của phổi</span></span>
          <span class="ln2"><span class="lang-en">before the body speaks.</span><span class="lang-vi">trước khi cơ thể lên tiếng.</span></span>
        </h1>
      </div>
      <p class="hero-desc">
        <span class="lang-en">Integrating advanced deep learning architectures and medical language models, the system supports early detection and assessment of respiratory abnormalities based on stethoscope recordings.</span>
        <span class="lang-vi">Tích hợp các kiến trúc học sâu tiên tiến và mô hình ngôn ngữ y tế, hệ thống hỗ trợ phát hiện sớm và đánh giá các bất thường hô hấp dựa trên dữ liệu ghi âm từ stethoscope.</span>
      </p>
      <div class="hero-stats">
        <div class="stat-item"><div class="stat-num">93.4%</div><div class="stat-label"><span class="lang-en">Accuracy</span><span class="lang-vi">Độ chính xác</span></div></div>
        <div class="stat-item"><div class="stat-num">&lt;3s</div><div class="stat-label"><span class="lang-en">Analysis</span><span class="lang-vi">Phân tích</span></div></div>
        <div class="stat-item"><div class="stat-num">3</div><div class="stat-label"><span class="lang-en">Disease Groups</span><span class="lang-vi">Nhóm bệnh</span></div></div>
      </div>
    </div>
    <div class="lung-visual">
      <canvas id="lungCanvas"></canvas>
      <div class="lung-label ll1">BRONCHIAL TREE</div>
      <div class="lung-label ll2">O₂ EXCHANGE</div>
      <div class="lung-label ll3">ALVEOLI · ACTIVE</div>
    </div>
  </div>

  <div class="sec-divider">
    <span><span class="lang-en">AI Models</span><span class="lang-vi">Mô hình AI</span></span>
    <div class="sec-line"></div>
  </div>
  <div class="models-row">
    <div class="model-card mc1">
      <div class="model-icon">
        <svg viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="5"/><path d="M7 12h10M12 7v10"/></svg>
      </div>
      <div class="model-info">
        <div class="model-name">
          <span class="lang-en">Multi-task Deep Learning based on ResNet18 + FPN</span>
          <span class="lang-vi">Mô hình học sâu đa nhiệm dựa trên ResNet18 kết hợp FPN</span>
        </div>
        <div class="model-desc">LogMel-spectrogram · Event & Disease classifier</div>
      </div>
      <div class="model-tag">CNN + PatientAttn</div>
    </div>
    <div class="model-card mc2">
      <div class="model-icon">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3c0 0 4 4 4 9s-4 9-4 9"/><path d="M3 12h18"/></svg>
      </div>
      <div class="model-info">
        <div class="model-name">PneumoGPT Diagnostic</div>
        <div class="model-desc">
          <span class="lang-en">Medical Language Model · QLoRA fine-tuned Qwen2.5-7B</span>
          <span class="lang-vi">Mô hình ngôn ngữ y tế · QLoRA fine-tuned Qwen2.5-7B</span>
        </div>
      </div>
      <div class="model-tag">LLM-Med v5.2</div>
    </div>
  </div>

  <div class="sec-divider">
    <span><span class="lang-en">Upload Audio</span><span class="lang-vi">Tải lên âm thanh</span></span>
    <div class="sec-line"></div>
  </div>
  <div class="upload-zone" id="uploadZone">
    <input type="file" id="fileInput" accept=".wav,.mp3,.ogg,.flac,.m4a,.webm" onchange="handleFile(this.files[0])">
    <div class="upload-orb">
      <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    </div>
    <div class="upload-title">
      <span class="lang-en">Drag and drop or click to select file</span>
      <span class="lang-vi">Kéo thả hoặc bấm để chọn file</span>
    </div>
    <div class="upload-sub">
      <span class="lang-en">Supports digital stethoscope recordings · Max 100 MB</span>
      <span class="lang-vi">Hỗ trợ ghi âm stethoscope kỹ thuật số · Tối đa 100 MB</span>
    </div>
    <div class="fmt-row">
      <span class="fmt">.WAV</span><span class="fmt">.MP3</span><span class="fmt">.FLAC</span>
      <span class="fmt">.OGG</span><span class="fmt">.M4A</span><span class="fmt">.WEBM</span>
    </div>
  </div>

  <div class="file-preview" id="filePreview">
    <div class="file-header">
      <div class="file-icon">
        <svg viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
      </div>
      <div>
        <div class="file-name" id="fileName">—</div>
        <div class="file-meta" id="fileMeta">—</div>
      </div>
      <button class="file-rm" onclick="removeFile()">×</button>
    </div>
    <canvas id="waveCanvas"></canvas>
    <div class="audio-controls">
      <button class="play-btn" onclick="toggleAudio()">
        <svg id="playIcon" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      </button>
      <span class="audio-time" id="audioTime">0:00 / 0:00</span>
      <div class="audio-progress" id="audioProgress" onclick="seekAudio(event)">
        <div class="audio-progress-fill" id="progressFill"></div>
      </div>
    </div>
  </div>

  <div class="analyze-wrap">
    <button class="analyze-btn" id="analyzeBtn" disabled onclick="runAnalysis()">
      <span class="btn-inner">
        <div class="btn-pulse"></div>
        <span class="lang-en">Run AI Analysis</span>
        <span class="lang-vi">Chạy phân tích AI</span>
        <svg viewBox="0 0 24 24" style="width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
      </span>
    </button>
  </div>

  <div class="results-section" id="resultsSection">
    <div class="results-header">
      <h2><span class="lang-en">Analysis Results</span><span class="lang-vi">Kết quả phân tích</span></h2>
      <div class="results-line"></div>
      <div class="results-ts" id="resultsTs"></div>
    </div>

    <div class="results-grid" id="resultsGrid"></div>
    <div id="top3DiseaseSection" style="display:none"></div>

    <div class="consensus" id="consensusPanel">
      <div class="con-head">
        <div class="con-pulse"></div>
        <h3><span class="lang-en">Clinical Diagnosis</span><span class="lang-vi">Chẩn đoán lâm sàng</span></h3>
        <div class="con-tag" id="llmSourceTag">LLM · QLORA</div>
      </div>
      <div class="con-body" id="consensusBody"></div>
    </div>

    <div class="qlora-panel" id="qloraPanel" style="display:none">
      <div class="qlora-head" onclick="toggleQloraPanel()">
        <div class="qlora-pulse"></div>
        <h3><span class="lang-en">QLoRA · 6-Step Per-Cycle Analysis (v5.2)</span><span class="lang-vi">QLoRA · Phân tích 6-Step Per-Cycle (v5.2)</span></h3>
        <div class="qlora-header-badges" id="qloraHeaderBadges"></div>
        <button class="qlora-toggle-btn" id="qloraToggleBtn" onclick="event.stopPropagation();toggleQloraPanel()">
          <span class="qarr">▾</span>
          <span id="qloraToggleLabel"><span class="lang-en">View Details</span><span class="lang-vi">Xem chi tiết</span></span>
        </button>
      </div>
      <div class="qlora-body-wrap" id="qloraBodyWrap">
        <div class="qlora-body" id="qloraBody"></div>
      </div>
    </div>

    <div class="card-section" id="cyclesSection" style="display:none">
      <div class="card-head">
        <div class="con-pulse"></div>
        <h3><span class="lang-en">Per-Cycle Details</span><span class="lang-vi">Chi tiết từng Cycle</span></h3>
        <div class="con-tag" id="cyclesTag">0 CYCLES</div>
      </div>
      <div class="cycles-body" id="cyclesBody"></div>
    </div>

    <div class="card-section" id="gradcamSection" style="display:none">
      <div class="card-head">
        <div class="con-pulse"></div>
        <h3>Grad-CAM Activation Maps</h3>
        <div class="con-tag">VISUAL XAI</div>
      </div>
      <div class="gradcam-body" id="gradcamBody"></div>
    </div>
  </div>
</div>

<audio id="audioEl"></audio>

<script>
/* ═══ LANGUAGE HELPERS ═══ */
function toggleLanguage() {
  const root = document.documentElement;
  const curr = root.getAttribute('lang');
  root.setAttribute('lang', curr === 'en' ? 'vi' : 'en');

  // Tự động chuyển đổi text bên trong button Expand / Collapse
  const wrap = document.getElementById('qloraBodyWrap');
  if (wrap && wrap.classList.contains('open')) {
    document.getElementById('qloraToggleLabel').innerHTML = t('Collapse', 'Thu gọn');
  }
}

function getLangText(en, vi) {
  return document.documentElement.getAttribute('lang') === 'vi' ? vi : en;
}

const t = (en, vi) => `<span class="lang-en">${en}</span><span class="lang-vi">${vi}</span>`;

/* ═══ AURORA ═══ */
(function(){
  const c=document.getElementById('auroraCanvas');
  const ctx=c.getContext('2d');
  let W,H,tm=0;
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  resize();window.addEventListener('resize',resize);
  function layer(baseY,c1,c2,speed,amp){
    ctx.beginPath();
    const b=Math.sin(tm*1.2)*.5+.5;
    for(let x=0;x<=W;x++){
      const nx=x/W;
      const w=Math.sin(nx*8+tm*speed)*(38*amp)+Math.sin(nx*4+tm*speed*.6)*(22*amp);
      const y=baseY+w*(.6+b*.8)+Math.sin(tm*.8+nx*5)*9*b;
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.lineTo(W,H);ctx.lineTo(0,H);ctx.closePath();
    const g=ctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0,c1);g.addColorStop(1,c2);
    ctx.fillStyle=g;ctx.fill();
  }
  function loop(){
    tm+=.016;
    ctx.fillStyle='rgba(7,11,18,0.18)';ctx.fillRect(0,0,W,H);
    const b=Math.sin(tm*1.2)*.5+.5;
    const glow=ctx.createRadialGradient(W/2,H*.7,0,W/2,H*.7,H*.8);
    glow.addColorStop(0,`rgba(126,216,255,${.07+b*.07})`);glow.addColorStop(1,'transparent');
    ctx.fillStyle=glow;ctx.fillRect(0,0,W,H);
    layer(H*.65,'rgba(126,216,255,0.06)','transparent',.8,.7);
    layer(H*.7,'rgba(168,237,190,0.1)','transparent',1,1);
    layer(H*.76,'rgba(232,197,255,0.09)','transparent',1.3,1.2);
    requestAnimationFrame(loop);
  }
  loop();
})();

/* ═══ THREE.JS LUNG ═══ */
(function(){
  const canvas=document.getElementById('lungCanvas');
  const cont=canvas.parentElement;
  const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  const scene=new THREE.Scene();
  const camera=new THREE.PerspectiveCamera(44,1,.1,100);
  camera.position.set(0,0,6.2);
  function resize(){
    const w=cont.clientWidth,h=cont.clientHeight;
    renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();
  }
  resize();window.addEventListener('resize',resize);
  scene.add(new THREE.AmbientLight(0xffffff,.5));
  const kl=new THREE.DirectionalLight(0x7ed8ff,3.5);kl.position.set(3,3,4);scene.add(kl);
  const rl=new THREE.DirectionalLight(0xa8edbe,1.8);rl.position.set(-3,-1,-2);scene.add(rl);
  const pt=new THREE.PointLight(0xff4444,4,9);pt.position.set(0,0,3);scene.add(pt);
  const pt2=new THREE.PointLight(0x40c4ff,5,8);pt2.position.set(0,1,3);scene.add(pt2);
  const lungGroup=new THREE.Group();scene.add(lungGroup);
  lungGroup.scale.set(0.75,0.75,0.75);
  const lungMat=new THREE.MeshStandardMaterial({color:0xcc1a1a,roughness:.45,metalness:.08,transparent:true,opacity:.93,emissive:0xcc0000,emissiveIntensity:0.35});
  const wireMat=new THREE.MeshBasicMaterial({color:0x40c4ff,wireframe:true,transparent:true,opacity:.055});
  function makeLobe(xOff,scale){
    const g=new THREE.SphereGeometry(1,80,80);
    const pos=g.attributes.position;
    for(let i=0;i<pos.count;i++){
      let x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);
      const ny=y;
      const isInnerSide=(xOff>0)?(x<0):(x>0);
      const taper=1-0.3*((ny+1)/2);
      const innerFactor=isInnerSide?4.0*Math.exp(-Math.abs(x)*2.5):1;
      const topToBottom=(1-ny)/2;
      const outerBulge=!isInnerSide?1+(0.4+4*topToBottom)*Math.exp(-Math.abs(x)*1.1):1;
      const bottomFactor=Math.max(0,-ny);
      const outerRatio=isInnerSide?0:Math.abs(x);
      const bottomTilt=-0.85*bottomFactor*outerRatio;
      const bottomConcave=1.2*Math.pow(bottomFactor,2.0)*Math.cos(outerRatio*Math.PI*0.1);
      const bottomBoost=1+0.7*bottomConcave;
      const frontFactor=1-0.12*Math.abs(z);
      const curvature=0.9+0.08*Math.sin(ny*Math.PI);
      const noiseBase=Math.sin(6*x)*Math.sin(6*y)*Math.sin(6*z);
      const noiseOuter=1+0.018*noiseBase;
      const noiseInner=1+0.07*noiseBase+0.04*Math.sin(10*x)*Math.sin(10*y);
      const noiseBottom=1+0.08*noiseBase+0.05*Math.sin(9*x)*Math.sin(9*z)*bottomFactor;
      const isBottomSurface=bottomFactor>0.6;
      const noiseBottomSurface=1+0.08*noiseBase+0.09*Math.sin(10*x)*Math.sin(10*z);
      const noise=isInnerSide?noiseInner:isBottomSurface?noiseBottomSurface:(bottomFactor>0.2?noiseBottom:noiseOuter);
      x=x*taper*innerFactor*outerBulge*curvature*noise;
      y=y*(1+0.3*Math.abs(ny))+bottomTilt*bottomBoost;
      z=z*taper*frontFactor*bottomBoost*noise;
      pos.setXYZ(i,x,y,z);
    }
    g.computeVertexNormals();
    const lg=new THREE.Group();
    const m=new THREE.Mesh(g,lungMat.clone());
    lg.add(m,new THREE.Mesh(g,wireMat));
    lg.scale.set(scale.x,scale.y,scale.z);
    lg.position.set(xOff,-.1,0);
    return{group:lg,mesh:m};
  }
  const lobeL=makeLobe(-.58,{x:.65,y:1.25,z:.6});
  const lobeR=makeLobe(.62,{x:.85,y:1.2,z:.65});
  lobeL.group.scale.x*=.9;lobeL.group.position.x-=.1;
  lungGroup.add(lobeL.group,lobeR.group);
  const bMat=new THREE.MeshStandardMaterial({color:0x9be7ff,emissive:0x40c4ff,emissiveIntensity:1.8,metalness:.3,roughness:.3});
  function branch(par,s,e,r){
    const d=new THREE.Vector3().subVectors(e,s),len=d.length();
    const g=new THREE.CylinderGeometry(r*.85,r,len,8);
    const m=new THREE.Mesh(g,bMat);
    m.position.copy(new THREE.Vector3().addVectors(s,e).multiplyScalar(.5));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());
    par.add(m);
  }
  const br=new THREE.Group();
  branch(br,new THREE.Vector3(0,1.5,.2),new THREE.Vector3(0,.6,.2),.09);
  branch(br,new THREE.Vector3(0,.6,.2),new THREE.Vector3(-.85,.1,.1),.07);
  branch(br,new THREE.Vector3(0,.6,.2),new THREE.Vector3(.85,.1,.1),.07);
  branch(br,new THREE.Vector3(-.85,.1,.1),new THREE.Vector3(-1.1,-.5,.05),.05);
  branch(br,new THREE.Vector3(-.85,.1,.1),new THREE.Vector3(-.55,-.55,.05),.04);
  branch(br,new THREE.Vector3(.85,.1,.1),new THREE.Vector3(1.1,-.5,.05),.05);
  branch(br,new THREE.Vector3(.85,.1,.1),new THREE.Vector3(.6,-.6,.05),.04);
  lungGroup.add(br);
  const artMat=new THREE.MeshStandardMaterial({color:0xdd0000,emissive:0xff1100,emissiveIntensity:2.2,roughness:.35,metalness:.12,transparent:true,opacity:.88});
  const veinMat=new THREE.MeshStandardMaterial({color:0x990000,emissive:0xcc0000,emissiveIntensity:1.4,roughness:.45,metalness:.1,transparent:true,opacity:.72});
  const capMat=new THREE.MeshStandardMaterial({color:0xff3333,emissive:0xff2200,emissiveIntensity:1.8,roughness:.3,metalness:.05,transparent:true,opacity:.6});
  const vesselGroup=new THREE.Group();
  lungGroup.add(vesselGroup);
  function addFiberTree(parent,start,dir,length,radius,depth,isArtery,side){
    if(depth<=0||length<0.03||radius<0.004)return;
    const end=new THREE.Vector3().addVectors(start,new THREE.Vector3().copy(dir).normalize().multiplyScalar(length));
    const mat=isArtery?(radius>0.04?artMat:capMat):veinMat;
    const geo=new THREE.CylinderGeometry(radius*.82,radius,length,6);
    const m=new THREE.Mesh(geo,mat);
    const mid=new THREE.Vector3().addVectors(start,end).multiplyScalar(.5);
    m.position.copy(mid);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.clone().normalize());
    parent.add(m);
    const children=depth>3?3:depth>1?2:1;
    for(let i=0;i<children;i++){
      const angle=(i/children)*Math.PI*1.7+Math.random()*.5;
      const tilt=0.3+Math.random()*.55;
      const perp=new THREE.Vector3(Math.cos(angle)*(side<0?-1:1)*.7+Math.random()*.2,-0.15+Math.random()*.3,Math.sin(angle)*.6+Math.random()*.25);
      const newDir=new THREE.Vector3().addVectors(dir.clone().normalize().multiplyScalar(1-tilt),perp.normalize().multiplyScalar(tilt)).normalize();
      addFiberTree(parent,end,newDir,length*(0.55+Math.random()*.25),radius*(0.52+Math.random()*.18),depth-1,isArtery,side);
    }
  }
  const artRoots=[
    {s:new THREE.Vector3(-.15,.6,.15),d:new THREE.Vector3(-1,.2,-.1),r:.065,art:true,side:-1},
    {s:new THREE.Vector3(-.1,.2,.1),d:new THREE.Vector3(-.9,-.3,.0),r:.058,art:true,side:-1},
    {s:new THREE.Vector3(.15,.6,.15),d:new THREE.Vector3(1,.2,-.1),r:.07,art:true,side:1},
    {s:new THREE.Vector3(.1,.2,.1),d:new THREE.Vector3(.9,-.3,.0),r:.062,art:true,side:1},
    {s:new THREE.Vector3(-.3,.55,.1),d:new THREE.Vector3(-1.1,.1,-.15),r:.055,art:false,side:-1},
    {s:new THREE.Vector3(.3,.55,.1),d:new THREE.Vector3(1.1,.1,-.15),r:.058,art:false,side:1},
  ];
  artRoots.forEach(a=>{addFiberTree(vesselGroup,a.s,a.d,.55,a.r,6,a.art,a.side);});
  const trunkMat=new THREE.MeshStandardMaterial({color:0xee0000,emissive:0xff2200,emissiveIntensity:3,roughness:.3,metalness:.15});
  function trunk(s,e,r){
    const d=new THREE.Vector3().subVectors(e,s),len=d.length();
    const g=new THREE.CylinderGeometry(r*.9,r,len,10);
    const m=new THREE.Mesh(g,trunkMat);
    m.position.copy(new THREE.Vector3().addVectors(s,e).multiplyScalar(.5));
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());
    vesselGroup.add(m);
  }
  trunk(new THREE.Vector3(0,1.3,-.05),new THREE.Vector3(-.18,.55,.1),.072);
  trunk(new THREE.Vector3(0,1.3,-.05),new THREE.Vector3(.18,.55,.1),.078);
  const alvM=new THREE.MeshStandardMaterial({color:0xa8edbe,emissive:0x40c4ff,emissiveIntensity:2.2,transparent:true,opacity:.65});
  const alveoli=[],aGeo=new THREE.SphereGeometry(.032,6,6);
  for(let i=0;i<60;i++){
    const m=new THREE.Mesh(aGeo,alvM.clone());
    const side=i<30?-1:1,r=.35+Math.random()*.6,theta=Math.random()*Math.PI,phi=Math.random()*Math.PI*2;
    m.position.set(side*(.9+r*Math.sin(theta)*Math.cos(phi)*.7),-.1+r*Math.cos(theta)*.95,r*Math.sin(theta)*Math.sin(phi)*.5);
    m.userData={phase:Math.random()*Math.PI*2,speed:.7+Math.random()*1.3};
    scene.add(m);alveoli.push(m);
  }
  let rotX=0,rotY=0,targetRotX=0,targetRotY=0,t2=0;
  cont.addEventListener('mousemove',e=>{
    const rect=cont.getBoundingClientRect();
    targetRotY=((e.clientX-rect.left)/rect.width-.5)*0.6;
    targetRotX=-((e.clientY-rect.top)/rect.height-.5)*0.5;
  });
  function lerp(a,b,t_val){return a+(b-a)*t_val;}
  function animate(){
    requestAnimationFrame(animate);t2+=.016;
    const breath=1+Math.sin(t2*1.2)*.075;
    lobeL.group.scale.set(breath*.78,breath*1.1,breath*.68);
    lobeR.group.scale.set(breath*.88,breath*1.1,breath*.68);
    lobeL.mesh.material.emissiveIntensity=.3+Math.sin(t2*1.2)*.18;
    lobeR.mesh.material.emissiveIntensity=.3+Math.sin(t2*1.2)*.18;
    lungGroup.position.y=Math.sin(t2*1.2)*.12;
    const hb=Math.abs(Math.sin(t2*1.4))*.6;
    artMat.emissiveIntensity=1.8+hb*2.0;
    trunkMat.emissiveIntensity=2.5+hb*2.5;
    capMat.emissiveIntensity=1.2+hb*1.8;
    pt.intensity=3+hb*4;
    alveoli.forEach(a=>{
      const s=.65+Math.sin(t2*a.userData.speed+a.userData.phase)*.52;
      a.scale.setScalar(Math.max(.1,s));
      a.material.opacity=.15+Math.max(0,Math.sin(t2*a.userData.speed+a.userData.phase))*.55;
    });
    rotY=lerp(rotY,targetRotY,.08);
    rotX=lerp(rotX,targetRotX,.08);
    rotX=Math.max(-.5,Math.min(.5,rotX));
    lungGroup.rotation.y=rotY+Math.sin(t2*.3)*.05;
    lungGroup.rotation.x=rotX+Math.sin(t2*.2)*.03;
    pt2.position.x=Math.sin(t2*.5)*2.2;
    pt2.intensity=4+Math.sin(t2*1.5)*1.8;
    renderer.render(scene,camera);
  }
  animate();
})();

/* ═══ STATUS ═══ */
async function checkStatus(){
  try{
    const r=await fetch('/health',{signal:AbortSignal.timeout(4000)});
    if(!r.ok)throw new Error();
    const d=await r.json();
    const pill=document.getElementById('modelStatusPill');
    if(d.model_loaded){pill.textContent=`Model: ✓ ${(d.device||'cpu').toUpperCase()} · v5.2`;pill.style.color='var(--green)';}
    else pill.textContent='Model: lazy';
  }catch(e){document.getElementById('modelStatusPill').textContent='Model: offline';}
}
checkStatus();

/* ═══ AUDIO & FILE ═══ */
let currentFile=null,isPlaying=false;
const zone=document.getElementById('uploadZone');
zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('drag-over');});
zone.addEventListener('dragleave',e=>{if(!zone.contains(e.relatedTarget))zone.classList.remove('drag-over');});
zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('drag-over');if(e.dataTransfer.files[0])handleFile(e.dataTransfer.files[0]);});

function handleFile(file){
  if(!file)return;
  if(!file.name.match(/\.(wav|mp3|ogg|flac|m4a|webm)$/i)&&!file.type.startsWith('audio')){
    showToast(getLangText('Please select a valid audio file', 'Vui lòng chọn file âm thanh hợp lệ'));
    return;
  }
  currentFile=file;
  document.getElementById('fileName').textContent=file.name;
  document.getElementById('fileMeta').textContent=`${(file.size/1024).toFixed(1)} KB · ${file.type||'audio'}`;
  document.getElementById('filePreview').classList.add('show');
  document.getElementById('analyzeBtn').disabled=false;
  const url=URL.createObjectURL(file);
  const audio=document.getElementById('audioEl');
  audio.src=url;
  audio.addEventListener('loadedmetadata',updateTime);
  audio.addEventListener('timeupdate',updateProgress);
  audio.addEventListener('ended',()=>{isPlaying=false;document.getElementById('playIcon').innerHTML='<polygon points="5 3 19 12 5 21 5 3"/>';});
  drawWaveform(file);
}

function removeFile(){
  currentFile=null;
  document.getElementById('filePreview').classList.remove('show');
  document.getElementById('analyzeBtn').disabled=true;
  document.getElementById('resultsSection').classList.remove('show');
  document.getElementById('audioEl').src='';
  document.getElementById('fileInput').value='';
}

function toggleAudio(){
  const a=document.getElementById('audioEl');
  if(!a.src)return;
  if(isPlaying){a.pause();isPlaying=false;document.getElementById('playIcon').innerHTML='<polygon points="5 3 19 12 5 21 5 3"/>';}
  else{a.play();isPlaying=true;document.getElementById('playIcon').innerHTML='<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';}
}

function updateTime(){
  const a=document.getElementById('audioEl');
  const fmt=s=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
  document.getElementById('audioTime').textContent=`${fmt(a.currentTime)} / ${fmt(a.duration||0)}`;
}
function updateProgress(){
  const a=document.getElementById('audioEl');updateTime();
  document.getElementById('progressFill').style.width=(a.duration?(a.currentTime/a.duration*100):0)+'%';
}
function seekAudio(e){
  const a=document.getElementById('audioEl');if(!a.duration)return;
  const rect=e.currentTarget.getBoundingClientRect();
  a.currentTime=((e.clientX-rect.left)/rect.width)*a.duration;
}

async function drawWaveform(file){
  const canvas=document.getElementById('waveCanvas');
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*window.devicePixelRatio;
  canvas.height=64*window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio,window.devicePixelRatio);
  const W=canvas.offsetWidth,H=64;
  try{
    const ac=new AudioContext();
    const buf=await ac.decodeAudioData(await file.arrayBuffer());
    const data=buf.getChannelData(0);
    const step=Math.floor(data.length/W);
    const g=ctx.createLinearGradient(0,0,W,0);
    g.addColorStop(0,'rgba(126,216,255,0.4)');
    g.addColorStop(.5,'rgba(168,237,190,0.75)');
    g.addColorStop(1,'rgba(126,216,255,0.4)');
    ctx.fillStyle=g;
    for(let i=0;i<W;i++){
      let max=0;
      for(let j=0;j<step;j++)max=Math.max(max,Math.abs(data[i*step+j]));
      const h=Math.max(2,max*H*.82);
      ctx.fillRect(i,(H-h)/2,1.2,h);
    }
    await ac.close();
  }catch(e){
    ctx.strokeStyle='rgba(126,216,255,0.4)';ctx.lineWidth=1.5;ctx.beginPath();
    for(let i=0;i<W;i++){const y=H/2+Math.sin(i*.1)*16*(.5+Math.random()*.5);i===0?ctx.moveTo(i,y):ctx.lineTo(i,y);}
    ctx.stroke();
  }
}

/* ═══ API CALL ═══ */
async function runAnalysis(){
  if(!currentFile)return;
  const btn=document.getElementById('analyzeBtn');
  btn.disabled=true;

  showLoading(
    t('Analyzing lung sounds...', 'Đang phân tích âm thanh phổi…'),
    t('Sending file to AI server...', 'Gửi file tới server AI…')
  );

  try{
    const formData=new FormData();
    formData.append('file',currentFile,currentFile.name);

    document.getElementById('loadingSub').innerHTML = t('ResNet18 is processing cycles...', 'ResNet18 đang xử lý cycles…');
    const response=await fetch('/analyze',{method:'POST',body:formData});

    if(!response.ok){let msg=`HTTP ${response.status}`;try{const j=await response.json();msg=j.detail||msg;}catch(_){}throw new Error(msg);}

    document.getElementById('loadingSub').innerHTML = t('QLoRA sequential per-cycle is analyzing...', 'QLoRA sequential per-cycle đang phân tích…');
    const data=await response.json();
    hideLoading();
    renderResults(data.result);
  }catch(err){
    hideLoading();
    showToast(getLangText('Analysis error: ', 'Lỗi phân tích: ') + err.message);
    btn.disabled=false;
  }
}

/* ═══ HELPERS ═══ */
function escapeHtml(s){
  return String(s||'')
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

function animateBars(container){
  (container||document).querySelectorAll('[data-w]').forEach(el=>{el.style.width=el.dataset.w;});
}

/* ═══ QLORA PANEL TOGGLE ═══ */
let _qloraOpen=false;

function toggleQloraPanel(){
  const wrap=document.getElementById('qloraBodyWrap');
  const btn=document.getElementById('qloraToggleBtn');
  const label=document.getElementById('qloraToggleLabel');
  _qloraOpen=!_qloraOpen;
  if(_qloraOpen){
    wrap.classList.add('open');
    wrap.style.maxHeight=wrap.scrollHeight+'px';
    btn.classList.add('open');
    label.innerHTML=t('Collapse', 'Thu gọn');
    setTimeout(()=>animateBars(wrap),80);
  }else{
    wrap.style.maxHeight='0';
    wrap.classList.remove('open');
    btn.classList.remove('open');
    label.innerHTML=t('View Details', 'Xem chi tiết');
  }
}

/* Toggle từng step card */
function toggleStepCard(prefix,idx){
  const card=document.getElementById(`${prefix}-step-card-${idx}`);
  const bodyWrap=document.getElementById(`${prefix}-step-body-wrap-${idx}`);
  if(!card||!bodyWrap)return;
  const isOpen=card.classList.contains('expanded');
  if(!isOpen){
    card.classList.add('expanded');
    bodyWrap.classList.add('open');
    bodyWrap.style.maxHeight=bodyWrap.scrollHeight+'px';
  }else{
    card.classList.remove('expanded');
    bodyWrap.style.maxHeight='0';
    bodyWrap.classList.remove('open');
  }
}

/* Switch cycle tab */
function switchCycleTab(tabGroup,idx){
  document.querySelectorAll(`.qlora-cycle-tab[data-group="${tabGroup}"]`).forEach(t_el=>t_el.classList.remove('active'));
  document.querySelectorAll(`.qlora-cycle-panel[data-group="${tabGroup}"]`).forEach(p=>p.classList.remove('active'));
  const tab=document.querySelector(`.qlora-cycle-tab[data-group="${tabGroup}"][data-idx="${idx}"]`);
  const panel=document.querySelector(`.qlora-cycle-panel[data-group="${tabGroup}"][data-idx="${idx}"]`);
  if(tab)tab.classList.add('active');
  if(panel){
    panel.classList.add('active');
    setTimeout(()=>animateBars(panel),60);
  }
  const wrap=document.getElementById('qloraBodyWrap');
  if(wrap&&_qloraOpen){
    wrap.style.maxHeight='none';
  }
}

/* ═══ TOP-3 CYCLES SECTION ═══ */
function buildTop3DiseaseSection(top3Cycles){
  if(!top3Cycles||top3Cycles.length===0)return'';
  const rankClass=['r1','r2','r3'];
  const rankLabel=['#1','#2','#3'];

  const cycleCards=top3Cycles.map((cyc,i)=>{
    const rank=cyc.rank||(i+1);
    const rankIdx=Math.min(rank-1,2);
    const isAbnormal=cyc.event!=='Normal';
    const evChipCls=isAbnormal?'abnormal':'normal';
    const disPred=cyc.disease_pred||'—';
    const altDis=cyc.alt_disease||'—';
    const camD=cyc.cam_disease||{};
    const camDAlt=cyc.cam_disease_alt||{};
    const camDiff=cyc.cam_diff||{};
    const predPeak=Math.round((camD.peak||0)*100);
    const altPeak=Math.round((camDAlt.peak||0)*100);
    const evConf=((cyc.event_confidence||0)*100).toFixed(1);

    const imgHTML=cyc.gradcam_image_url
      ?`<img src="${cyc.gradcam_image_url}" alt="GradCAM Cycle ${cyc.cycle_index}" loading="lazy">`
      :`<div style="width:100%;height:80px;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.3);border-radius:8px;border:1px solid var(--border);font-family:var(--f-mono);font-size:10px;color:var(--text3)">No CAM</div>`;

    return `
      <div class="t3d-cycle-card ${rank===1?'rank-1':''}">
        <div class="t3d-cycle-header">
          <div class="t3d-cycle-rank-badge ${rankClass[rankIdx]}">${rankLabel[rankIdx]}</div>
          <div class="t3d-cycle-time">Cycle ${cyc.cycle_index} · ${cyc.start_sec}s–${cyc.end_sec}s</div>
          <span class="t3d-cycle-event-chip ${evChipCls}"><span class="t3d-dot"></span>${t(cyc.event, cyc.event_vi||cyc.event)}</span>
          <div class="t3d-cycle-ev-conf">ev_conf: ${evConf}%</div>
        </div>
        <div class="t3d-cycle-content">
          <div class="t3d-gradcam-img-wrap">
            ${imgHTML}
            <div class="t3d-gradcam-label">GradCAM · Dual-target</div>
          </div>
          <div class="t3d-metrics-col">
            <div class="t3d-disease-row">
              <div><div class="t3d-disease-badge-key">CNN Pred Disease</div><div class="t3d-disease-badge pred">${disPred}</div></div>
              <div><div class="t3d-disease-badge-key">Alt Disease</div><div class="t3d-disease-badge alt">${altDis}</div></div>
            </div>
            <div class="t3d-cam-bar-wrap">
              <div class="t3d-cam-bar-row">Disease CAM · Predicted (${disPred})<span>${predPeak}%</span></div>
              <div class="t3d-cam-bar"><div class="t3d-cam-fill" style="width:0%" data-w="${predPeak}%"></div></div>
            </div>
            <div class="t3d-cam-bar-wrap">
              <div class="t3d-cam-bar-row">Disease CAM · Alternative (${altDis})<span>${altPeak}%</span></div>
              <div class="t3d-cam-bar"><div class="t3d-cam-fill alt-fill" style="width:0%" data-w="${altPeak}%"></div></div>
            </div>
            <div class="t3d-cam-grid">
              <div class="t3d-cam-item"><div class="t3d-cam-key">Freq High · Pred</div><div class="t3d-cam-val green">${((camD.freq_high||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Freq Mid · Pred</div><div class="t3d-cam-val">${((camD.freq_mid||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Freq Low · Pred</div><div class="t3d-cam-val">${((camD.freq_low||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Freq High · Alt</div><div class="t3d-cam-val amber">${((camDAlt.freq_high||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Freq Mid · Alt</div><div class="t3d-cam-val amber">${((camDAlt.freq_mid||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Hot Ratio · Pred</div><div class="t3d-cam-val">${((camD.hot_ratio||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">CAM Peak · Pred</div><div class="t3d-cam-val green">${((camD.peak||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">CAM Peak · Alt</div><div class="t3d-cam-val amber">${((camDAlt.peak||0)*100).toFixed(1)}%</div></div>
              <div class="t3d-cam-item"><div class="t3d-cam-key">Entropy · CAM</div><div class="t3d-cam-val purple">${(camD.entropy||0).toFixed(3)}</div></div>
            </div>
            <div class="t3d-diff-row">
              <div class="t3d-diff-label">CAM Diff Signal</div>
              <div class="t3d-diff-chips">
                <span class="t3d-diff-chip">diff_peak <span>${(camDiff.diff_peak||0).toFixed(4)}</span></span>
                <span class="t3d-diff-chip">diff_min <span>${(camDiff.diff_min||0).toFixed(4)}</span></span>
                <span class="t3d-diff-chip">diff_abs_mean <span>${(camDiff.diff_abs_mean||0).toFixed(4)}</span></span>
                <span class="t3d-diff-chip">diff_entropy <span>${(camDiff.diff_entropy||0).toFixed(4)}</span></span>
              </div>
            </div>
          </div>
        </div>
      </div>`;
  }).join('');

  return `
    <div class="top3-disease-block">
      <div class="t3d-head">
        <div class="t3d-pulse"></div>
        <h3>Top-3 Cycles · Disease Branch Analysis</h3>
        <div class="t3d-tag">DISEASE BRANCH · v5.2</div>
      </div>
      <div class="t3d-strategy">${t('Prioritize cycles with abnormal events (Crackle/Wheeze/Both) → sort by disease branch peak CAM. Each cycle: dual-target GradCAM pred vs alt + contrast diff signal.', 'Ưu tiên cycles có event bất thường (Crackle/Wheeze/Both) → sort theo disease branch peak CAM. Mỗi cycle: dual-target GradCAM pred vs alt + contrast diff signal.')}</div>
      <div class="t3d-body">${cycleCards}</div>
    </div>`;
}

/* ═══ QLORA PANEL RENDER ═══ */
function renderQloraPanel(r){
  const panel=document.getElementById('qloraPanel');
  const body=document.getElementById('qloraBody');
  const headerBadges=document.getElementById('qloraHeaderBadges');

  const isQloRA     = r.llm_source==='qlora';
  const perCycle    = r.qlora_per_cycle||[];
  const aggDisCorr  = r.qlora_dis_correct!==undefined ? r.qlora_dis_correct : null;
  const altDx       = r.qloraAlternativeDiagnosis||null;
  const unc         = r.uncertainty||{};
  const signals     = r.qlora_input_signals||{};

  panel.style.display='block';

  let srcBadge=isQloRA
    ?`<span class="qlora-source-badge qsrc-qlora"><span class="qlora-dot"></span>QLoRA · Qwen2.5-7B · v5.2</span>`
    :`<span class="qlora-source-badge qsrc-fallback"><span class="qlora-dot"></span>Fallback</span>`;

  let verdictBadge='';
  if(isQloRA){
    const nCycles=perCycle.length;
    if(aggDisCorr===true)
      verdictBadge=`<span class="qlora-verdict-badge qverdict-correct"><span class="qlora-dot"></span>✓ ${t(`Majority Correct (${nCycles} cycles)`, `Majority Correct (${nCycles} cycles)`)}</span>`;
    else if(aggDisCorr===false)
      verdictBadge=`<span class="qlora-verdict-badge qverdict-incorrect"><span class="qlora-dot"></span>⚠ ${t(`Potential deviation detected (${nCycles} cycles)`, `Sai lệch detected (${nCycles} cycles)`)}</span>`;
    else
      verdictBadge=`<span class="qlora-verdict-badge qverdict-unknown"><span class="qlora-dot"></span>Inference — ${nCycles} cycles</span>`;
  }
  headerBadges.innerHTML=srcBadge+verdictBadge;

  _qloraOpen=false;
  const wrap=document.getElementById('qloraBodyWrap');
  wrap.classList.remove('open');
  wrap.style.maxHeight='0';
  document.getElementById('qloraToggleBtn').classList.remove('open');
  document.getElementById('qloraToggleLabel').innerHTML=t('View Details', 'Xem chi tiết');

  if(!isQloRA){
    body.innerHTML=`
      <div class="qlora-empty">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        ${t('QLoRA is unavailable at this time.<br>Clinical diagnosis used rule-based fallback.', 'QLoRA không khả dụng tại thời điểm này.<br>Chẩn đoán lâm sàng đã sử dụng rule-based fallback.')}
      </div>`;
    return;
  }

  let verdictCls,verdictIcon,verdictLabel,verdictDesc;
  const nCorrect=perCycle.filter(c=>c.dis_correct===true).length;
  const nIncorrect=perCycle.filter(c=>c.dis_correct===false).length;
  const nNull=perCycle.filter(c=>c.dis_correct===null||c.dis_correct===undefined).length;

  if(aggDisCorr===true){
    verdictCls='correct';
    verdictIcon=`<svg viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>`;
    verdictLabel=t('QLoRA Majority Vote: CNN prediction correct', 'QLoRA Majority Vote: Dự đoán CNN chính xác');
    verdictDesc=t(`${perCycle.length} cycles analyzed independently. Result: ${nCorrect} correct · ${nIncorrect} incorrect · ${nNull} inference. Majority vote confirms CNN prediction.`, `${perCycle.length} cycles được phân tích độc lập. Kết quả: ${nCorrect} correct · ${nIncorrect} incorrect · ${nNull} inference. Majority vote xác nhận dự đoán CNN.`);
  }else if(aggDisCorr===false){
    verdictCls='incorrect';
    verdictIcon=`<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    verdictLabel=t('QLoRA Majority Vote: Potential deviation detected', 'QLoRA Majority Vote: Phát hiện sai lệch tiềm năng');
    verdictDesc=t(`${perCycle.length} cycles analyzed independently: ${nCorrect} correct · ${nIncorrect} incorrect · ${nNull} inference. Majority vote suggests CNN prediction may be incorrect — see alternative diagnosis.`, `${perCycle.length} cycles phân tích độc lập: ${nCorrect} correct · ${nIncorrect} incorrect · ${nNull} inference. Majority vote gợi ý dự đoán CNN có thể chưa chính xác — xem gợi ý chẩn đoán thay thế.`);
  }else{
    verdictCls='unknown';
    verdictIcon=`<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><polyline points="12 8 12 12 14 14"/></svg>`;
    verdictLabel=t(`QLoRA Sequential: ${perCycle.length} independent cycles analyzed`, `QLoRA Sequential: ${perCycle.length} cycles phân tích độc lập`);
    verdictDesc=t(`Each of the ${perCycle.length} top cycles was fed into QLoRA independently. No ground truth found for comparison (inference mode). See Step 6 of each cycle for conclusions.`, `Mỗi trong ${perCycle.length} top cycles được đưa vào QLoRA độc lập. Không tìm thấy ground truth để so sánh (inference mode). Xem Step 6 từng cycle để biết kết luận.`);
  }

  const cnnDis   = r.cnn_pred_disease||'—';
  const cnnConf  = r.cnn_confidence||0;
  const cnnAlt   = r.cnn_alt_disease||'—';

  const finalLabels=perCycle.map(c=>c.final_label||cnnDis);
  const labelCount={};
  finalLabels.forEach(l=>{labelCount[l]=(labelCount[l]||0)+1;});
  const voteChips=Object.entries(labelCount).sort((a,b)=>b[1]-a[1]).map(([label,cnt])=>{
    const isWinner=(Object.entries(labelCount).sort((a,b)=>b[1]-a[1])[0][0]===label);
    return `<span class="qlora-vote-chip ${isWinner?'winner':''}">${label} (${cnt}/${perCycle.length})</span>`;
  }).join('');

  const voteSummaryHTML=`
    <div class="qlora-vote-summary">
      <div class="qlora-vote-row">
        <span class="qlora-vote-label">CNN Disease (không đổi)</span>
        <span class="qlora-vote-chip winner">${cnnDis} · ${cnnConf}%</span>
      </div>
      <div class="qlora-vote-row">
        <span class="qlora-vote-label">QLoRA Final Labels (majority vote)</span>
        <div class="qlora-vote-chips">${voteChips}</div>
      </div>
      <div style="font-family:var(--f-mono);font-size:10px;color:var(--text3);margin-top:6px;">
        CNN Alt Disease: <span style="color:var(--amber)">${cnnAlt}</span>
        &nbsp;·&nbsp; Cycles analyzed: <span style="color:var(--c1)">${perCycle.length}</span>
        &nbsp;·&nbsp; Mode: <span style="color:var(--c3)">Sequential Independent</span>
      </div>
    </div>`;

  const tabGroupId='qlora-main';
  const rankLabel=['#1','#2','#3'];

  if(perCycle.length===0){
    body.innerHTML=`
      <div class="qlora-verdict-banner ${verdictCls}">
        <div class="qlora-verdict-icon">${verdictIcon}</div>
        <div><div class="qlora-verdict-label">${verdictLabel}</div><div class="qlora-verdict-desc">${verdictDesc}</div></div>
      </div>
      <div class="qlora-empty">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        ${t('No per-cycle data from backend. Check qlora_per_cycle in response.', 'Không có dữ liệu per-cycle từ backend. Kiểm tra qlora_per_cycle trong response.')}
      </div>`;
    return;
  }

  const tabsHTML=perCycle.map((cyc,i)=>{
    const rank=cyc.cycle_rank||(i+1);
    const rankIdx=Math.min(rank-1,2);
    const dc=cyc.dis_correct;
    const dotCls=dc===true?'correct':dc===false?'incorrect':'unknown';
    const rankStr=rankLabel[rankIdx]||`#${rank}`;
    return `<div class="qlora-cycle-tab ${i===0?'active':''}" data-group="${tabGroupId}" data-idx="${i}" onclick="switchCycleTab('${tabGroupId}',${i})">
      <span class="tab-verdict-dot ${dotCls}"></span>
      Cycle ${cyc.cycle_index||'?'} · ${rankStr}
    </div>`;
  }).join('');

  const panelsHTML=perCycle.map((cyc,i)=>{
    const rank=cyc.cycle_rank||(i+1);
    const rankIdx=Math.min(rank-1,2);
    const steps=cyc.qlora_steps||{};
    const dc=cyc.dis_correct;
    const ec=cyc.ev_correct;
    const finalLabel=cyc.final_label||cnnDis;
    const gtDis=cyc.gt_disease||null;
    const parseOk=cyc.parse_ok||false;
    const prefix=`${tabGroupId}-c${i}`;

    const dcTxt=dc===true?'correct':dc===false?'incorrect':'inference';
    const dcCls=dc===true?'vc':dc===false?'vi':'vu';
    const ecTxt=ec===true?'correct':ec===false?'incorrect':'—';
    const ecCls=ec===true?'vc':ec===false?'vi':'vu';

    const cycleSummaryHTML=`
      <div class="qlora-cycle-summary">
        <span class="qlora-cycle-summary-item">Cycle <span>${cyc.cycle_index||'?'}</span></span>
        <span class="qlora-cycle-summary-item">Rank <span>${rankLabel[rankIdx]||rank}</span></span>
        <span class="qlora-cycle-summary-item">Time <span>${cyc.start_sec||0}s–${cyc.end_sec||0}s</span></span>
        <span class="qlora-cycle-summary-item">Event <span>${cyc.cycle_event||'?'}</span></span>
        <span class="qlora-cycle-summary-item ${dcCls}">dis_correct: ${dcTxt}</span>
        <span class="qlora-cycle-summary-item ${ecCls}">ev_correct: ${ecTxt}</span>
        <span class="qlora-cycle-summary-item">final_label: <span style="color:${finalLabel!==cnnDis?'var(--amber)':'var(--c2)'}">${finalLabel}</span></span>
        ${gtDis?`<span class="qlora-cycle-summary-item">gt_disease: <span style="color:var(--red)">${gtDis}</span></span>`:''}
        <span class="qlora-cycle-summary-item">parse_ok: <span style="color:${parseOk?'var(--green)':'var(--text3)'}">${parseOk?'✓':'—'}</span></span>
      </div>`;

    const stepDefs=[
      {key:'step1_event',     num:'S1', label:'STEP 1 — Event Branch Assessment'},
      {key:'step2_disease',   num:'S2', label:'STEP 2 — Disease Branch Assessment'},
      {key:'step3_retrieval', num:'S3', label:'STEP 3 — Soft Retrieval Signal'},
      {key:'step4_prototype', num:'S4', label:'STEP 4 — Prototype Signal'},
      {key:'step5_conflict',  num:'S5', label:'STEP 5 — Conflict Analysis'},
      {key:'step6_conclusion',num:'S6', label:'STEP 6 — Final Conclusion'},
    ];

    const stepCards=stepDefs.map((sd,si)=>{
      const content=steps[sd.key]||'';
      const hasContent=content.trim().length>0;
      const autoOpen=si===5&&hasContent;
      return `
        <div class="qlora-step-card ${hasContent?'has-content':''} ${autoOpen?'expanded':''}" id="${prefix}-step-card-${si}">
          <div class="qlora-step-head" onclick="toggleStepCard('${prefix}',${si})">
            <span class="qlora-step-num">${sd.num}</span>
            <span class="qlora-step-label">${sd.label}</span>
            <span class="qlora-step-status ${hasContent?'ok':'empty'}">${hasContent?'✓':t('No output','Không có output')}</span>
            <span class="qlora-step-arr">▾</span>
          </div>
          <div class="qlora-step-body-wrap ${autoOpen?'open':''}" id="${prefix}-step-body-wrap-${si}"
               style="${autoOpen?'max-height:2000px;opacity:1':''}">
            <div class="qlora-step-body ${!hasContent?'empty-body':''}">${hasContent?escapeHtml(content):t('QLoRA did not return content for this step during this inference.', 'QLoRA không trả về nội dung cho step này trong lần inference này.')}</div>
          </div>
        </div>`;
    }).join('');

    return `
      <div class="qlora-cycle-panel ${i===0?'active':''}" data-group="${tabGroupId}" data-idx="${i}">
        ${cycleSummaryHTML}
        <div class="qlora-section-title">${t(`6-Step Analysis — Cycle ${cyc.cycle_index||'?'} (independent)`, `Phân tích 6 Bước — Cycle ${cyc.cycle_index||'?'} (độc lập)`)}</div>
        <div style="font-family:var(--f-mono);font-size:10px;color:var(--text3);margin-bottom:10px;padding:8px 12px;background:rgba(198,148,231,.04);border:1px solid rgba(198,148,231,.1);border-radius:8px;">
          ${t(`QLoRA receives input specifically for Cycle ${cyc.cycle_index||'?'} (${cyc.start_sec||0}s–${cyc.end_sec||0}s) · event=${cyc.cycle_event||'?'} — without sharing context with other cycles.`, `QLoRA nhận input riêng của Cycle ${cyc.cycle_index||'?'} (${cyc.start_sec||0}s–${cyc.end_sec||0}s) · event=${cyc.cycle_event||'?'} — không share context với các cycles khác.`)}
        </div>
        <div class="qlora-steps-grid">${stepCards}</div>
      </div>`;
  }).join('');

  const softRet=signals.soft_retrieval||{};
  const evUncD=unc.event||{};
  const disUncD=unc.disease||{};
  const cyclesUsed=(signals.top3_cycles_used||[]).join(', ')||'—';

  const avgSimEntries=Object.entries(softRet.avg_sim||{});
  const avgSimStr=avgSimEntries.length>0?avgSimEntries.map(([k,v])=>`${k}: ${v}`).join(' | '):'—';
  const evProbEntries=Object.entries((unc.event||{}).probs||{});
  const evProbStr=evProbEntries.length>0?evProbEntries.map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' | '):'—';
  const disProbEntries=Object.entries((unc.disease||{}).probs||{});
  const disProbStr=disProbEntries.length>0?disProbEntries.map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' | '):'—';

  const inputSignalsHTML=`
    <div>
      <div class="qlora-section-title">Patient-Level Signals (dùng chung cho cả 3 QLoRA calls)</div>
      <div class="qlora-signals-grid">
        <div class="qlora-signal-card">
          <div class="qlora-signal-key">Cycles gửi vào QLoRA</div>
          <div class="qlora-signal-val green">${cyclesUsed}</div>
        </div>
        <div class="qlora-signal-card">
          <div class="qlora-signal-key">Soft Retrieval · avg_sim</div>
          <div class="qlora-signal-val">${avgSimStr}</div>
        </div>
        <div class="qlora-signal-card">
          <div class="qlora-signal-key">sim_gap_top2 · Ambiguous</div>
          <div class="qlora-signal-val ${softRet.is_ambiguous?'amber':''}">${softRet.sim_gap_top2||0} · ${softRet.is_ambiguous?'YES':'NO'}</div>
        </div>
        <div class="qlora-signal-card">
          <div class="qlora-signal-key">Event Uncertainty</div>
          <div class="qlora-signal-val">H=${(evUncD.entropy||0).toFixed(3)} · M=${(evUncD.margin||0).toFixed(3)}</div>
        </div>
        <div class="qlora-signal-card">
          <div class="qlora-signal-key">Disease Uncertainty</div>
          <div class="qlora-signal-val">H=${(disUncD.entropy||0).toFixed(3)} · M=${(disUncD.margin||0).toFixed(3)}</div>
        </div>
        <div class="qlora-signal-card" style="grid-column:1/-1">
          <div class="qlora-signal-key">Event Probs (patient-level)</div>
          <div class="qlora-signal-val">${evProbStr}</div>
        </div>
        <div class="qlora-signal-card" style="grid-column:1/-1">
          <div class="qlora-signal-key">Disease Probs (patient-level)</div>
          <div class="qlora-signal-val">${disProbStr}</div>
        </div>
      </div>
    </div>`;

  let altAlertHTML='';
  if(altDx&&altDx.disease&&altDx.disease!==cnnDis){
    altAlertHTML=`
      <div>
        <div class="qlora-section-title">${t('Alternative Diagnosis Suggestion (Majority Vote Step 6)','Gợi ý Chẩn đoán Thay thế (Majority Vote Step 6)')}</div>
        <div class="qlora-alt-alert">
          <div class="qlora-alt-alert-icon">⚠️</div>
          <div class="qlora-alt-alert-body">
            <div class="qlora-alt-alert-title">${t('QLoRA Sequential suggests further review','QLoRA Sequential đề xuất xem xét thêm')}</div>
            <div class="qlora-alt-alert-name">${altDx.name||''} ${altDx.nameEN?`<span style="font-size:12px;font-weight:400;color:var(--text3)">(${altDx.nameEN})</span>`:''}</div>
            <div class="qlora-alt-alert-note">${altDx.note||t('Requires clinical correlation by a doctor.','Cần bác sĩ đối chiếu lâm sàng.')}</div>
          </div>
        </div>
      </div>`;
  }

  const uncHTML=`
    <div>
      <div class="qlora-section-title">Uncertainty · Patient Level</div>
      <div class="qlora-unc-row">
        <span class="qlora-unc-chip">Event Entropy <span>${(evUncD.entropy||0).toFixed(4)}</span></span>
        <span class="qlora-unc-chip">Event Margin <span>${(evUncD.margin||0).toFixed(4)}</span></span>
        <span class="qlora-unc-chip">Disease Entropy <span>${(disUncD.entropy||0).toFixed(4)}</span></span>
        <span class="qlora-unc-chip">Disease Margin <span>${(disUncD.margin||0).toFixed(4)}</span></span>
        <span class="qlora-unc-chip">Mode <span style="color:var(--c3)">Sequential v5.2</span></span>
      </div>
    </div>`;

  body.innerHTML=`
    <div class="qlora-verdict-banner ${verdictCls}">
      <div class="qlora-verdict-icon">${verdictIcon}</div>
      <div>
        <div class="qlora-verdict-label">${verdictLabel}</div>
        <div class="qlora-verdict-desc">${verdictDesc}</div>
      </div>
    </div>
    <div>
      <div class="qlora-section-title">${t('CNN vs QLoRA Summary — Majority Vote','Tổng hợp CNN vs QLoRA — Majority Vote')}</div>
      ${voteSummaryHTML}
    </div>
    <div>
      <div class="qlora-section-title">${t('Independent Per-Cycle Analysis · Click tab to view','Phân tích Độc lập từng Cycle · Click tab để xem')}</div>
      <div class="qlora-cycle-tabs">${tabsHTML}</div>
      ${panelsHTML}
    </div>
    ${inputSignalsHTML}
    ${altAlertHTML}
    ${uncHTML}
  `;
}

/* ═══ RENDER RESULTS ═══ */
function renderResults(r){
  document.getElementById('resultsTs').innerHTML=
    t((new Date(r.timestamp).toLocaleString('en-US')+` · ${r.processing_time_ms}ms · ${(r.model_device||'cpu').toUpperCase()} · ${r.model_version||''}`),
      (new Date(r.timestamp).toLocaleString('vi-VN')+` · ${r.processing_time_ms}ms · ${(r.model_device||'cpu').toUpperCase()} · ${r.model_version||''}`));

  document.getElementById('resultsGrid').innerHTML=buildEventPanel(r)+buildDiseasePanel(r);
  document.getElementById('llmSourceTag').textContent=r.llm_source==='qlora'?'LLM · QLoRA v5.2':'LLM · FALLBACK';

  const top3Sec=document.getElementById('top3DiseaseSection');
  if(r.top3_cycles&&r.top3_cycles.length>0){
    top3Sec.innerHTML=buildTop3DiseaseSection(r.top3_cycles);
    top3Sec.style.display='block';
  }else top3Sec.style.display='none';

  const pd=r.primaryDiagnosis||{};
  const recs=r.recommendations||[];
  const sevCls={high:'sev-high',medium:'sev-med',low:'sev-low'}[pd.severity]||'sev-med';
  const sevLbl={high:t('⚠ High Risk','⚠ Nguy cơ cao'),medium:t('◐ Medium','◐ Trung bình'),low:t('✓ Mild','✓ Nhẹ')}[pd.severity]||'—';

  document.getElementById('consensusBody').innerHTML=`
    <div class="con-grid">
      <div>
        <div class="con-dx-label">${t(`Disease Diagnosis · CNN ${r.cnn_confidence||0}% confidence`,`Chẩn đoán bệnh · CNN ${r.cnn_confidence||0}% confidence`)}</div>
        <div class="con-dx-name">${pd.name||'—'}</div>
        <div class="con-dx-en">${pd.nameEN||''}</div>
        <span class="sev-badge ${sevCls}">${sevLbl}</span>
        <div class="con-note" style="margin-top:18px">${r.clinicalNote||t('No clinical notes.','Không có ghi chú lâm sàng.')}</div>
        <div class="rec-title">${t('Clinical Recommendations','Khuyến nghị lâm sàng')}</div>
        <div class="rec-list">
          ${recs.map(rec=>`<div class="rec-item"><div class="rec-bullet"><svg viewBox="0 0 24 24"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg></div><span>${rec}</span></div>`).join('')}
        </div>
      </div>
      <div class="con-stats">
        <div><div class="con-stat-num">${pd.probability||0}%</div><div class="con-stat-label">${t('Disease Probability','Xác suất bệnh')}</div></div>
        <div class="con-stat-div"></div>
        <div><div class="con-stat-num">${r.cnn_confidence||0}%</div><div class="con-stat-label">CNN Confidence</div></div>
        <div class="con-stat-div"></div>
        <div><div class="con-stat-num">${r.totalCycles||0}</div><div class="con-stat-label">Total Cycles</div></div>
        <div class="con-stat-div"></div>
        <div><div class="con-stat-num">${r.top3_cycles?r.top3_cycles.length:0}</div><div class="con-stat-label">Top3 QLoRA</div></div>
        <div class="con-stat-div"></div>
        <div><div class="con-stat-num">${r.audioDuration?r.audioDuration.toFixed(1)+'s':'—'}</div><div class="con-stat-label">Duration</div></div>
      </div>
    </div>`;

  renderQloraPanel(r);

  const top3Idxs=new Set((r.top3_cycles||[]).map(c=>c.cycle_index));
  const cyclesSec=document.getElementById('cyclesSection');
  if(r.cycles&&r.cycles.length>0){
    document.getElementById('cyclesTag').textContent=`${r.totalCycles} CYCLES · top3 highlighted`;
    document.getElementById('cyclesBody').innerHTML=r.cycles.map(c=>{
      const isTop3=top3Idxs.has(c.cycle_index);
      const peakVal=c.cam_disease?.peak!==undefined?((c.cam_disease.peak||0)*100).toFixed(1)+'%':'—';
      return `<div class="cycle-card${isTop3?' top3-highlight':''}">
        ${c.gradcam_image_url?`<img class="cycle-img" src="${c.gradcam_image_url}" alt="Cycle ${c.cycle_index}" loading="lazy">`:`<div class="cycle-img" style="display:flex;align-items:center;justify-content:center;color:var(--text3);font-family:var(--f-mono);font-size:10px">No CAM</div>`}
        <div class="cycle-info">
          <div class="cycle-time">${c.start_sec}s–${c.end_sec}s</div>
          <div class="cycle-event"><span class="sound-chip chip-${c.event_chip||'normal'}" style="padding:3px 9px;font-size:10px;margin:4px 0 0"><span class="chip-dot"></span>${t(c.event, c.event_vi||c.event)}</span></div>
          <div class="cycle-conf">ev_conf: ${(c.event_confidence*100).toFixed(1)}%</div>
          <div class="cycle-peak">dis_peak: ${peakVal}</div>
        </div>
      </div>`;
    }).join('');
    cyclesSec.style.display='block';
  }else cyclesSec.style.display='none';

  const gcSec=document.getElementById('gradcamSection');
  if(r.gradcam_images&&r.gradcam_images.length>0){
    document.getElementById('gradcamBody').innerHTML=r.gradcam_images.map((url,i)=>{
      const isTop3=top3Idxs.has(i+1);
      return `<div class="gradcam-img-wrap" style="${isTop3?'border-color:rgba(168,237,190,.45);box-shadow:0 0 12px rgba(168,237,190,.15)':''}">
        <img src="${url}" alt="GradCAM ${i+1}" loading="lazy" title="${isTop3?'★ TOP-3 ':''}Cycle ${i+1}">
      </div>`;
    }).join('');
    gcSec.style.display='block';
  }else gcSec.style.display='none';

  document.getElementById('resultsSection').classList.add('show');
  setTimeout(()=>document.getElementById('resultsSection').scrollIntoView({behavior:'smooth',block:'start'}),100);
  setTimeout(()=>animateBars(document),250);
  document.getElementById('analyzeBtn').disabled=false;
}

/* ═══ EVENT PANEL ═══ */
function buildEventPanel(r){
  const st=r.soundType||'normal';
  const conf=r.confidence||80;
  const cc=conf>=80?'var(--green)':conf>=60?'var(--amber)':'var(--red)';
  const cycles=r.cycles||[];
  const shown=cycles.slice(0,4);
  const extra=cycles.slice(4);

  const extraRows=extra.map(c=>`
    <div class="dd-item">
      <span class="dd-name">[${c.start_sec}s–${c.end_sec}s] ${c.event}</span>
      <div class="dd-track"><div class="dd-bar" style="width:0%" data-w="${(c.event_confidence*100).toFixed(0)}%"></div></div>
      <span class="dd-num">${(c.event_confidence*100).toFixed(0)}%</span>
    </div>`).join('');

  const moreBtn=extra.length>0?`
    <button class="dd-more-btn" id="moreCyclesBtn" onclick="toggleExtraCycles(this)">
      <span class="arr">▾</span><span>${t(`+ ${extra.length} other cycles`,`+ ${extra.length} cycles khác`)}</span>
    </button>
    <div class="dd-extra" id="extraCycles"><div class="dd-list">${extraRows}</div></div>`:'';

  return `<div class="result-panel p1">
    <div class="panel-head">
      <span class="p-badge">Event Model</span>
      <span class="p-name">ResNet18 + FPN · CNN</span>
      <span class="p-conf" style="color:${cc}">${conf}%</span>
    </div>
    <div class="panel-body">
      <div class="sound-chip chip-${st}"><span class="chip-dot"></span>${t(st, r.soundTypeVN||st)}</div>
      <div class="dx-box">
        <div class="dx-label">${t('Dominant Sound','Âm thanh chủ đạo')}</div>
        <div class="dx-name">${t(st, r.soundTypeVN||'—')}</div>
        <div class="dx-en">Dominant: ${r.dominantEvent||st}</div>
        <div class="prob-bar"><div class="prob-fill fill-amber" style="width:0%" data-w="${conf}%"></div></div>
        <div class="prob-meta"><span>Confidence</span><span>${conf}%</span></div>
      </div>
      <div class="dx-label" style="margin-bottom:8px">${t(`Analyzed Cycles (${cycles.length} total)`,`Cycles phân tích (${cycles.length} total)`)}</div>
      <div class="dd-list">
        ${shown.map(c=>`<div class="dd-item"><span class="dd-name">[${c.start_sec}s–${c.end_sec}s] ${c.event}</span><div class="dd-track"><div class="dd-bar" style="width:0%" data-w="${(c.event_confidence*100).toFixed(0)}%"></div></div><span class="dd-num">${(c.event_confidence*100).toFixed(0)}%</span></div>`).join('')}
      </div>
      ${moreBtn}
    </div>
  </div>`;
}

function toggleExtraCycles(btn){
  const extra=document.getElementById('extraCycles');
  const isOpen=extra.classList.contains('open');
  if(!isOpen){extra.classList.add('open');extra.style.maxHeight=extra.scrollHeight+'px';btn.classList.add('open');setTimeout(()=>extra.querySelectorAll('.dd-bar').forEach(el=>el.style.width=el.dataset.w),80);}
  else{extra.style.maxHeight='0';extra.classList.remove('open');btn.classList.remove('open');}
}

/* ═══ DISEASE PANEL ═══ */
function buildDiseasePanel(r){
  const pd=r.primaryDiagnosis||{};
  const prob=pd.probability||0;
  const sev=pd.severity||'medium';
  const sc={high:'sev-high',medium:'sev-med',low:'sev-low'}[sev];
  const sl={high:t('⚠ High Risk','⚠ Nguy cơ cao'),medium:t('◐ Medium','◐ Trung bình'),low:t('✓ Mild','✓ Nhẹ')}[sev];
  const fc={high:'fill-red',medium:'fill-amber',low:'fill-green'}[sev];
  const diffs=r.differentials||[];
  const top3=r.top3_cycles||[];
  const disUnc=(r.uncertainty||{}).disease||{};
  const signals=r.qlora_input_signals||{};
  const softRet=signals.soft_retrieval||{};
  const topCls=softRet.top_class||'—';
  const gap=softRet.sim_gap_top2||0;
  const isAmb=softRet.is_ambiguous;

  const perCycle=r.qlora_per_cycle||[];
  let qloraPerCycleMini='';
  if(perCycle.length>0&&r.llm_source==='qlora'){
    const voteRows=perCycle.map(c=>{
      const dc=c.dis_correct;
      const col=dc===true?'var(--green)':dc===false?'var(--red)':'var(--text3)';
      return `<div class="dd-item">
        <span class="dd-name" style="color:${col}">Cycle ${c.cycle_index} [rank#${c.cycle_rank}] · ${c.cycle_event}</span>
        <span class="dd-num" style="color:${col};min-width:60px">${dc===true?'✓ correct':dc===false?'✗ diff':'— inf'}</span>
        <span style="font-family:var(--f-mono);font-size:10px;color:var(--amber);flex-shrink:0">${c.final_label||r.cnn_pred_disease}</span>
      </div>`;
    }).join('');
    qloraPerCycleMini=`
      <div style="margin-top:12px;padding:10px 12px;background:rgba(198,148,231,.04);border:1px solid rgba(198,148,231,.15);border-radius:10px;">
        <div class="dx-label" style="margin-bottom:8px;color:var(--c3)">${t('QLoRA Sequential · Results of 3 independent cycles','QLoRA Sequential · Kết quả 3 cycles độc lập')}</div>
        <div class="dd-list">${voteRows}</div>
      </div>`;
  }

  const top3Mini=top3.length>0?`
    <div class="dx-label" style="margin-bottom:8px;margin-top:14px">Top-3 Cycles · Disease Branch Peak CAM</div>
    <div class="dd-list">
      ${top3.map(c=>{
        const disP=Math.round((c.cam_disease?.peak||0)*100);
        const rCl=c.rank===1?'color:var(--c2)':c.rank===2?'color:var(--c1)':'color:var(--text3)';
        return `<div class="dd-item"><span class="dd-name" style="${rCl}">[${c.start_sec}s–${c.end_sec}s] ${t(c.event, c.event_vi||c.event)} · ${c.disease_pred}</span><div class="dd-track"><div class="dd-bar" style="width:0%;background:linear-gradient(90deg,var(--c2),var(--c1))" data-w="${disP}%"></div></div><span class="dd-num" style="${rCl}">${disP}%</span></div>`;
      }).join('')}
    </div>`:'' ;

  const retMiniHTML=softRet.avg_sim?`
    <div style="margin-top:10px;padding:9px 11px;background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:9px;">
      <div class="dx-label" style="margin-bottom:6px">Soft Retrieval Signal</div>
      <div style="font-family:var(--f-mono);font-size:10px;color:var(--text2);line-height:1.8;">
        top_class: <span style="color:var(--c2)">${topCls}</span> · sim_gap: <span style="color:${gap<0.05?'var(--amber)':'var(--c1)'}">${gap}</span> · ambiguous: <span style="color:${isAmb?'var(--amber)':'var(--green)'}">${isAmb?'YES':'NO'}</span>
      </div>
    </div>`:'';

  return `<div class="result-panel p2">
    <div class="panel-head">
      <span class="p-badge">Disease Model</span>
      <span class="p-name">PatientAttention · Multi-task · v5.2</span>
      <span class="p-conf" style="color:var(--c2)">${prob}%</span>
    </div>
    <div class="panel-body">
      <div class="dx-box">
        <div class="dx-label">${t('Primary Diagnosis (CNN · unchanged by QLoRA)','Chẩn đoán chính (CNN · không thay đổi bởi QLoRA)')}</div>
        <div class="dx-name">${pd.name||'—'}</div>
        <div class="dx-en">${pd.nameEN||''}</div>
        <div class="prob-bar"><div class="prob-fill ${fc}" style="width:0%" data-w="${prob}%"></div></div>
        <div class="prob-meta"><span>${t('Probability','Xác suất')}</span><span>${prob}%</span></div>
      </div>
      ${diffs.length>0?`<div class="dx-label" style="margin-bottom:8px">${t('Differential Diagnosis','Chẩn đoán phân biệt')}</div><div class="dd-list">${diffs.map(d=>`<div class="dd-item"><span class="dd-name">${t(d.name, d.nameVI||d.name)}</span><div class="dd-track"><div class="dd-bar" style="width:0%" data-w="${d.probability}%"></div></div><span class="dd-num">${d.probability}%</span></div>`).join('')}</div>`:''}
      <span class="sev-badge ${sc}">${sl}</span>
      ${qloraPerCycleMini}
      ${top3Mini}
      ${retMiniHTML}
      ${disUnc.entropy!==undefined?`<div style="margin-top:10px;font-family:var(--f-mono);font-size:10px;color:var(--text3)">Unc: entropy=${(disUnc.entropy||0).toFixed(3)} · margin=${(disUnc.margin||0).toFixed(3)}</div>`:''}
    </div>
  </div>`;
}

/* ═══ UTILS ═══ */
function showLoading(txt, sub){
  document.getElementById('loadingText').innerHTML=txt;
  document.getElementById('loadingSub').innerHTML=sub;
  document.getElementById('loadingOverlay').classList.add('show');
}
function hideLoading(){document.getElementById('loadingOverlay').classList.remove('show');}
function showToast(msg,ok=false){
  const t_el=document.getElementById('toast');t_el.innerHTML=msg;t_el.className='toast show'+(ok?' ok':'');
  setTimeout(()=>t_el.classList.remove('show'),4500);
}
</script>
</body>
</html>
"""

# ═════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "PneumoAI Lung Sound API",
    description = (
        "DualBranch ResNet18 + CrossAttention + PatientAttn + "
        "QLoRA Sequential Per-Cycle Independent (Qwen2.5-7B). "
        "v5.2: Each Top-3 cycle is fed into QLoRA independently and sequentially, with no shared context."
    ),
    version  = "5.2.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def on_startup():
    import asyncio, concurrent.futures
    loop     = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def load_all():
        try:
            get_model();    log.info("✅ DualBranch CNN loaded")
        except Exception as exc:
            log.warning(f"CNN pre-load failed: {exc}")
        try:
            m, _ = get_qlora_model()
            if m is not None: log.info("✅ QLoRA LLM loaded")
            else:             log.warning("⚠️  QLoRA could not be loaded, using fallback")
        except Exception as exc:
            log.warning(f"QLoRA pre-load failed: {exc}")

    await loop.run_in_executor(executor, load_all)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_ui():
    return HTMLResponse(content=HTML_CONTENT)


@app.get("/health", tags=["System"])
def health():
    return {
        "status"            : "ok",
        "device"            : DEVICE,
        "model_loaded"      : _model is not None,
        "qlora_loaded"      : _qlora_model is not None,
        "ngrok_url"         : _ngrok_public_url,
        "checkpoint"        : _resolve_checkpoint(CHECKPOINT_PATH),
        "qlora_path"        : QLORA_ADAPTER_PATH,
        "qlora_base_model"  : "Qwen/Qwen2.5-7B-Instruct",
        "qlora_mode"        : "sequential_per_cycle_independent",
        "top_cycles_for_dis": TOP_CYCLES_FOR_DISEASE,
        "version"           : "5.2.0",
        "key_changes"       : [
            "SEQ-1: call_local_llm_sequential_cycles — call QLoRA independently and sequentially per cycle",
            "SEQ-2: build_input_text_single_cycle — separate input per cycle",
            "SEQ-3: _call_qlora_single — 1 call = 1 cycle, wait for output before proceeding",
            "SEQ-4: aggregate_cycle_qlora_results — majority vote of 3 results",
            "response.qlora_per_cycle — array of individual per-cycle results for UI",
        ],
        "timestamp"         : datetime.utcnow().isoformat(),
    }


@app.get("/ngrok-url", tags=["System"])
def get_ngrok_url():
    return {
        "ngrok_url": _ngrok_public_url,
        "local_url": f"http://localhost:{PORT}",
        "active"   : bool(_ngrok_public_url),
    }

# ═════════════════════════════════════════════════════════════════════════════
#  /analyze  ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/analyze", tags=["Inference"])
async def analyze(
    file: UploadFile = File(
        ...,
        description="Stethoscope recording file: WAV/MP3/FLAC/OGG/M4A/WEBM (≤ 100 MB)",
    )
):
    t0 = time.time()

    # ── Validate ──────────────────────────────────────────────────────────────
    ALLOWED = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    ext     = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED and not (file.content_type or "").startswith("audio"):
        raise HTTPException(400, f"Format '{ext}' not supported. Use: {sorted(ALLOWED)}")

    wav_bytes = await file.read()
    if len(wav_bytes) < 1_024:
        raise HTTPException(422, "Audio file is too small or empty")
    if len(wav_bytes) > 100 * 1024 * 1024:
        raise HTTPException(413, "File too large (maximum 100 MB)")

    # ── Load audio ────────────────────────────────────────────────────────────
    try:
        cycles, duration, _ = load_wav_bytes(wav_bytes)
    except Exception as exc:
        log.error(traceback.format_exc())
        raise HTTPException(422, f"Unable to read audio file: {exc}")

    log.info(f"[{file.filename}] {len(cycles)} cycles | duration={duration}s")

    model      = get_model()
    request_id = uuid.uuid4().hex[:8]
    cycle_results: List[dict] = []
    all_emb_d : List[torch.Tensor] = []
    gradcam_imgs: List[str] = []

    # ── Pass 1: event + emb_d per cycle ──────────────────────────────────────
    with torch.no_grad():
        for idx, cyc in enumerate(cycles):
            mel_t = cyc["mel_tensor"].to(DEVICE)
            emb_e, emb_d, ev_logits = model(mel_t)
            pred_ev  = int(ev_logits.argmax(1).item())
            ev_probs = torch.softmax(ev_logits[0], dim=0).cpu().numpy().tolist()
            all_emb_d.append(emb_d.float())

            seg_dis_logits = model.patient_disease(emb_d.float())
            pred_dis_seg   = int(seg_dis_logits.argmax(1).item())

            cycle_results.append({
                "cycle_index"        : idx + 1,
                "start_sec"          : cyc["start_sec"],
                "end_sec"            : cyc["end_sec"],
                "mel_tensor_ref"     : cyc["mel_tensor"],
                "event"              : EVENT_NAMES[pred_ev],
                "event_chip"         : EVENT_CHIP[EVENT_NAMES[pred_ev]][0],
                "event_vi"           : EVENT_CHIP[EVENT_NAMES[pred_ev]][1],
                "event_confidence"   : round(float(ev_probs[pred_ev]), 4),
                "event_probabilities": {EVENT_NAMES[i]: round(p, 4)
                                        for i, p in enumerate(ev_probs)},
                "pred_ev_idx"        : pred_ev,
                "pred_dis_seg_idx"   : pred_dis_seg,
                "cam_event"          : {},
                "cam_disease"        : {},
                "cam_disease_alt"    : {},
                "cam_diff"           : {},
            })

    # ── Patient-level disease ─────────────────────────────────────────────────
    stacked = torch.cat(all_emb_d, dim=0).to(DEVICE)
    with torch.no_grad():
        dis_logits = model.patient_disease(stacked)
        pred_dis   = int(dis_logits.argmax(1).item())
        dis_probs  = torch.softmax(dis_logits[0], dim=0).cpu().numpy().tolist()

    dis_name       = DISEASE_NAMES[pred_dis]
    dis_conf       = round(float(dis_probs[pred_dis]) * 100)
    sorted_dis_idx = np.argsort(dis_probs)[::-1]
    alt_dis_idx    = int(sorted_dis_idx[1]) if len(sorted_dis_idx) > 1 else pred_dis
    alt_dis_name   = DISEASE_NAMES[alt_dis_idx]

    # ── GradCAM per cycle ─────────────────────────────────────────────────────
    for idx, (cyc, cr) in enumerate(zip(cycles, cycle_results)):
        mel_t   = cr["mel_tensor_ref"].to(DEVICE)
        pred_ev = cr["pred_ev_idx"]

        cam_ev_pred_map              = _gcam_ev.compute(mel_t, pred_ev)
        cam_dis_pred_map, cam_dis_alt_map = _gcam_dis.dual_target(
            mel_t, pred_dis, alt_dis_idx)

        cr["cam_event"]         = extract_cam_raw(cam_ev_pred_map)
        cr["cam_disease"]       = extract_cam_raw(cam_dis_pred_map)
        cr["cam_disease_alt"]   = extract_cam_raw(cam_dis_alt_map)
        cr["cam_diff"]          = extract_cam_diff(cam_dis_pred_map, cam_dis_alt_map)
        cr["alt_disease"]       = alt_dis_name

        mel_np   = cr["mel_tensor_ref"].squeeze().numpy()
        img_name = f"{request_id}_c{idx:02d}.png"
        save_cam_image(
            mel_np, cam_ev_pred_map, cam_ev_pred_map,
            cam_dis_pred_map, cam_dis_alt_map,
            str(OUT_DIR / img_name),
            title=(
                f"Cycle {idx+1} [{cr['start_sec']}s–{cr['end_sec']}s] "
                f"| Event: {cr['event']} | Disease: {DISEASE_NAMES[cr['pred_dis_seg_idx']]} "
                f"| Alt: {alt_dis_name}"
            ),
        )
        cr["gradcam_image_url"] = f"/static/gradcam/{img_name}"
        gradcam_imgs.append(f"/static/gradcam/{img_name}")

        cr.pop("mel_tensor_ref", None)

    # ── Select Top-3 cycles ───────────────────────────────────────────────────
    top3_cycles = select_top_cycles(cycle_results, n=TOP_CYCLES_FOR_DISEASE)

    log.info(
        f"[{file.filename}] Top-3 cycles: "
        + ", ".join(
            f"C{c['cycle_index']}({c['event']},peak={c['cam_disease'].get('peak',0):.3f})"
            for c in top3_cycles
        )
    )

    # ── Dominant event ────────────────────────────────────────────────────────
    ev_counts = {}
    for cr in cycle_results:
        ev_counts[cr["event"]] = ev_counts.get(cr["event"], 0) + 1
    dominant_event   = max(ev_counts, key=ev_counts.get)
    dom_chip, dom_vi = EVENT_CHIP[dominant_event]

    # ── Uncertainty ───────────────────────────────────────────────────────────
    top1           = top3_cycles[0] if top3_cycles else cycle_results[0]
    top1_ev_tensor = torch.tensor(list(top1["event_probabilities"].values()))
    ev_unc  = compute_uncertainty(torch.log(top1_ev_tensor + 1e-8), EVENT_NAMES)
    dis_unc = compute_uncertainty(dis_logits[0].cpu(), DISEASE_NAMES)

    # ── Differentials ─────────────────────────────────────────────────────────
    differentials = sorted(
        [{"name": DISEASE_NAMES[i], "nameVI": DISEASE_VI[DISEASE_NAMES[i]][0],
          "probability": round(dis_probs[i] * 100)}
         for i in range(NUM_DISEASE) if i != pred_dis],
        key=lambda x: x["probability"], reverse=True,
    )

    # ── [SEQ] QLoRA: call sequentially and independently per cycle ────────────
    llm_source = "fallback"
    log.info(f"[{file.filename}] Starting QLoRA sequential — {len(top3_cycles)} cycles")

    cycle_qlora_results = call_local_llm_sequential_cycles(
        top3_cycles    = top3_cycles,
        pred_disease   = dis_name,
        dominant_event = dominant_event,
        n_segments     = len(cycles),
        ev_unc         = ev_unc,
        dis_unc        = dis_unc,
        dis_probs      = dis_probs,
        alt_disease    = alt_dis_name,
    )

    n_success = sum(1 for r in cycle_qlora_results if r is not None)
    log.info(f"[{file.filename}] QLoRA sequential complete — {n_success}/{len(top3_cycles)} successful")

    if n_success > 0:
        llm_source = "qlora"
        clinical   = aggregate_cycle_qlora_results(
            cycle_results_qlora = cycle_qlora_results,
            pred_disease        = dis_name,
            dis_conf            = dis_conf,
        )
    else:
        log.info(f"[{file.filename}] All QLoRA cycles failed -> using fallback")
        clinical = fallback_llm_v2(dis_name, dominant_event)

    # ── Patient-level signals for UI ──────────────────────────────────────────
    pseudo_topk  = build_pseudo_topk_v2(dis_name, dominant_event, dis_probs)
    pseudo_proto = build_prototype_scores(dis_probs)
    soft_ret     = soft_retrieval_stats(pseudo_topk)
    proto_dict   = {DISEASE_NAMES[i]: pseudo_proto[i] for i in range(NUM_DISEASE)}

    # ── Build response ────────────────────────────────────────────────────────
    proc_ms = round((time.time() - t0) * 1000)
    log.info(
        f"[{file.filename}] ✅ {proc_ms}ms | "
        f"cnn_disease={dis_name}({dis_conf}%) | "
        f"event={dominant_event} | llm={llm_source} | "
        f"top3={[c['cycle_index'] for c in top3_cycles]}"
    )

    return JSONResponse({
        "result": {
            # ── Event ────────────────────────────────────────────────────────
            "soundType"    : dom_chip,
            "soundTypeVN"  : dom_vi,
            "dominantEvent": dominant_event,
            "eventCounts"  : ev_counts,

            # ── CNN primary diagnosis ─────────────────────────────────────────
            "primaryDiagnosis": {
                "name"       : clinical["dis_vi"],
                "nameEN"     : clinical["dis_en"],
                "probability": dis_conf,
                "severity"   : clinical["severity"],
                "source"     : "CNN-DualBranch",
                "disease"    : dis_name,
            },
            "differentials"            : differentials,
            "qloraAlternativeDiagnosis": clinical.get("qlora_alt_diagnosis"),

            # ── LLM aggregated output ─────────────────────────────────────────
            "confidence"     : clinical["confidence"],
            "clinicalNote"   : clinical["clinicalNote"],
            "recommendations": clinical["recommendations"],

            # ── QLoRA aggregated verdict ──────────────────────────────────────
            "qlora_dis_correct" : clinical.get("dis_correct"),
            "qlora_is_correct"  : clinical.get("is_correct"),
            "qlora_correct_flag": clinical.get("correct_flag"),
            "qlora_parse_ok"    : clinical.get("parse_ok", False),

            # ── [SEQ] Individual per-cycle results ────────────────────────────
            # Array of 3 elements, each with its own qlora_steps
            "qlora_per_cycle"   : clinical.get("qlora_per_cycle", []),

            # ── CNN raw ───────────────────────────────────────────────────────
            "cnn_pred_disease"   : dis_name,
            "cnn_pred_disease_vi": DISEASE_VI[dis_name][0],
            "cnn_confidence"     : dis_conf,
            "cnn_alt_disease"    : alt_dis_name,
            "uncertainty": {
                "event"  : ev_unc,
                "disease": dis_unc,
            },

            # ── Top-3 cycles (CNN selection) ──────────────────────────────────
            "top3_cycles": [
                {
                    "cycle_index"     : c["cycle_index"],
                    "start_sec"       : c["start_sec"],
                    "end_sec"         : c["end_sec"],
                    "rank"            : c.get("top_cycle_rank", 0),
                    "event"           : c["event"],
                    "event_vi"        : c["event_vi"],
                    "event_confidence": c["event_confidence"],
                    "disease_pred"    : dis_name,
                    "alt_disease"     : c.get("alt_disease", alt_dis_name),
                    "cam_event"       : c["cam_event"],
                    "cam_disease"     : c["cam_disease"],
                    "cam_disease_alt" : c["cam_disease_alt"],
                    "cam_diff"        : c["cam_diff"],
                    "gradcam_image_url": c.get("gradcam_image_url", ""),
                }
                for c in top3_cycles
            ],

            # ── All cycles ────────────────────────────────────────────────────
            "cycles": [
                {k: v for k, v in cr.items()
                 if k not in ("mel_tensor_ref", "pred_ev_idx",
                              "pred_dis_seg_idx", "alt_disease_idx")}
                for cr in cycle_results
            ],
            "totalCycles"   : len(cycle_results),
            "audioDuration" : duration,
            "gradcam_images": gradcam_imgs,

            # ── Patient-level signals ─────────────────────────────────────────
            "qlora_input_signals": {
                "soft_retrieval"  : soft_ret,
                "prototype_scores": proto_dict,
                "ev_uncertainty"  : ev_unc,
                "dis_uncertainty" : dis_unc,
                "top3_cycles_used": [c["cycle_index"] for c in top3_cycles],
            },

            # ── Meta ──────────────────────────────────────────────────────────
            "request_id"        : request_id,
            "processing_time_ms": proc_ms,
            "timestamp"         : datetime.utcnow().isoformat(),
            "model_device"      : DEVICE,
            "model_version"     : "DualBranch-v5.2",
            "qlora_base_model"  : "Qwen/Qwen2.5-7B-Instruct",
            "qlora_mode"        : "sequential_per_cycle_independent",
            "llm_source"        : llm_source,
            "ngrok_url"         : _ngrok_public_url,
        }
    })


# ═════════════════════════════════════════════════════════════════════════════
#  /analyze/top3  — Lightweight (no QLoRA call)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/analyze/top3", tags=["Inference"])
async def analyze_top3(
    file: UploadFile = File(..., description="Stethoscope recording file")
):
    """Lightweight: returns only top-3 cycles + disease, does not call QLoRA."""
    t0 = time.time()

    ALLOWED = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    ext     = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED and not (file.content_type or "").startswith("audio"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    wav_bytes = await file.read()
    if len(wav_bytes) < 1_024:
        raise HTTPException(422, "Audio file too small")

    try:
        cycles, duration, _ = load_wav_bytes(wav_bytes)
    except Exception as exc:
        raise HTTPException(422, f"Audio read error: {exc}")

    model         = get_model()
    request_id    = uuid.uuid4().hex[:8]
    cycle_results = []
    all_emb_d     = []

    with torch.no_grad():
        for idx, cyc in enumerate(cycles):
            mel_t = cyc["mel_tensor"].to(DEVICE)
            emb_e, emb_d, ev_logits = model(mel_t)
            pred_ev  = int(ev_logits.argmax(1).item())
            ev_probs = torch.softmax(ev_logits[0], dim=0).cpu().numpy().tolist()
            all_emb_d.append(emb_d.float())

            seg_dis_logits = model.patient_disease(emb_d.float())
            pred_dis_seg   = int(seg_dis_logits.argmax(1).item())
            dis_probs_seg  = torch.softmax(seg_dis_logits[0], dim=0).cpu().numpy().tolist()
            sorted_d       = np.argsort(dis_probs_seg)[::-1]
            alt_seg        = int(sorted_d[1]) if len(sorted_d) > 1 else pred_dis_seg

            cam_ev_pred_map              = _gcam_ev.compute(mel_t, pred_ev)
            cam_dis_pred_map, cam_dis_alt_map = _gcam_dis.dual_target(mel_t, pred_dis_seg, alt_seg)

            cycle_results.append({
                "cycle_index"     : idx + 1,
                "start_sec"       : cyc["start_sec"],
                "end_sec"         : cyc["end_sec"],
                "event"           : EVENT_NAMES[pred_ev],
                "event_vi"        : EVENT_CHIP[EVENT_NAMES[pred_ev]][1],
                "event_confidence": round(float(ev_probs[pred_ev]), 4),
                "cam_event"       : extract_cam_raw(cam_ev_pred_map),
                "cam_disease"     : extract_cam_raw(cam_dis_pred_map),
                "cam_disease_alt" : extract_cam_raw(cam_dis_alt_map),
                "cam_diff"        : extract_cam_diff(cam_dis_pred_map, cam_dis_alt_map),
            })

    stacked = torch.cat(all_emb_d, dim=0).to(DEVICE)
    with torch.no_grad():
        dis_logits = model.patient_disease(stacked)
        pred_dis   = int(dis_logits.argmax(1).item())
        dis_probs  = torch.softmax(dis_logits[0], dim=0).cpu().numpy().tolist()

    dis_name = DISEASE_NAMES[pred_dis]
    dis_conf = round(float(dis_probs[pred_dis]) * 100)
    top3     = select_top_cycles(cycle_results, n=TOP_CYCLES_FOR_DISEASE)
    proc_ms  = round((time.time() - t0) * 1000)

    return JSONResponse({
        "top3_cycles": top3,
        "disease": {
            "name"      : dis_name,
            "nameVI"    : DISEASE_VI[dis_name][0],
            "confidence": dis_conf,
            "severity"  : SEVERITY_MAP[dis_name],
            "probs"     : {DISEASE_NAMES[i]: round(dis_probs[i]*100, 1)
                           for i in range(NUM_DISEASE)},
        },
        "total_cycles"      : len(cycle_results),
        "processing_time_ms": proc_ms,
        "request_id"        : request_id,
    })


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

import nest_asyncio
nest_asyncio.apply()

import asyncio


async def start_server():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=PORT, log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


threading.Thread(target=_delayed_ngrok, args=(PORT,), daemon=True).start()

await start_server()
