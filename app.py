import os
import math
import json
import tempfile

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch

from config import PTBXLConfig
from model import build_model
from signal_ops import (
    bandpass_filter, zscore_normalize, zscore_normalize_global,
    fix_length, sanitize,
)

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

CLASS_LABELS_LONG = {
    'NORM': 'EKG Normal',
    'MI': 'Infark Miokard',
    'STTC': 'Perubahan ST/T',
    'CD': 'Gangguan Konduksi',
    'HYP': 'Hipertrofi',
}

CLASS_COLORS = {
    'NORM': '#22C55E',
    'MI': '#EF4444',
    'STTC': '#A855F7',
    'CD': '#F59E0B',
    'HYP': '#3B82F6',
}

CLASS_COLORS_RGBA = {
    'NORM': 'rgba(34, 197, 94, 0.22)',
    'MI': 'rgba(239, 68, 68, 0.22)',
    'STTC': 'rgba(168, 85, 247, 0.22)',
    'CD': 'rgba(245, 158, 11, 0.22)',
    'HYP': 'rgba(59, 130, 246, 0.22)',
}

TARGET_LEADS = 12

SPIKE_WINDOW_SEC = 0.16
SPIKE_MIN_DISTANCE_SEC = 0.15
SPIKES_PER_LABEL = 5

MODEL_PATH = "checkpoints/best_model.pt"
THRESHOLDS_PATH = "checkpoints/best_thresholds.json"

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

st.set_page_config(
    page_title="ECG Multi-Label Diagnosis",
    page_icon=":anatomical_heart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_ECG_PERIOD = ("l 24 0 l 6 -5 l 6 5 l 16 0 l 4 6 l 4 -46 l 4 40 "
               "l 4 10 l 4 -10 l 10 0 l 8 -8 l 8 8 l 102 0 ")
_ECG_PATH_D = "M 0 40 " + (_ECG_PERIOD * 12)

def render_ecg_trace(extra_class=""):
    return (
        f'<div class="ecg-trace-wrap {extra_class}">'
        f'<svg viewBox="0 -8 1200 60" preserveAspectRatio="none" class="ecg-trace-svg">'
        f'<path d="{_ECG_PATH_D}" class="ecg-trace-path" fill="none"></path>'
        f'</svg></div>'
    )

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg-0: #0B0F14; --bg-1: #11161D; --bg-2: #161D26; --bg-3: #1B232E;
  --border: rgba(255,255,255,0.08); --text-1: #E8EDF2; --text-2: #8B97A6;
  --accent: #26C6DA; --accent-2: #7C3AED;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container { max-width: 1100px !important; padding-top: 1.6rem !important; padding-bottom: 3rem !important; }

.wide-plot-stage { width: 100%; overflow-x: auto; border-radius: 14px; border: 1px solid var(--border);
  background: var(--bg-2); padding: 6px 6px 2px 6px; }
.wide-plot-stage::-webkit-scrollbar { height: 10px; }
.wide-plot-stage::-webkit-scrollbar-track { background: var(--bg-1); border-radius: 6px; }
.wide-plot-stage::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 6px; opacity: 0.6; }
.wide-plot-hint { display: flex; align-items: center; gap: 6px; color: var(--text-2); font-size: 12px;
  margin: 6px 2px 12px 2px; }
.wide-plot-hint svg { opacity: 0.7; }

.metric-row { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }

.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(38,198,218,0.07), transparent 42%),
    radial-gradient(circle at 88% 92%, rgba(124,58,237,0.06), transparent 42%),
    repeating-linear-gradient(0deg, rgba(38,198,218,0.035) 0px, rgba(38,198,218,0.035) 1px, transparent 1px, transparent 28px),
    repeating-linear-gradient(90deg, rgba(38,198,218,0.035) 0px, rgba(38,198,218,0.035) 1px, transparent 1px, transparent 28px),
    var(--bg-0);
  background-attachment: fixed;
}

h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; background: linear-gradient(135deg, #E8EDF2 0%, #26C6DA 120%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 2px !important; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em !important; color: var(--text-1) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-2) !important; }

.hero { padding-top: 2px; }
.hero-heart { display: inline-block; animation: heartbeat 2.1s ease-in-out infinite; transform-origin: 50% 60%; }
.hero-sub { color: var(--text-2); font-size: 14.5px; margin: 2px 0 14px 0; }

.ecg-trace-wrap { width: 100%; height: 46px; overflow: hidden; margin: 0 0 22px 0; opacity: 0.9;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent);
  mask-image: linear-gradient(90deg, transparent, #000 10%, #000 90%, transparent); }
.ecg-trace-svg { width: 100%; height: 46px; display: block; }
.ecg-trace-path { stroke: var(--accent); stroke-width: 2.2; filter: drop-shadow(0 0 6px rgba(38,198,218,0.55));
  animation: ecgScroll 6.5s linear infinite; }
.ecg-trace-wrap.dim .ecg-trace-path { stroke: #7C3AED; filter: drop-shadow(0 0 6px rgba(124,58,237,0.55)); }

.stButton > button { border-radius: 10px; font-weight: 600; border: 1px solid var(--border); transition: all 0.18s cubic-bezier(.22,.85,.32,1); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #26C6DA 0%, #1E88E5 100%); border: none; box-shadow: 0 4px 14px rgba(38,198,218,0.25); }
.stButton > button[kind="primary"]:hover { box-shadow: 0 6px 22px rgba(38,198,218,0.45); transform: translateY(-1px); }
.stButton > button[kind="primary"]:active { transform: translateY(0); }
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
  border-radius: 8px !important; background: var(--bg-2) !important; border: 1px solid var(--border) !important;
  transition: border-color .15s ease, box-shadow .15s ease; }
.stTextInput input:focus, .stNumberInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(38,198,218,0.18) !important; }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

.glass-card { background: var(--bg-2); border: 1px solid var(--border); border-radius: 16px; padding: 20px 22px; backdrop-filter: blur(8px); }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; background: rgba(38,198,218,0.12); color: #26C6DA; border: 1px solid rgba(38,198,218,0.25); }

.dx-card { position: relative; border-radius: 16px; padding: 18px 14px 16px 14px; text-align: center; overflow: hidden;
  transition: transform 0.22s cubic-bezier(.22,.85,.32,1), box-shadow 0.22s cubic-bezier(.22,.85,.32,1); }
.dx-card:hover { transform: translateY(-4px) scale(1.015); }
.dx-label { font-size: 16px; font-weight: 800; letter-spacing: -0.01em; color: #FFFFFF; margin-bottom: 1px; }
.dx-sublabel { font-size: 10.5px; font-weight: 500; color: rgba(255,255,255,0.75); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.dx-ring { width: 96px; height: 96px; margin: 2px auto 2px auto; display: block; }
.dx-ring-text { font-family: 'JetBrains Mono', monospace; font-size: 17px; font-weight: 700; fill: #FFFFFF; }
.dx-thr { margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: rgba(255,255,255,0.7); }
.dx-badge { position: absolute; top: 10px; right: 10px; width: 9px; height: 9px; border-radius: 50%; background: #fff; box-shadow: 0 0 8px rgba(255,255,255,0.9); }

.legend-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: var(--bg-2); border: 1px solid var(--border); color: var(--text-1); margin-right: 8px; margin-bottom: 8px;
  transition: transform .15s ease, border-color .15s ease; }
.legend-chip:hover { transform: translateY(-1px); border-color: var(--accent); }
.legend-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

.step-card { display: flex; flex-direction: column; }
.step-next { margin-top: 10px; color: var(--accent); font-size: 13px; font-weight: 700; text-align: right; opacity: 0.85; }

.scan-banner { margin: 6px 0 22px 0; }
.scan-track { position: relative; height: 10px; border-radius: 999px; background: var(--bg-2); border: 1px solid var(--border); overflow: hidden; }
.scan-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent-2));
  transition: width .45s cubic-bezier(.22,.85,.32,1); position: relative; overflow: hidden; }
.scan-fill::after { content: ""; position: absolute; inset: 0; width: 40%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
  animation: shimmerSlide 1.3s ease-in-out infinite; }
.scan-label { margin-top: 8px; font-size: 12.5px; color: var(--text-2); font-family: 'JetBrains Mono', monospace; letter-spacing: .02em; }

div[data-testid="stExpander"] { background: var(--bg-2); border: 1px solid var(--border); border-radius: 12px; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; transition: color .15s ease; }
hr { border-color: var(--border) !important; }

.fade-in-up { animation: fadeInUp 0.5s cubic-bezier(.22,.85,.32,1) both; }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes heartbeat { 0%, 100% { transform: scale(1); } 14% { transform: scale(1.18); } 28% { transform: scale(1); } 42% { transform: scale(1.12); } 70% { transform: scale(1); } }
@keyframes ecgScroll { from { transform: translateX(0); } to { transform: translateX(-1200px); } }
@keyframes shimmerSlide { 0% { transform: translateX(-130%); } 100% { transform: translateX(280%); } }
@keyframes pulseGlow { 0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.0); } 50% { box-shadow: 0 0 0 6px rgba(255,255,255,0.18); } }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_and_config(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_args = ckpt.get("config", {}) or {}

    ptb_config = PTBXLConfig(sampling_rate=run_args.get("sampling_rate", 100))
    if run_args.get("normalize_mode") is not None:
        ptb_config.normalize_mode = run_args["normalize_mode"]
    if run_args.get("use_lead_dropout") is not None:
        ptb_config.use_lead_dropout = run_args["use_lead_dropout"]
    if run_args.get("lead_dropout_prob") is not None:
        ptb_config.lead_dropout_prob = run_args["lead_dropout_prob"]

    model = build_model(
        variant=run_args.get("model_variant", "tiny"),
        in_channels=ptb_config.n_leads,
        num_classes=len(CLASSES),
        drop_path_rate=run_args.get("drop_path_rate", 0.1),
        se_reduction=run_args.get("se_reduction", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ptb_config

def load_thresholds(path):
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            return {c: float(raw.get(c, 0.5)) for c in CLASSES}
        except Exception:
            pass
    return {c: 0.5 for c in CLASSES}

def read_wfdb_record(hea_bytes, dat_bytes, record_name="uploaded_record"):
    import wfdb

    hea_text = hea_bytes.decode('utf-8', errors='ignore')
    first_line = hea_text.strip().split('\n')[0]

    if first_line:
        true_name = first_line.split()[0].strip()
        true_name = os.path.basename(true_name)
    else:
        true_name = record_name

    tmp_dir = tempfile.mkdtemp()

    hea_path = os.path.join(tmp_dir, true_name + ".hea")
    dat_path = os.path.join(tmp_dir, true_name + ".dat")

    with open(hea_path, "wb") as f:
        f.write(hea_bytes)
    with open(dat_path, "wb") as f:
        f.write(dat_bytes)

    record = wfdb.rdrecord(os.path.join(tmp_dir, true_name))
    signal = record.p_signal.astype(np.float32)
    fs = record.fs
    sig_names = record.sig_name

    try:
        os.remove(hea_path)
        os.remove(dat_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return signal, fs, sig_names

def _parse_hea_meta(hea_text):
    lines = [l for l in hea_text.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
    meta = {"fs": None, "sig_names": [], "gains": [], "baselines": []}
    if not lines:
        return meta
    header_tokens = lines[0].split()
    try:
        meta["n_sig"] = int(header_tokens[1])
    except Exception:
        meta["n_sig"] = max(0, len(lines) - 1)
    try:
        meta["fs"] = float(header_tokens[2])
    except Exception:
        meta["fs"] = None
    for line in lines[1:1 + meta.get("n_sig", 0)]:
        toks = line.split()
        gain, baseline = 1.0, 0.0
        if len(toks) > 2:
            gain_field = toks[2]
            try:
                if "(" in gain_field and ")" in gain_field:
                    gain_str, base_str = gain_field.split("(")
                    gain = float(gain_str) if gain_str else 1.0
                    baseline = float(base_str.replace(")", "").split("/")[0])
                else:
                    gain = float(gain_field.split("/")[0])
            except Exception:
                gain, baseline = 1.0, 0.0
        name = toks[-1] if len(toks) >= 9 else (toks[-1] if toks else "")
        meta["sig_names"].append(name)
        meta["gains"].append(gain if gain not in (0, None) else 1.0)
        meta["baselines"].append(baseline)
    return meta

def read_mat_record(mat_bytes, hea_bytes=None, record_name="uploaded_record"):
    import io as _io

    if hea_bytes:
        try:
            import wfdb

            hea_text = hea_bytes.decode('utf-8', errors='ignore')
            first_line = hea_text.strip().split('\n')[0]
            true_name = os.path.basename(first_line.split()[0].strip()) if first_line else record_name

            tmp_dir = tempfile.mkdtemp()
            hea_path = os.path.join(tmp_dir, true_name + ".hea")
            mat_path = os.path.join(tmp_dir, true_name + ".mat")
            with open(hea_path, "wb") as f:
                f.write(hea_bytes)
            with open(mat_path, "wb") as f:
                f.write(mat_bytes)

            record = wfdb.rdrecord(os.path.join(tmp_dir, true_name))
            signal = record.p_signal.astype(np.float32)
            fs = record.fs
            sig_names = record.sig_name

            try:
                os.remove(hea_path)
                os.remove(mat_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass

            return signal, fs, sig_names
        except Exception:
            pass

    from scipy.io import loadmat

    mat_dict = loadmat(_io.BytesIO(mat_bytes))
    array_keys = [k for k in mat_dict.keys() if not k.startswith("__")]

    candidate = None
    for preferred in ("val", "ECG", "ecg", "data", "signal", "Data"):
        if preferred in mat_dict and isinstance(mat_dict[preferred], np.ndarray):
            candidate = mat_dict[preferred]
            break
    if candidate is None:
        for k in array_keys:
            v = mat_dict[k]
            if isinstance(v, np.ndarray) and v.ndim == 2 and min(v.shape) > 1:
                candidate = v
                break
    if candidate is None:
        raise ValueError(
            f"Could not find a usable signal matrix inside the .mat file "
            f"(keys found: {array_keys})."
        )

    signal = np.asarray(candidate, dtype=np.float32)

    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    n_samples, n_leads = signal.shape

    fs = 500.0
    sig_names = None

    if hea_bytes:
        meta = _parse_hea_meta(hea_bytes.decode('utf-8', errors='ignore'))
        if meta.get("fs"):
            fs = meta["fs"]
        if meta.get("sig_names") and len(meta["sig_names"]) == n_leads:
            sig_names = meta["sig_names"]

    if sig_names is None:
        sig_names = LEAD_NAMES[:n_leads] if n_leads == TARGET_LEADS else [f"Lead {i+1}" for i in range(n_leads)]

    return signal, fs, sig_names

def resample_to_rate(signal, fs, target_rate):
    n_samples, n_leads = signal.shape
    if fs == target_rate:
        return signal.astype(np.float32)
    duration_sec = n_samples / float(fs)
    n_target = max(1, int(round(duration_sec * target_rate)))
    x_old = np.linspace(0, duration_sec, n_samples, endpoint=False)
    x_new = np.linspace(0, duration_sec, n_target, endpoint=False)
    out = np.zeros((n_target, n_leads), dtype=np.float32)
    for lead in range(n_leads):
        out[:, lead] = np.interp(x_new, x_old, signal[:, lead])
    return out

def build_model_input(leads_12, fs, config: PTBXLConfig):
    resampled_full = resample_to_rate(leads_12, fs, config.sampling_rate)
    sig = resampled_full.T.astype(np.float32)
    sig = sanitize(sig)

    if config.use_bandpass_filter:
        sig = bandpass_filter(sig, config.sampling_rate, config.lowcut,
                               config.highcut, config.filter_order)

    sig = fix_length(sig, config.target_length, mode=config.crop_mode_eval)
    display_resampled = sig.T.copy()

    model_input = zscore_normalize_global(sig) if config.normalize_mode == "global" \
        else zscore_normalize(sig)
    model_input = sanitize(model_input)

    return model_input, display_resampled

def prepare_leads(signal, sig_names):
    n_samples, n_have = signal.shape
    out = np.zeros((n_samples, TARGET_LEADS), dtype=np.float32)
    found = []

    name_map = {name.upper().replace(" ", ""): i for i, name in enumerate(sig_names)}
    for target_idx, lead_name in enumerate(LEAD_NAMES):
        key = lead_name.upper()
        if key in name_map:
            out[:, target_idx] = signal[:, name_map[key]]
            found.append(lead_name)

    if not found:
        n_copy = min(n_have, TARGET_LEADS)
        out[:, :n_copy] = signal[:, :n_copy]
        found = sig_names[:n_copy]

    return out, found

def run_inference(model, signal_12xN, device):
    x = torch.tensor(signal_12xN, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs

def compute_lead_activity(signal):
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (signal - mean) / std
    return z ** 2

def find_top_spikes(activity, t_axis, n_spikes, min_distance_sec=SPIKE_MIN_DISTANCE_SEC):
    n_spikes = max(0, int(n_spikes))
    if n_spikes == 0 or len(activity) == 0:
        return []

    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 1.0
    min_distance_samples = max(1, int(min_distance_sec / dt)) if dt > 0 else 1

    order = np.argsort(activity)[::-1]
    chosen = []
    for idx in order:
        if all(abs(int(idx) - c) >= min_distance_samples for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n_spikes:
            break
    return sorted(chosen)

def render_diagnosis_ring(color, prob, threshold, active, index):
    r = 32
    circumference = 2 * math.pi * r
    target_offset = circumference * (1 - max(0.0, min(1.0, prob)))
    entrance_delay = index * 0.09
    ring_delay = entrance_delay + 0.15

    opacity = "1" if active else "0.32"
    box_shadow = f"0 10px 28px {color}55" if active else "none"
    border = f"1px solid {color}" if active else "1px solid var(--border)"
    badge = (
        f'<div class="dx-badge" style="background:{color}; animation: pulseGlow 1.8s ease-in-out infinite;"></div>'
        if active else ""
    )

    return f"""
    <style>@keyframes ringFill{index} {{ to {{ stroke-dashoffset: {target_offset:.2f}; }} }}</style>
    <div class="dx-card fade-in-up" style="animation-delay:{entrance_delay:.2f}s;
         background: linear-gradient(160deg, {color}E6 0%, {color}99 100%);
         opacity:{opacity}; border:{border}; box-shadow:{box_shadow};">
      {badge}
      <div class="dx-label">{CLASSES[index]}</div>
      <div class="dx-sublabel">{CLASS_LABELS_LONG[CLASSES[index]]}</div>
      <svg viewBox="0 0 80 80" class="dx-ring">
        <circle cx="40" cy="40" r="{r}" fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="7"></circle>
        <circle cx="40" cy="40" r="{r}" fill="none" stroke="#FFFFFF" stroke-width="7" stroke-linecap="round"
          stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"
          transform="rotate(-90 40 40)"
          style="animation: ringFill{index} 1s {ring_delay:.2f}s cubic-bezier(.22,.85,.32,1) forwards;"></circle>
        <text x="40" y="46" text-anchor="middle" class="dx-ring-text">{prob*100:.0f}%</text>
      </svg>
      <div class="dx-thr">ambang {threshold:.2f}</div>
    </div>
    """

st.markdown(
    '<div class="hero"><h1>Diagnosis Multi-Label EKG</h1>'
    '<p class="hero-sub">Klasifikasi ConvNeXt-1D + Squeeze-and-Excitation</p></div>',
    unsafe_allow_html=True,
)
st.markdown(render_ecg_trace(), unsafe_allow_html=True)

st.markdown('<div class="glass-card fade-in-up">', unsafe_allow_html=True)
st.markdown("##### Unggah Rekaman EKG")

input_format = st.radio(
    "Format file", ["WFDB (.hea + .dat)", "MATLAB (.mat)"],
    horizontal=True, label_visibility="collapsed",
)

c1, c2 = st.columns(2)
if input_format == "WFDB (.hea + .dat)":
    with c1:
        hea_file = st.file_uploader(".hea (header)", type=["hea"])
    with c2:
        dat_file = st.file_uploader(".dat (sinyal)", type=["dat"])
    mat_file = None
else:
    with c1:
        mat_file = st.file_uploader(".mat (sinyal)", type=["mat"])
    with c2:
        hea_file = st.file_uploader(".hea (header, opsional)", type=["hea"])
    dat_file = None

run_btn = st.button("Jalankan Diagnosis", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    if input_format == "WFDB (.hea + .dat)":
        if hea_file is None:
            st.error("Silakan unggah file .hea.")
            st.stop()
        if dat_file is None:
            st.error("Silakan unggah file .dat.")
            st.stop()
    else:
        if mat_file is None:
            st.error("Silakan unggah file .mat.")
            st.stop()

    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress_ph = st.empty()

    def show_progress(label, pct):
        progress_ph.markdown(
            f'<div class="scan-banner"><div class="scan-track">'
            f'<div class="scan-fill" style="width:{pct}%;"></div></div>'
            f'<div class="scan-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    show_progress("Reading record…", 12)
    try:
        if input_format == "WFDB (.hea + .dat)":
            raw_signal, fs, sig_names = read_wfdb_record(
                hea_file.getvalue(), dat_file.getvalue(), record_name="record"
            )
        else:
            raw_signal, fs, sig_names = read_mat_record(
                mat_file.getvalue(),
                hea_bytes=hea_file.getvalue() if hea_file is not None else None,
                record_name="record",
            )
    except Exception as e:
        progress_ph.empty()
        st.error(f"Gagal membaca rekaman: {e}")
        st.stop()

    show_progress("Mapping 12-lead layout…", 30)
    leads_12, found_leads = prepare_leads(raw_signal, sig_names)
    missing = [l for l in LEAD_NAMES if l not in found_leads]
    if missing:
        st.warning(f"Lead yang hilang diisi nol: {', '.join(missing)}")

    show_progress("Loading model weights…", 48)
    try:
        model, ptb_config = load_model_and_config(MODEL_PATH, device)
    except Exception as e:
        progress_ph.empty()
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    show_progress("Resampling, filtering & normalisasi sinyal…", 70)
    model_input, resampled = build_model_input(leads_12, fs, ptb_config)

    show_progress("Running inference…", 92)
    probs = run_inference(model, model_input, device)

    show_progress("Done", 100)
    progress_ph.empty()
    st.toast("Diagnosis selesai")

    thresholds = load_thresholds(THRESHOLDS_PATH)

    preds = {c: bool(probs[i] >= thresholds[c]) for i, c in enumerate(CLASSES)}
    active_labels = [c for c in CLASSES if preds[c]]

    st.markdown(
        '<div class="fade-in-up metric-row">'
        '<span class="pill">Rekaman berhasil dimuat</span>'
        f'<span class="pill">{fs} Hz</span>'
        f'<span class="pill">{raw_signal.shape[0]} samples</span>'
        f'<span class="pill">{len(found_leads)}/12 leads found</span>'
        f'<span class="pill">perangkat: {device}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    tab_dx, tab_signal, tab_prob = st.tabs(["Diagnosis", "Sinyal 12-Lead", "Probabilitas"])

    with tab_dx:
        st.markdown("### Papan Diagnosis")
        cols = st.columns(len(CLASSES))
        for i, c in enumerate(CLASSES):
            with cols[i]:
                st.markdown(
                    render_diagnosis_ring(CLASS_COLORS[c], float(probs[i]), thresholds[c], preds[c], i),
                    unsafe_allow_html=True,
                )

        st.write("")
        if active_labels:
            chips = "".join(
                f'<span class="legend-chip"><span class="legend-dot" style="background:{CLASS_COLORS[c]}"></span>{c}</span>'
                for c in active_labels
            )
            st.markdown(f"**Label aktif:** {chips}", unsafe_allow_html=True)
        else:
            st.info("Tidak ada label yang melewati ambang batas.")

    with tab_signal:
        st.markdown("### Sinyal 12-Lead")

        c_show, c_zoom, c_legend = st.columns([1, 1, 2.4])
        with c_show:
            show_raw = st.toggle("Sinyal mentah (panjang asli)", value=True)
        with c_zoom:
            plot_zoom = st.select_slider(
                "Lebar grafik", options=["1x", "1.5x", "2x", "2.5x", "3x"], value="2x",
            )
        zoom_factor = float(plot_zoom.replace("x", ""))

        plot_signal = raw_signal if show_raw else resampled
        plot_leads = sig_names if show_raw else LEAD_NAMES

        t_axis = np.arange(plot_signal.shape[0]) / (fs if show_raw else ptb_config.sampling_rate)
        t_min, t_max = float(t_axis[0]), float(t_axis[-1])

        n_leads_plot = plot_signal.shape[1]
        activity_per_lead = compute_lead_activity(plot_signal)
        lead_spike_assignments = []
        for lead_idx in range(n_leads_plot):
            n_spikes_total = len(active_labels) * SPIKES_PER_LABEL
            lead_spikes = find_top_spikes(activity_per_lead[:, lead_idx], t_axis, n_spikes=n_spikes_total)
            lead_spike_assignments.append([
                (active_labels[i % len(active_labels)], idx) for i, idx in enumerate(lead_spikes)
            ] if active_labels else [])

        with c_legend:
            if active_labels:
                legend_html = "".join(
                    f'<span class="legend-chip"><span class="legend-dot" style="background:{CLASS_COLORS[c]}"></span>{c}</span>'
                    for c in active_labels
                )
                st.markdown(legend_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span class="legend-chip"><span class="legend-dot" style="background:#4B5563"></span>Tidak ada diagnosis aktif — tidak ada lonjakan yang disorot</span>',
                    unsafe_allow_html=True,
                )

        fig = make_subplots(
            rows=n_leads_plot, cols=1, shared_xaxes=True,
            subplot_titles=[str(l) for l in plot_leads],
            vertical_spacing=0.012
        )

        shapes = []
        lead_y_ranges = []
        half_window = SPIKE_WINDOW_SEC / 2.0
        for lead_idx in range(n_leads_plot):
            row = lead_idx + 1
            xref = "x" if row == 1 else f"x{row}"
            yref = "y" if row == 1 else f"y{row}"

            lead_signal = plot_signal[:, lead_idx]
            lead_min = float(lead_signal.min())
            lead_max = float(lead_signal.max())
            pad = (lead_max - lead_min) * 0.08 or 1.0
            y0, y1 = lead_min - pad, lead_max + pad
            lead_y_ranges.append((y0, y1))

            for c, idx in lead_spike_assignments[lead_idx]:
                spike_t = float(t_axis[idx])
                x0 = max(t_min, spike_t - half_window)
                x1 = min(t_max, spike_t + half_window)
                shapes.append(dict(
                    type="rect",
                    xref=xref, yref=yref,
                    x0=x0, x1=x1, y0=y0, y1=y1,
                    fillcolor=CLASS_COLORS_RGBA[c],
                    line=dict(width=1, color=CLASS_COLORS[c]),
                    layer="below",
                ))

        for i in range(n_leads_plot):
            fig.add_trace(
                go.Scatter(x=t_axis, y=plot_signal[:, i], mode="lines",
                           line=dict(width=1.3, color="#26C6DA"), showlegend=False),
                row=i + 1, col=1
            )

        ECG_PAPER_BG = "#FFF7F5"
        ECG_MINOR_GRID = "#FFC9C4"
        ECG_MAJOR_GRID = "#E8524A"
        ECG_TRACE_COLOR = "#16324F"
        ECG_TEXT_COLOR = "#5B2A26"

        x_major_dtick = 0.2
        x_minor_dtick = 0.04

        fig.update_layout(
            height=88 * n_leads_plot,
            width=int(1380 * zoom_factor),
            margin=dict(l=44, r=24, t=30, b=30),
            template="plotly_white",
            plot_bgcolor=ECG_PAPER_BG,
            paper_bgcolor=ECG_PAPER_BG,
            shapes=shapes,
            font=dict(family="Inter, sans-serif", color=ECG_TEXT_COLOR),
            transition=dict(duration=300, easing="cubic-in-out"),
        )
        for ann in fig.layout.annotations:
            ann.font = dict(family="Inter, sans-serif", color=ECG_TEXT_COLOR, size=11)

        for i in range(n_leads_plot):
            fig.data[i].line.color = ECG_TRACE_COLOR

        fig.update_xaxes(title_text="seconds", row=n_leads_plot, col=1)
        fig.update_xaxes(
            showgrid=True, gridcolor=ECG_MAJOR_GRID, gridwidth=1.1,
            griddash="solid", dtick=x_major_dtick, range=[t_min, t_max],
            minor=dict(showgrid=True, gridcolor=ECG_MINOR_GRID, gridwidth=0.6, dtick=x_minor_dtick),
            zeroline=False, linecolor=ECG_MAJOR_GRID,
        )

        for lead_idx in range(n_leads_plot):
            y0, y1 = lead_y_ranges[lead_idx]
            span = (y1 - y0) or 1.0

            y_major_dtick = span / 6.0
            y_minor_dtick = y_major_dtick / 5.0
            fig.update_yaxes(
                showgrid=True, gridcolor=ECG_MAJOR_GRID, gridwidth=1.1,
                griddash="solid", dtick=y_major_dtick,
                minor=dict(showgrid=True, gridcolor=ECG_MINOR_GRID, gridwidth=0.6, dtick=y_minor_dtick),
                zeroline=False, linecolor=ECG_MAJOR_GRID, showticklabels=False,
                row=lead_idx + 1, col=1,
            )

        st.markdown(
            '<div class="wide-plot-hint">'
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            '<path d="M8 5l-5 7 5 7M16 5l5 7-5 7"/></svg>'
            f'Ditampilkan pada lebar {plot_zoom} — geser secara horizontal di dalam panel untuk melihat semua {n_leads_plot} lead.'
            '</div>',
            unsafe_allow_html=True,
        )

        plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})
        stage_html = f"""
        <div class="wide-plot-stage" style="overflow-x:auto; border-radius:14px; border:1px solid #E8524A55;
             background:#FFF7F5; padding:6px 6px 2px 6px;">
            <div style="width:{int(1380 * zoom_factor)}px;">{plot_html}</div>
        </div>
        """
        components.html(stage_html, height=88 * n_leads_plot + 40, scrolling=False)

    with tab_prob:
        st.markdown("### Probabilitas Per Kelas")
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=CLASSES, y=probs,
            marker=dict(
                color=[CLASS_COLORS[c] for c in CLASSES],
                line=dict(width=1.5, color=["#FFFFFF" if preds[c] else "rgba(0,0,0,0)" for c in CLASSES]),
            ),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="#E8EDF2", size=13),
        ))
        for i, c in enumerate(CLASSES):
            bar_fig.add_hline(y=thresholds[c], line_dash="dot", line_color=CLASS_COLORS[c],
                               annotation_text=f"{c} thr", annotation_position="top right",
                               opacity=0.45)
        bar_fig.update_layout(
            yaxis_range=[0, 1], template="plotly_dark", height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#E8EDF2"),
            transition=dict(duration=400, easing="cubic-in-out"),
        )
        bar_fig.update_xaxes(showgrid=False)
        bar_fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(bar_fig, use_container_width=True)

else:
    st.write("")
    p1, p2, p3, p4 = st.columns(4)
    steps = [
        ("1", "Unggah", "File .dat atau .mat, .hea wajib untuk WFDB."),
        ("2", "Praproses", "Sinyal dipetakan ke 12 lead, di-resample, difilter, dan dinormalisasi persis seperti saat training."),
        ("3", "Prediksi", "ConvNeXt-1D + SE menghasilkan probabilitas per label."),
        ("4", "Diagnosis", "Ambang batas tersimpan menentukan label yang aktif."),
    ]
    for idx, (col, (num, title, desc)) in enumerate(zip([p1, p2, p3, p4], steps)):
        with col:
            next_hint = '<div class="step-next">selanjutnya →</div>' if idx < 3 else ''
            st.markdown(
                f'<div class="glass-card step-card fade-in-up" style="animation-delay:{idx*0.1:.2f}s; min-height:160px;">'
                f'<div class="pill">Langkah {num}</div>'
                f'<h4 style="margin:10px 0 6px 0;">{title}</h4>'
                f'<span style="color:var(--text-2); font-size:13.5px;">{desc}</span>'
                f'{next_hint}'
                '</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(render_ecg_trace("dim"), unsafe_allow_html=True)
    st.markdown("##### Legenda label")
    legend_html = "".join(
        f'<span class="legend-chip"><span class="legend-dot" style="background:{CLASS_COLORS[c]}"></span>{c} — {CLASS_LABELS_LONG[c]}</span>'
        for c in CLASSES
    )
    st.markdown(legend_html, unsafe_allow_html=True)