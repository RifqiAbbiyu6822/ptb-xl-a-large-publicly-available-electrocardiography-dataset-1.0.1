import os
import sys

ROOT = r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
CODE_DIR = os.path.join(ROOT, "preprocessing")
for p in (CODE_DIR, ROOT):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import wfdb
import matplotlib.pyplot as plt

from labels import load_ptbxl_metadata
from signal_ops import (
    bandpass_filter, fix_length, zscore_normalize_global, sanitize, lead_dropout,
)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SHOW = 1                      # lead II untuk gambar 3.10 dan 3.11
OUT_DIR = os.path.join(CODE_DIR, "gambar")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
FS_TARGET_LEN = 1000


# ---------- 1. Ambil 1 rekaman contoh & proses seperti pipeline (tanpa augmentasi) ----------
df = load_ptbxl_metadata(ROOT, CLASSES)
cand = df[(df["NORM"] == 1) & (df[CLASSES].sum(axis=1) == 1)]
row = cand.sample(1, random_state=SEED).iloc[0]
ecg_id = cand.sample(1, random_state=SEED).index[0]

sig, meta = wfdb.rdsamp(os.path.join(ROOT, row["filename_lr"]))
fs = meta["fs"]
x = sanitize(sig.T.astype(np.float32))
x = bandpass_filter(x, fs, 0.5, 40.0, 4)
x = fix_length(x, FS_TARGET_LEN, mode="center")
x = zscore_normalize_global(x)            # sinyal siap augmentasi, shape (12, 1000)
t = np.arange(x.shape[1]) / fs
print("ecg_id:", ecg_id, "| shape:", x.shape)


# ---------- 2. Gambar 3.10: derau Gaussian ----------
rng = np.random.RandomState(SEED)
noise = rng.normal(0, 0.01, x.shape).astype(np.float32)
x_noisy = x + noise

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].plot(t, x[LEAD_SHOW], color="black", lw=0.9)
axes[0].set_title(f"(a) Sinyal asli (lead {LEADS[LEAD_SHOW]})", fontsize=10, loc="left")
axes[1].plot(t, x_noisy[LEAD_SHOW], color="black", lw=0.9)
axes[1].set_title("(b) Setelah penambahan derau Gaussian (σ = 0,01)", fontsize=10, loc="left")
axes[1].set_ylim(axes[0].get_ylim())
axes[2].plot(t, noise[LEAD_SHOW], color="gray", lw=0.7)
axes[2].set_title("(c) Derau yang ditambahkan (skala sumbu berbeda)", fontsize=10, loc="left")
for ax in axes:
    ax.set_ylabel("satuan z")
    ax.grid(True, alpha=0.3)
axes[2].set_xlabel("Waktu (detik)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gambar_3_10_derau_gaussian.png"), dpi=300)
plt.close()


# ---------- 3. Gambar 3.11: penskalaan amplitudo & pergeseran waktu ----------
win = slice(0, 400)                      # tampilkan 4 detik pertama supaya jelas
tw = t[win]
sig_l = x[LEAD_SHOW]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(tw, sig_l[win], color="black", lw=1.1, label="Asli")
axes[0].plot(tw, (sig_l * 0.9)[win], color="gray", lw=1.0, ls="--", label="Faktor 0,9")
axes[0].plot(tw, (sig_l * 1.1)[win], color="gray", lw=1.0, ls=":", label="Faktor 1,1")
axes[0].set_title("(a) Penskalaan amplitudo", fontsize=10)

shift = int(x.shape[1] * 0.1)            # 10% panjang sinyal = 100 sampel = 1 detik
axes[1].plot(tw, sig_l[win], color="black", lw=1.1, label="Asli")
axes[1].plot(tw, np.roll(sig_l, shift)[win], color="gray", lw=1.0, ls="--",
             label=f"Geser +{shift/fs:.0f} detik")
axes[1].plot(tw, np.roll(sig_l, -shift)[win], color="gray", lw=1.0, ls=":",
             label=f"Geser -{shift/fs:.0f} detik")
axes[1].set_title("(b) Pergeseran waktu (melingkar)", fontsize=10)

for ax in axes:
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("satuan z")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gambar_3_11_scaling_timeshift.png"), dpi=300)
plt.close()


# ---------- 4. Gambar 3.12: lead dropout (paksa 2 lead ter-nol-kan) ----------
np.random.seed(SEED)
x_drop = x
while True:
    x_drop = lead_dropout(x, p_drop=1.0, max_leads=2)      # p=1.0 -> pasti aktif untuk ilustrasi
    dropped = [i for i in range(12) if np.allclose(x_drop[i], 0)]
    if len(dropped) == 2:
        break

fig, axes = plt.subplots(6, 2, figsize=(12, 9), sharex=True)
for i, ax in enumerate(axes.T.flatten()):   # kolom kiri: I-aVF, kolom kanan: V1-V6
    ax.plot(t, x_drop[i], color="black", lw=0.9)
    ax.set_ylabel(LEADS[i], rotation=0, labelpad=18, fontsize=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    if i in dropped:
        ax.set_facecolor("#DDDDDD")
        ax.text(0.99, 0.82, "di-nol-kan", transform=ax.transAxes,
                ha="right", fontsize=9)
axes[-1, 0].set_xlabel("Waktu (detik)")
axes[-1, 1].set_xlabel("Waktu (detik)")
fig.suptitle(f"Lead dropout: lead {', '.join(LEADS[i] for i in dropped)} di-nol-kan",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gambar_3_12_lead_dropout.png"), dpi=300)
plt.close()

print("Selesai. Gambar tersimpan di:", OUT_DIR)