import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")   # folder yang berisi labels.py dan signal_ops.py
from labels import load_ptbxl_metadata
from signal_ops import bandpass_filter, zscore_normalize, zscore_normalize_global

PTBXL_ROOT = r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
V1, V5 = 6, 10   # indeks lead V1 dan V5 pada urutan standar PTB-XL


def p2p(x):
    return float(x.max() - x.min())


df = load_ptbxl_metadata(PTBXL_ROOT, CLASSES)

# kandidat: rekaman HYP saja (single-label). Pilih yang rasio amplitudo V5/V1
# paling jelas supaya perbedaan kedua normalisasi mudah terlihat.
cand = df[(df["HYP"] == 1) & (df[CLASSES].sum(axis=1) == 1)].sample(60, random_state=42)

best = None
for ecg_id, row in cand.iterrows():
    sig, meta = wfdb.rdsamp(os.path.join(PTBXL_ROOT, row["filename_lr"]))
    filt = bandpass_filter(sig.T.astype(np.float32), meta["fs"], 0.5, 40.0, 4)
    ratio = p2p(filt[V5]) / max(p2p(filt[V1]), 1e-8)
    if best is None or ratio > best[0]:
        best = (ratio, ecg_id, filt, meta["fs"])

_, ecg_id, filt, fs = best
per_lead = zscore_normalize(filt)
global_n = zscore_normalize_global(filt)

t = np.arange(filt.shape[1]) / fs
panels = [
    ("(a) Sesudah penyaringan\n(sebelum normalisasi)", filt, "mV"),
    ("(b) Normalisasi per lead", per_lead, "satuan z"),
    ("(c) Normalisasi global", global_n, "satuan z"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, (title, x, unit) in zip(axes, panels):
    ax.plot(t, x[V1], color="gray", lw=0.9, label="V1")
    ax.plot(t, x[V5], color="black", lw=0.9, label="V5")
    r = p2p(x[V5]) / max(p2p(x[V1]), 1e-8)
    ax.set_title(f"{title}\nrasio amplitudo V5/V1 = {r:.2f}", fontsize=10)
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("gambar_3_9_normalisasi.png", dpi=300)
print("ecg_id:", ecg_id)