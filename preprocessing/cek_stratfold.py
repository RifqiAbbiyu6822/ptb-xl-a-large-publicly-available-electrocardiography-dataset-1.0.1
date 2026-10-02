import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from labels import load_ptbxl_metadata
from signal_ops import bandpass_filter

PTBXL_ROOT = r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_IDX, LEAD_NAME = 1, "II"   # lead II, jelas untuk lihat QRS dan baseline

df = load_ptbxl_metadata(PTBXL_ROOT, CLASSES)

# pilih rekaman yang ada baseline drift-nya supaya efek filter kelihatan
cand = df[df["baseline_drift"].notna() & (df["baseline_drift"].astype(str).str.strip() != "")]
row = cand.sample(1, random_state=42).iloc[0]
ecg_id = cand.sample(1, random_state=42).index[0]

sig, meta = wfdb.rdsamp(os.path.join(PTBXL_ROOT, row["filename_lr"]))
fs = meta["fs"]
raw = sig.T.astype(np.float32)                       # (12, 1000)
filt = bandpass_filter(raw, fs, 0.5, 40.0, 4)

t = np.arange(raw.shape[1]) / fs
fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)

axes[0].plot(t, raw[LEAD_IDX], color="black", lw=0.9)
axes[0].set_title(f"(a) Sebelum penyaringan (lead {LEAD_NAME})", fontsize=10, loc="left")

axes[1].plot(t, filt[LEAD_IDX], color="black", lw=0.9)
axes[1].set_title("(b) Sesudah penyaringan bandpass 0,5-40 Hz", fontsize=10, loc="left")

axes[2].plot(t, raw[LEAD_IDX], color="gray", lw=0.8, label="Sebelum")
axes[2].plot(t, filt[LEAD_IDX], color="black", lw=0.9, label="Sesudah")
axes[2].set_title("(c) Perbandingan (tumpang tindih)", fontsize=10, loc="left")
axes[2].legend(loc="upper right", fontsize=9)

for ax in axes:
    ax.set_ylabel("mV")
    ax.grid(True, alpha=0.3)
axes[2].set_xlabel("Waktu (detik)")

plt.tight_layout()
plt.savefig("gambar_3_8_filter.png", dpi=300)
print("ecg_id:", ecg_id, "| baseline_drift:", row["baseline_drift"])