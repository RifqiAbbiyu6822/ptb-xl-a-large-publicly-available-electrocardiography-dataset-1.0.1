"""
chapman_dataset.py
Loader sinyal Chapman-Shaoxing (.mat + .hea, format WFDB, satu folder,
nama file = record_id) + PyTorch Dataset yang menerapkan preprocessing
ON-THE-FLY YANG SAMA PERSIS dengan PTBXLDataset (bandpass filter, fix_length,
normalisasi, augmentasi), supaya sample dari kedua dataset bisa digabung
lewat ConcatDataset tanpa distribution shift dari preprocessing yang beda.

PERINGATAN PENTING (unit fisik / mV):
  Kita sudah tahu (lihat signal_ops.zscore_normalize_global) bahwa untuk HYP,
  rasio & magnitude amplitudo ANTAR-LEAD itu informatif secara klinis. Ini
  artinya konversi ADC -> unit fisik (mV) di sini HARUS BENAR -- kalau tidak,
  data boost dari Chapman bisa punya skala amplitudo yang tidak konsisten
  dengan PTB-XL (yang sudah di unit fisik lewat wfdb.rdsamp), dan MENCEMARI
  training alih-alih membantu, apalagi untuk kelas yang paling sensitif
  terhadap ini.

  -> SELALU jalankan sanity_check_amplitude() di bawah dan bandingkan manual
     dengan statistik amplitudo PTB-XL SEBELUM training sungguhan pakai data
     boost ini. Kalau skalanya beda jauh (>10x), curigai gain/baseline di
     .hea salah di-parse (format .hea vendor kadang punya variasi kecil).
"""

import os
import re
from math import gcd

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from config import PTBXLConfig
from signal_ops import (
    bandpass_filter, zscore_normalize, zscore_normalize_global, fix_length,
    augment_signal, lead_dropout, sanitize,
)

STANDARD_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
                        "V1", "V2", "V3", "V4", "V5", "V6"]


def _parse_signal_line(line: str) -> dict:
    """
    Parse 1 baris deskripsi sinyal di file .hea (format WFDB standar):
      <file> <format> <gain>(<baseline>)/<units> <adc_res> <adc_zero> ... <nama_lead>
    Nama lead diasumsikan token TERAKHIR di baris (konvensi WFDB umum).
    """
    parts = line.split()
    gain_field = parts[2]
    adc_zero = float(parts[4]) if len(parts) > 4 else 0.0
    lead_name = parts[-1]

    m = re.match(r"([\-\d\.]+)(\(([\-\d\.]+)\))?/(\w+)", gain_field)
    if m:
        gain = float(m.group(1))
        baseline = float(m.group(3)) if m.group(3) is not None else adc_zero
        units = m.group(4)
    else:
        gain, baseline, units = 1.0, adc_zero, "mV"

    return {"lead_name": lead_name, "gain": gain if gain != 0 else 1.0,
            "baseline": baseline, "units": units}


def parse_hea(hea_path: str) -> dict:
    """Parse file .hea WFDB secara manual (header line + N baris sinyal)."""
    with open(hea_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    header = lines[0].split()
    record_name = header[0]
    n_sig = int(header[1])
    fs = float(header[2])
    n_samples = int(header[3]) if len(header) > 3 else None

    signal_specs = [_parse_signal_line(lines[i]) for i in range(1, 1 + n_sig)]

    return {"record_name": record_name, "n_sig": n_sig, "fs": fs,
            "n_samples": n_samples, "signal_specs": signal_specs}


def load_chapman_record(record_id: str, chapman_root: str) -> tuple:
    """
    Baca 1 record Chapman (.hea + .mat). Return:
      signal : (n_leads, n_samples) float32, SUDAH dikonversi ke unit fisik (mV)
               dan SUDAH di-reorder ke STANDARD_LEAD_ORDER
      fs     : sampling rate asli file (Hz)
    """
    hea_path = os.path.join(chapman_root, record_id + ".hea")
    mat_path = os.path.join(chapman_root, record_id + ".mat")

    meta = parse_hea(hea_path)
    mat = loadmat(mat_path)
    sig_key = "val" if "val" in mat else next(k for k in mat.keys() if not k.startswith("__"))
    raw = mat[sig_key].astype(np.float64)
    if raw.shape[0] != meta["n_sig"] and raw.shape[1] == meta["n_sig"]:
        raw = raw.T

    # ADC -> unit fisik (mV), per-lead pakai gain/baseline dari .hea
    physical = np.empty_like(raw, dtype=np.float32)
    lead_names = []
    for i, spec in enumerate(meta["signal_specs"]):
        physical[i] = (raw[i] - spec["baseline"]) / spec["gain"]
        lead_names.append(spec["lead_name"])

    idx_map = {name: i for i, name in enumerate(lead_names)}
    missing = [l for l in STANDARD_LEAD_ORDER if l not in idx_map]
    if missing:
        raise ValueError(f"[chapman_dataset] Record {record_id}: lead hilang di header: {missing} "
                          f"(lead tersedia: {lead_names})")
    order_idx = [idx_map[l] for l in STANDARD_LEAD_ORDER]
    signal = physical[order_idx, :]

    return signal, meta["fs"]


def resample_signal(signal: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Resample (n_leads, n_samples) dari fs_from ke fs_to pakai polyphase filter (exact ratio)."""
    if fs_from == fs_to:
        return signal.astype(np.float32)
    g = gcd(int(fs_from), int(fs_to))
    up, down = int(fs_to) // g, int(fs_from) // g
    return resample_poly(signal, up, down, axis=1).astype(np.float32)


def sanity_check_amplitude(chapman_root: str, sample_record_ids: list) -> None:
    """
    WAJIB dijalankan sekali secara manual sebelum training sungguhan dengan
    data boost ini. Cetak statistik amplitudo beberapa sample Chapman untuk
    dibandingkan manual dengan skala PTB-XL (biasanya std per-lead di kisaran
    puluhan mikrovolt s.d. beberapa mV, mean mendekati 0). Kalau angkanya beda
    10x-100x lipat dari PTB-XL, gain/baseline kemungkinan salah parse --
    JANGAN training dulu sebelum ini dicek.
    """
    print("[sanity_check_amplitude] Statistik amplitudo sample Chapman "
          "(bandingkan manual dengan std/mean sinyal PTB-XL mentah):")
    for rid in sample_record_ids[:5]:
        signal, fs = load_chapman_record(rid, chapman_root)
        print(f"  {rid}: fs={fs}Hz  shape={signal.shape}  "
              f"mean={signal.mean():.4f}  std={signal.std():.4f}  "
              f"min={signal.min():.4f}  max={signal.max():.4f}")


class ChapmanBoostDataset(Dataset):
    """
    Dataset tambahan (boost) dari Chapman-Shaoxing, HANYA untuk train set
    (lihat dataset.py: build_dataloaders(..., boost_dataset=...)). Preprocessing
    identik dengan PTBXLDataset mode="train" supaya distribusinya konsisten
    setelah digabung lewat ConcatDataset.
    """

    def __init__(self, df: pd.DataFrame, config: PTBXLConfig, chapman_root: str):
        """df: hasil chapman_labels.filter_boost_class() -- harus punya kolom
        'record_id' + kolom config.target_classes (0/1)."""
        self.df = df.reset_index(drop=True)
        self.config = config
        self.chapman_root = chapman_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        signal, fs = load_chapman_record(record_id, self.chapman_root)
        signal = sanitize(signal)

        if self.config.use_bandpass_filter:
            signal = bandpass_filter(signal, fs, self.config.lowcut,
                                      self.config.highcut, self.config.filter_order)

        signal = resample_signal(signal, fs_from=fs, fs_to=self.config.sampling_rate)

        # boost dataset cuma dipakai untuk training -> selalu mode "train"
        signal = fix_length(signal, self.config.target_length, mode=self.config.crop_mode_train)

        signal = zscore_normalize_global(signal) if self.config.normalize_mode == "global" \
            else zscore_normalize(signal)

        if self.config.augment_train:
            signal = augment_signal(signal, self.config.noise_std,
                                     self.config.scale_range, self.config.shift_max)
            if self.config.use_lead_dropout:
                signal = lead_dropout(signal, self.config.lead_dropout_prob,
                                       self.config.lead_dropout_max_leads)

        signal = sanitize(signal)

        label = row[self.config.target_classes].values.astype(np.float32)

        signal_t = torch.from_numpy(np.ascontiguousarray(signal)).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).float()
        return signal_t, label_t
