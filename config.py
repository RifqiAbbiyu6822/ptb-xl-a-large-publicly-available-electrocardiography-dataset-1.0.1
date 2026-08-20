from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PTBXLConfig:
    # --- Lokasi dataset ---
    # Folder root PTB-XL hasil download dari PhysioNet, harus berisi:
    #   ptbxl_database.csv, scp_statements.csv, records100/, records500/
    ptbxl_root: str = r"C:\Users\imneo\Documents\Skripsi\data\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

    # --- Sinyal ---
    sampling_rate: int = 100          # pilih 100 atau 500 (Hz), sesuai folder records100/records500
    target_length_sec: float = 10.0   # semua rekaman PTB-XL ~10 detik, dipaksa seragam di sini
    n_leads: int = 12

    # --- Label target (multi-label, 5 diagnostic superclass PTB-XL) ---
    target_classes: List[str] = field(
        default_factory=lambda: ["NORM", "MI", "STTC", "CD", "HYP"]
    )

    # --- Filtering (baseline wander & noise removal) ---
    use_bandpass_filter: bool = True
    lowcut: float = 0.5
    highcut: float = 40.0
    filter_order: int = 4

    # --- Normalisasi (BARU: bisa pilih per-lead vs global) ---
    # "per_lead" -> versi lama, tiap lead di-zscore terpisah. Baik untuk
    #   stabilitas umum tapi MENGHAPUS rasio amplitudo antar-lead.
    # "global"   -> satu mean/std dari semua lead, MENJAGA rasio amplitudo
    #   antar-lead tetap utuh. Direkomendasikan sebagai default karena kelas
    #   HYP kriterianya berbasis perbandingan voltase antar-lead
    #   (Sokolow-Lyon, Cornell) yang hilang kalau pakai per_lead.
    normalize_mode: str = "global"

    # --- Crop / pad strategy supaya panjang sinyal selalu SAMA (stabil untuk batching) ---
    crop_mode_train: str = "random"   # random crop -> sekaligus jadi augmentasi ringan
    crop_mode_eval: str = "center"    # center crop -> deterministik untuk val/test

    # --- Augmentasi umum (khusus training, on-the-fly, tidak disimpan ke disk) ---
    augment_train: bool = True
    noise_std: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shift_max: float = 0.1            # fraksi panjang sinyal untuk random time-shift

    # --- Augmentasi lead-dropout (BARU, khusus training) ---
    use_lead_dropout: bool = True
    lead_dropout_prob: float = 0.15     # probabilitas augmentasi ini aktif per sample
    lead_dropout_max_leads: int = 2     # maksimum jumlah lead yang di-drop kalau aktif

    # --- Split (mengikuti rekomendasi resmi PTB-XL memakai kolom strat_fold) ---
    test_fold: int = 10
    val_fold: int = 9
    seed: int = 42

    @property
    def target_length(self) -> int:
        return int(self.target_length_sec * self.sampling_rate)
