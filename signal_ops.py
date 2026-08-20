
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(sig: np.ndarray, fs: float, lowcut: float = 0.5,
                     highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """Buang baseline wander (drift) & noise frekuensi tinggi per lead."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.999)  # jaga-jaga highcut tidak melebihi Nyquist
    b, a = butter(order, [low, high], btype="band")
    out = np.empty_like(sig)
    for lead in range(sig.shape[0]):
        out[lead] = filtfilt(b, a, sig[lead])
    return out


def zscore_normalize(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalisasi per-lead (zero mean, unit variance) -> skala stabil antar sample.
    PERINGATAN: cara ini menghapus rasio amplitudo ANTAR lead, karena tiap lead
    dipaksa unit-variance sendiri-sendiri. Untuk kelas yang kriterianya berbasis
    perbandingan voltase antar-lead (mis. HYP: Sokolow-Lyon = S_V1 + R_V5/V6),
    ini menghilangkan justru fitur paling diagnostik. Lihat zscore_normalize_global
    di bawah untuk alternatif yang menjaga informasi ini."""
    mean = sig.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True)
    return (sig - mean) / (std + eps)


def zscore_normalize_global(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalisasi GLOBAL per-sample (BARU, direkomendasikan sebagai default):
    satu mean & std dihitung dari SELURUH 12 lead sekaligus (bukan per-lead),
    lalu dipakai untuk menskalakan semua lead dengan faktor yang SAMA.

    Kenapa ini penting khusus untuk HYP: kriteria klinis hypertrophy (Sokolow-
    Lyon index, Cornell voltage, dll) adalah PERBANDINGAN amplitudo ANTAR lead
    (mis. S di V1 dibanding R di V5/V6). zscore_normalize() versi per-lead
    menghapus perbandingan ini karena tiap lead dipaksa unit-variance sendiri.
    zscore_normalize_global menjaga rasio amplitudo antar-lead tetap utuh --
    lead yang secara alami bervoltase besar (mis. V5/V6) tetap "terlihat" lebih
    besar dibanding lead bervoltase kecil (mis. V1) setelah normalisasi,
    persis seperti kondisi aslinya, hanya diskalakan sekali secara global
    supaya rentang nilai tetap stabil untuk training.

    Efek samping: sample-to-sample amplitude variability (mis. karena kontak
    elektroda beda-beda) tidak lagi diredam sekuat versi per-lead. Ini trade-off
    yang wajar -- kalau target_classes tidak termasuk HYP, versi per-lead lama
    kemungkinan masih lebih baik untuk stabilitas training.
    """
    mean = sig.mean()
    std = sig.std()
    return (sig - mean) / (std + eps)


def compute_hyp_voltage_features(sig: np.ndarray, lead_order=None) -> np.ndarray:
    """
    Fitur voltase eksplisit ala kriteria klinis HYP (proxy, BUKAN implementasi
    medis presisi -- tetap perlu sinyal RAW/belum di-zscore, satuan mV asli
    PTB-XL). Dipakai sebagai fitur tambahan (auxiliary input) yang dikonkat ke
    representasi CNN sebelum classifier head, supaya model tidak harus
    "menemukan sendiri" pola perbandingan amplitudo antar-lead dari nol lewat
    convolution -- fitur ini langsung memberi sinyal itu secara eksplisit.

    lead_order default mengikuti urutan standar PTB-XL 12-lead:
    ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    Mengembalikan array (3,): [sokolow_lyon_proxy, cornell_proxy, max_abs_voltage]
    """
    if lead_order is None:
        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]
    idx = {name: i for i, name in enumerate(lead_order)}

    def peak_amp(lead_name):
        # proxy sederhana: rentang amplitudo (max-min) di lead itu sebagai
        # pengganti pengukuran S/R-wave presisi (butuh QRS detection penuh
        # untuk versi klinis akurat)
        s = sig[idx[lead_name]]
        return float(s.max() - s.min())

    sokolow_lyon = peak_amp("V1") + max(peak_amp("V5"), peak_amp("V6"))
    cornell = peak_amp("aVL") + peak_amp("V3")
    max_abs_voltage = float(np.abs(sig).max())

    return np.array([sokolow_lyon, cornell, max_abs_voltage], dtype=np.float32)


def fix_length(sig: np.ndarray, target_len: int, mode: str = "center") -> np.ndarray:
    """
    Paksa panjang sinyal jadi persis target_len, dengan crop atau zero-pad.
    mode="random" -> crop/pad posisi acak (dipakai saat training, sekaligus augmentasi).
    mode="center" -> crop/pad simetris (dipakai saat val/test, deterministik).
    """
    n_leads, n_samples = sig.shape
    if n_samples == target_len:
        return sig

    if n_samples > target_len:
        if mode == "random":
            start = np.random.randint(0, n_samples - target_len + 1)
        else:
            start = (n_samples - target_len) // 2
        return sig[:, start:start + target_len]

    pad_total = target_len - n_samples
    if mode == "random":
        pad_left = np.random.randint(0, pad_total + 1)
    else:
        pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(sig, ((0, 0), (pad_left, pad_right)), mode="constant")


def augment_signal(sig: np.ndarray, noise_std: float = 0.01,
                    scale_range=(0.9, 1.1), shift_max: float = 0.1) -> np.ndarray:
    """
    Augmentasi ringan khusus training, dijalankan on-the-fly tiap kali sample
    diambil (bukan disimpan) supaya setiap epoch melihat variasi berbeda tanpa
    memperbesar dataset di disk.
    """
    sig = sig + np.random.normal(0, noise_std, sig.shape).astype(sig.dtype)

    scale = np.random.uniform(*scale_range)
    sig = sig * scale

    n_samples = sig.shape[1]
    max_shift = int(n_samples * shift_max)
    if max_shift > 0:
        shift = np.random.randint(-max_shift, max_shift + 1)
        sig = np.roll(sig, shift, axis=1)

    return sig


def lead_dropout(sig: np.ndarray, p_drop: float = 0.15, max_leads: int = 2) -> np.ndarray:
    """
    Augmentasi robustness antar-lead (BARU): dengan probabilitas p_drop,
    nol-kan 1..max_leads lead secara acak (simulasi lead lepas / noise berat
    yang memang kadang terjadi di rekaman klinis).

    Kenapa ini relevan untuk masalah imbalance/kelas sulit seperti HYP:
    kriteria diagnosis hypertrophy bergantung pada voltage & axis di lead
    spesifik (mis. aVL, aVF, V1-V6). Kalau model "curang" dan cuma belajar
    dari 1-2 lead termudah, generalisasinya jelek justru di kelas yang
    butuh informasi tersebar seperti ini. Dengan random drop lead saat
    training, model dipaksa membangun representasi yang lebih robust across
    semua 12 lead.
    """
    if np.random.rand() >= p_drop:
        return sig
    sig = sig.copy()
    n_leads = sig.shape[0]
    k = np.random.randint(1, max_leads + 1)
    drop_idx = np.random.choice(n_leads, size=min(k, n_leads), replace=False)
    sig[drop_idx, :] = 0.0
    return sig


def sanitize(sig: np.ndarray) -> np.ndarray:
    """Ganti NaN/Inf (kadang muncul di rekaman PTB-XL) supaya training tidak NaN."""
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig
