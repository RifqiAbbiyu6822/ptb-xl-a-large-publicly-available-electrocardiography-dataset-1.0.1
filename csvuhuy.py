"""
chapman_snomed_labels.py

Baca label diagnosis LANGSUNG dari file .hea (header WFDB) dataset
Chapman-Shaoxing / Chapman-Shaoxing-Ningbo ("ecg-arrhythmia" di PhysioNet),
lewat kode SNOMED-CT yang tertulis di baris komentar:

    #Dx: 426177001,164889003

pada tiap file .hea, lalu petakan ke 7 kelas target sesuai tabel yang diminta:

    Class Name        Chapman/Shaoxing Labels Included
    Normal            NORM, SB, SR, ST
    CD                1AVB, 2AVB2, AVB, AVNRT, AT, CAVB, CLBBB, IIAVBI, IVB,
                      JEB, JPT, Nonspecific BBB, PRIE, PRWP, PWC, SAAWR,
                      SVT, VEB, VET, VPB, VPE, WAVN, WPW
    HYP               ALS, ARS, CR, LVH, LVHV, RAH, RAVC, RVH
    MI                MILW
    STTC              STDD, STE, STTC, STTU, TTW, TWO
    A. Fib/Aflutter   AF, AFIB
    Other             ABI, APB, AQW, ERV, FQRS, LVQRSCL, LVQRSLL, PTW, UW, VB

Kode SNOMED-CT untuk tiap akronim diambil dari file resmi
`ConditionNames_SNOMED-CT.csv` milik dataset "A large scale 12-lead
electrocardiogram database for arrhythmia study" (Chapman-Shaoxing-Ningbo,
PhysioNet):
https://physionet.org/content/ecg-arrhythmia/1.0.0/ConditionNames_SNOMED-CT.csv

PENTING -- baca sebelum pakai ke data asli:
Akronim "NONSPECIFICBBB" dan "PRWP" TIDAK punya baris resmi di file SNOMED-CT
csv manapun yang saya temukan -- kemungkinan tidak pernah muncul sebagai kode
SNOMED tersendiri di file .hea dataset ini. "NORM" juga tidak punya kode
SNOMED sendiri (diwakili oleh SR/sinus rhythm). Akronim lain di tabel kelas
yang tadinya tidak ketemu (LVHV, RAVC, TTW, PTW) sudah diatasi lewat
verifikasi ulang ke dx_mapping resmi PhysioNet/CinC Challenge 2021 -- lihat
komentar di SYNONYM_TO_CANONICAL, sebagian berdasarkan catatan resmi
("kami skor kode X sama dengan kode Y"), sebagian lagi DUGAAN berdasarkan
pola penamaan (ditandai jelas "DUGAAN" di komentarnya) yang sebaiknya kamu
verifikasi manual kalau presisi tinggi diperlukan (mis. lewat
https://browser.ihtsdotools.org/). Kalau `--report_unmapped` (default aktif)
masih menunjukkan kode SNOMED tak dikenal yang sering muncul di data kamu,
cek dulu artinya lalu tambahkan ke SNOMED_TO_ACRONYM di bawah.

Cara pakai:
    python chapman_snomed_labels.py \
        --chapman_root /path/ke/WFDBRecords \
        --output_csv chapman_labels_snomed.csv

Folder didukung baik FLAT (semua .hea di satu folder, seperti asumsi
chapman_dataset.py di project ini) maupun NESTED (struktur resmi
WFDBRecords/xx/xx_yy/*.hea) -- script mencari file .hea secara rekursif.
"""

import argparse
import csv
import os
import re
import time
from collections import Counter

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class ProgressReporter:
    """
    UI sederhana untuk memantau progress di terminal.

    Kalau `tqdm` terpasang -> progress bar interaktif standar (persentase,
    kecepatan file/detik, ETA) PLUS live counter (jumlah per kelas & kode
    tak dikenal) ditampilkan di sisi kanan bar lewat set_postfix.

    Kalau `tqdm` TIDAK terpasang -> fallback cetak baris progress manual
    tiap `every` file (persentase, kecepatan, ETA, + counter yang sama),
    supaya tetap bisa dipantau tanpa dependency tambahan.
    """

    def __init__(self, total: int, desc: str = "Memproses", every: int = 500):
        self.total = total
        self.every = max(1, every)
        self._start = time.time()
        self._n = 0
        if _HAS_TQDM:
            self._bar = tqdm(total=total, desc=desc, unit="file")
        else:
            self._bar = None
            print(f"[{desc}] 0/{total} (0.0%)")

    def update(self, postfix: dict = None):
        self._n += 1
        if self._bar is not None:
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.update(1)
        elif self._n % self.every == 0 or self._n == self.total:
            elapsed = time.time() - self._start
            rate = self._n / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self._n) / rate if rate > 0 else float("inf")
            pct = 100 * self._n / self.total if self.total else 100.0
            extra = ("  " + "  ".join(f"{k}={v}" for k, v in postfix.items())) if postfix else ""
            print(f"  {self._n}/{self.total} ({pct:5.1f}%)  {rate:6.1f} file/s  "
                  f"ETA {eta:6.0f}s{extra}")

    def close(self):
        if self._bar is not None:
            self._bar.close()
        else:
            print(f"[selesai] {self._n}/{self.total} (100.0%)")

# ---------------------------------------------------------------------------
# 1) SNOMED-CT code -> akronim asli, dari ConditionNames_SNOMED-CT.csv RESMI
#    (dataset ecg-arrhythmia / Chapman-Shaoxing-Ningbo, PhysioNet)
# ---------------------------------------------------------------------------
SNOMED_TO_ACRONYM = {
    "270492004": "1AVB",
    "195042002": "2AVB",
    "54016002":  "2AVB1",
    "28189009":  "2AVB2",
    "27885002":  "3AVB",
    "251173003": "ABI",
    "39732003":  "ALS",
    "284470004": "APB",
    "164917005": "AQW",
    "47665007":  "ARS",
    "233917008": "AVB",
    "251199005": "CCR",
    "251198002": "CR",
    "428417006": "ERV",
    "164942001": "FQRS",
    "698252002": "IVB",     # kode sama dgn "IDC" di tabel resmi
    "426995002": "JEB",
    "251164006": "JPT",
    "164909002": "LBBB",    # kode sama dgn "LBBBB"/"LFBBB" di tabel resmi
    "164873001": "LVH",
    "251146004": "LVQRSAL",
    "251148003": "LVQRSCL",
    "251147008": "LVQRSLL",
    "164865005": "MI",      # kode sama dgn "MIBW"/"MIFW"/"MILW"/"MISW"
    "164947007": "PRIE",
    "164912004": "PWC",
    "111975006": "QTIE",
    "446358003": "RAH",
    "59118001":  "RBBB",
    "89792004":  "RVH",
    "429622005": "STDD",
    "164930006": "STE",
    "428750005": "STTC",
    "164931005": "STTU",
    "164934002": "TWC",
    "59931005":  "TWO",
    "164937009": "UW",
    "11157007":  "VB",
    "75532003":  "VEB",
    "13640000":  "VFW",
    "17338001":  "VPB",
    "195060002": "VPE",
    "251180001": "VET",
    "195101003": "WAVN",    # kode sama dgn "SAAWR" di tabel resmi
    "74390002":  "WPW",
    "426177001": "SB",
    "426783006": "SR",
    "164889003": "AFIB",
    "427084000": "ST",
    "164890007": "AF",
    "427393009": "SA",
    "426761007": "SVT",
    "713422000": "AT",
    "233896004": "AVNRT",
    "233897008": "AVRT",
}

# Kode tambahan (di luar 55 kode "resmi" Chapman) yang TERNYATA banyak muncul
# di data asli, karena dataset "ecg-arrhythmia" (WFDBRecords) yang kamu pakai
# adalah gabungan 45.152 rekaman Chapman-Shaoxing + Ningbo, dan memakai kode
# SNOMED yang lebih luas -- sama seperti dx_mapping_scored.csv /
# dx_mapping_unscored.csv resmi PhysioNet/CinC Challenge 2021:
# https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv
# https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_unscored.csv
SNOMED_TO_ACRONYM.update({
    "55827005":  "LVHV",   # left ventricular high voltage
    "733534002": "CLBBB",  # complete left bundle branch block (RESMI: skor sama dgn 164909002/LBBB)
    "713427006": "CRBBB",  # complete right bundle branch block (RESMI: skor sama dgn 59118001/RBBB)
    "713426002": "IRBBB",  # incomplete right bundle branch block
    "445118002": "LAnFB",  # left anterior fascicular block
    "6374002":   "BBB",    # bundle branch block (generik, tidak spesifik sisi)
    "55930002":  "STC",    # s t changes (ST changes, generik)
    "10370003":  "PR",     # pacing rhythm
    "251223006": "TPW",    # tall p wave
    "106068003": "ARH",    # atrial rhythm
    "29320008":  "AVJR",   # atrioventricular junctional rhythm
    "425856008": "PVT",    # paroxysmal ventricular tachycardia
    "251205003": "PPW",    # prolonged P wave
    "81898007":  "VESR",   # ventricular escape rhythm
    "251170000": "BPAC",   # blocked premature atrial contraction
    "50799005":  "AVD",    # atrioventricular dissociation
    "164896001": "VF",     # ventricular fibrillation
    "54329005":  "ANMI",   # anterior myocardial infarction
    "57054005":  "AMI",    # acute myocardial infarction
    "67751000119106": "RAHV",  # right atrial high voltage
    "5609005":   "SARR",   # sinus arrest
    "426648003": "JTACH",  # junctional tachycardia
    "49578007":  "SPRI",   # shortened pr interval
    "251187003": "AED",    # atrial escape beat
    "251166008": "AVNRT",  # AV node reentrant tachycardia (kode SNOMED lain utk konsep yg sama dgn 233896004)
    "233892002": "AAR",    # accelerated atrial escape rhythm
    "61277005":  "AIVR",   # accelerated idioventricular rhythm
    "426664006": "AJR",    # accelerated junctional rhythm
    "61721007":  "CVCL",   # clockwise/counterclockwise vectorcardiographic loop
    "427172004": "PVC",    # premature ventricular contractions (RESMI: skor sama dgn 17338001/VPB)
})
# 6180003 : sengaja TIDAK ditambahkan -- belum berhasil dipastikan artinya
# dari dokumentasi resmi manapun. Kalau sering muncul di data kamu, cek manual
# lewat https://browser.ihtsdotools.org/ lalu tambahkan sendiri ke dict di atas.

# Akronim yang berbagi SNOMED code sama dengan akronim "kanonik" yang dipakai
# di CLASS_MAP di bawah (lihat komentar di SNOMED_TO_ACRONYM). Dipetakan ke
# akronim kanonik supaya tetap kena kelas yang benar walau nama beda-beda.
SYNONYM_TO_CANONICAL = {
    "MIBW": "MILW", "MIFW": "MILW", "MISW": "MILW", "MI": "MILW",
    "IDC": "IVB",
    "2AVB1": "IIAVBI",  # dugaan: "2 AVB Type One" == "IIAVBI" di tabel kelas
    "3AVB": "CAVB",     # dugaan: "3 degree AV block"/"complete heart block" == "complete AVB" (CAVB)
    "LBBB": "CLBBB",    # RESMI (catatan dx_mapping): 164909002(LBBB) diskor sama dgn 733534002(CLBBB)
    "SAAWR": "WAVN",    # kode SNOMED memang identik (195101003)

    # --- Tambahan hasil verifikasi ulang pakai dx_mapping resmi Challenge 2021 ---
    "STC": "STTC",      # "ST changes" (55930002) = varian generik dari "ST-T Change" (STTC)
    "TWC": "STTC",      # "T wave Change" (164934002): abnormalitas gelombang-T generik, masuk
                         # payung STTC. Kemungkinan besar inilah yang dimaksud "TTW" di tabel
                         # kelas (typo/variasi nama -- tidak ada kode resmi bernama persis "TTW")
    "ANMI": "MILW",     # "anterior myocardial infarction" (54329005) -- subtipe lokasi MI lain,
                         # sama seperti MILW (myocardial infarction lower wall), semua masuk MI,
                         # konsisten dgn definisi superclass "MI" PTB-XL yg dipakai project ini
    "AMI": "MILW",      # "acute myocardial infarction" (57054005) -- idem, subtipe MI -> kelas MI
    "PVC": "VPB",       # RESMI (catatan dx_mapping): 427172004(PVC) diskor sama dgn 17338001(VPB)
    "RAHV": "RAVC",     # DUGAAN: "right atrial high voltage" (67751000119106) kemungkinan besar
                         # yang dimaksud "RAVC" di tabel kelas -- pasangan RA dari LVH/LVHV
                         # (LV punya pasangan hypertrophy+high-voltage, RA seharusnya juga)
    "TPW": "PTW",       # DUGAAN: "tall p wave" (251223006) kemungkinan "PTW" di tabel kelas,
                         # hurufnya tertukar urutan (typo) jadi "TPW"
}

# Akronim di tabel kelas yang TIDAK ada baris resminya di ConditionNames_SNOMED-CT.csv
# dan TIDAK dianggap sinonim di atas -- didokumentasikan saja, isi manual kalau
# kamu menemukan kode SNOMED yang benar di data kamu.
# (LVHV, RAVC, TTW, PTW sudah teratasi lewat SYNONYM_TO_CANONICAL di atas
#  setelah verifikasi ulang -- lihat komentar masing-masing di sana)
ACRONYM_NOT_IN_OFFICIAL_TABLE = {
    "NONSPECIFICBBB": None,  # "Nonspecific BBB"
    "PRWP": None,
    "NORM": None,            # tidak ada kode SNOMED "normal" tersendiri;
                              # NORM di dataset ini diwakili oleh SR (sinus rhythm)
}

# ---------------------------------------------------------------------------
# 2) akronim -> kelas target, PERSIS sesuai tabel yang diminta
# ---------------------------------------------------------------------------
CLASS_MAP = {
    "NORM": ["NORM", "SB", "SR", "ST"],
    "CD": ["1AVB", "2AVB2", "AVB", "AVNRT", "AT", "CAVB", "CLBBB", "IIAVBI",
           "IVB", "JEB", "JPT", "NONSPECIFICBBB", "PRIE", "PRWP", "PWC",
           "SAAWR", "SVT", "VEB", "VET", "VPB", "VPE", "WAVN", "WPW"],
    "HYP": ["ALS", "ARS", "CR", "LVH", "LVHV", "RAH", "RAVC", "RVH"],
    "MI": ["MILW"],
    "STTC": ["STDD", "STE", "STTC", "STTU", "TTW", "TWO"],
    "AFIB": ["AF", "AFIB"],           # "A. Fib/Aflutter"
    "OTHER": ["ABI", "APB", "AQW", "ERV", "FQRS", "LVQRSCL", "LVQRSLL",
              "PTW", "UW", "VB"],
}
TARGET_CLASSES = ["NORM", "CD", "HYP", "MI", "STTC", "AFIB", "OTHER"]

ACRONYM_TO_CLASS = {ac: cls for cls, acs in CLASS_MAP.items() for ac in acs}

DX_LINE_RE = re.compile(r"^#\s*Dx\s*:\s*(.*)$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description="Baca kode SNOMED-CT dari .hea Chapman-Shaoxing/Ningbo, "
                     "petakan ke kelas target, simpan sebagai CSV.")
    p.add_argument("--chapman_root", type=str, required=True,
                    help="Folder root berisi file .hea (+.mat), flat atau nested.")
    p.add_argument("--output_csv", type=str, default="chapman_labels_snomed.csv",
                    help="Path CSV output.")
    p.add_argument("--require_mat", action="store_true",
                    help="Kalau diaktifkan, record TANPA file .mat pasangannya "
                         "(nama sama, ekstensi .mat) akan dilewati.")
    p.add_argument("--report_unmapped", dest="report_unmapped",
                    action="store_true", default=True,
                    help="Cetak ringkasan kode SNOMED yang tidak dikenal (default: ON).")
    p.add_argument("--no_report_unmapped", dest="report_unmapped", action="store_false")
    p.add_argument("--progress_every", type=int, default=500,
                    help="[fallback tanpa tqdm] cetak progress tiap N file. "
                         "Diabaikan kalau tqdm terpasang (progress bar dipakai).")
    return p.parse_args()


def find_hea_files(root: str):
    """Cari semua file .hea secara rekursif (mendukung flat maupun nested)."""
    hea_files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".hea"):
                hea_files.append(os.path.join(dirpath, fn))
    return sorted(hea_files)


def parse_dx_codes(hea_path: str) -> list:
    """Ambil daftar kode SNOMED-CT mentah dari baris '#Dx: ...' di file .hea."""
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = DX_LINE_RE.match(line)
            if m:
                raw = m.group(1)
                return [c.strip() for c in re.split(r"[,\s]+", raw) if c.strip()]
    return []


def map_codes_to_classes(codes: list):
    """
    SNOMED codes (1 record, bisa multi-kode) -> (set kelas aktif, akronim
    yang berhasil di-resolve, kode SNOMED yang tidak dikenal).
    Aturan NORM sama seperti chapman_labels.py: NORM hanya diberikan kalau
    TIDAK ada kelas abnormal lain sama sekali.
    """
    acronyms = []
    unmapped_codes = []
    for code in codes:
        acr = SNOMED_TO_ACRONYM.get(code)
        if acr is None:
            unmapped_codes.append(code)
        else:
            acronyms.append(SYNONYM_TO_CANONICAL.get(acr, acr))

    classes = {ACRONYM_TO_CLASS[a] for a in acronyms if a in ACRONYM_TO_CLASS}
    unmapped_acronyms = [a for a in acronyms if a not in ACRONYM_TO_CLASS]

    abnormal = classes - {"NORM"}
    if abnormal:
        classes = abnormal

    return classes, acronyms, unmapped_codes, unmapped_acronyms


def main():
    args = parse_args()

    hea_files = find_hea_files(args.chapman_root)
    print(f"[chapman_snomed_labels] Ditemukan {len(hea_files)} file .hea di {args.chapman_root}")
    if not hea_files:
        print("[chapman_snomed_labels] Tidak ada file .hea ditemukan, berhenti.")
        return

    rows = []
    class_counts = Counter()
    unmapped_code_counter = Counter()
    unmapped_acronym_counter = Counter()
    n_no_dx = 0
    n_skipped_no_mat = 0

    progress = ProgressReporter(len(hea_files), desc="Baca .hea",
                                 every=args.progress_every)
    try:
        for hea_path in hea_files:
            record_id = os.path.splitext(os.path.basename(hea_path))[0]

            if args.require_mat:
                mat_path = os.path.join(os.path.dirname(hea_path), record_id + ".mat")
                if not os.path.exists(mat_path):
                    n_skipped_no_mat += 1
                    progress.update({"skip": n_skipped_no_mat})
                    continue

            codes = parse_dx_codes(hea_path)
            if not codes:
                n_no_dx += 1

            classes, acronyms, unmapped_codes, unmapped_acronyms = map_codes_to_classes(codes)
            unmapped_code_counter.update(unmapped_codes)
            unmapped_acronym_counter.update(unmapped_acronyms)

            row = {
                "record_id": record_id,
                "hea_path": hea_path,
                "dx_codes_raw": ";".join(codes),
                "dx_acronyms_resolved": ";".join(acronyms),
            }
            for c in TARGET_CLASSES:
                row[c] = 1 if c in classes else 0
            rows.append(row)

            for c in classes:
                class_counts[c] += 1

            # live counter: kelas paling relevan (NORM/HYP/MI, sering jadi minoritas)
            # + jumlah kode SNOMED tak dikenal sejauh ini
            progress.update({
                "NORM": class_counts["NORM"],
                "HYP": class_counts["HYP"],
                "MI": class_counts["MI"],
                "unk": sum(unmapped_code_counter.values()),
            })
    finally:
        progress.close()

    fieldnames = ["record_id", "hea_path", "dx_codes_raw", "dx_acronyms_resolved"] + TARGET_CLASSES
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_no_class = sum(1 for r in rows if all(r[c] == 0 for c in TARGET_CLASSES))

    print(f"\n[chapman_snomed_labels] Selesai. {len(rows)} record ditulis ke {args.output_csv}")
    if args.require_mat:
        print(f"  Dilewati (tidak ada .mat pasangan) : {n_skipped_no_mat}")
    print(f"  Record tanpa baris '#Dx:' sama sekali : {n_no_dx}")
    print(f"  Record tanpa kelas target aktif       : {n_no_class}")

    print("\n  Distribusi kelas (per record, boleh multi-label):")
    for c in TARGET_CLASSES:
        print(f"    {c:6s}: {class_counts[c]}")

    if args.report_unmapped:
        if unmapped_code_counter:
            print("\n  [PERINGATAN] Kode SNOMED yang TIDAK DIKENAL (belum ada di "
                  "SNOMED_TO_ACRONYM), diabaikan dari label -- cek manual & "
                  "tambahkan ke script kalau relevan:")
            for code, cnt in unmapped_code_counter.most_common(30):
                print(f"    {code}: muncul {cnt}x")
        if unmapped_acronym_counter:
            print("\n  [PERINGATAN] Akronim berhasil di-resolve dari SNOMED tapi "
                  "TIDAK ADA di CLASS_MAP manapun (diabaikan dari label):")
            for acr, cnt in unmapped_acronym_counter.most_common(30):
                print(f"    {acr}: muncul {cnt}x")
        if not unmapped_code_counter and not unmapped_acronym_counter:
            print("\n  Semua kode SNOMED yang ditemukan berhasil dipetakan ke kelas target.")


if __name__ == "__main__":
    main()