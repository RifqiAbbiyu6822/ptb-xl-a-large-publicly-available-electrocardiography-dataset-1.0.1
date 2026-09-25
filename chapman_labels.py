import pandas as pd

# Kolom yang WAJIB ada di CSV metadata Chapman format baru (hasil mapping
# SNOMED-CT -> superclass sudah dilakukan sebelumnya di luar pipeline ini).
REQUIRED_BASE_COLUMNS = ["record_id", "hea_path"]


def load_chapman_metadata(chapman_csv: str, target_classes) -> pd.DataFrame:
    """
    Load metadata Chapman-Shaoxing dari CSV format BARU
    (contoh: chapman_labels_snomed.csv).

    Format CSV yang diharapkan (beda dari versi lama yang pakai kolom
    'superclasses' berisi string list):

        record_id, hea_path, dx_codes_raw, dx_acronyms_resolved,
        NORM, CD, HYP, MI, STTC, AFIB, OTHER

    - record_id            : identifier unik record (nama file tanpa
                              ekstensi, mis. 'JS00001').
    - hea_path              : path menuju file .hea record tsb SEPERTI YANG
                              TERCATAT saat CSV dibuat -- pada
                              chapman_labels_snomed.csv ini berupa path
                              ABSOLUT Windows dari mesin pembuat CSV (mis.
                              'C:\\Users\\imneo\\...\\WFDB_ShaoxingUniv\\JS00001.hea'),
                              JADI TIDAK PORTABLE apa adanya. chapman_dataset.py
                              hanya mengambil NAMA FILE dari kolom ini (basename,
                              robust terhadap '\\' dan '/'), lalu menggabungkannya
                              dengan --chapman_root milik pemanggil saat ini --
                              karena struktur folder Chapman-Shaoxing di sini
                              flat (satu folder, basename selalu persis
                              '<record_id>.hea'). File .mat diasumsikan ada di
                              folder chapman_root yang SAMA, basename SAMA,
                              cuma beda ekstensi ('.mat').
        dx_codes_raw, dx_acronyms_resolved : metadata tambahan (kode SNOMED-CT
                              mentah & hasil resolve ke akronim), tidak dipakai
                              langsung oleh training tapi berguna untuk audit/
                              debugging mapping label.
    - <setiap kelas di target_classes> : kolom 0/1 (int) yang menandakan kelas
                              itu aktif untuk record tsb. Mapping SNOMED-CT ->
                              superclass SUDAH dilakukan saat CSV ini dibuat,
                              jadi tidak ada parsing string list lagi di sini
                              (beda dengan versi lama yang punya kolom
                              'superclasses'). Kolom kelas lain yang tidak ada
                              di target_classes (mis. AFIB, OTHER kalau
                              target_classes cuma 5 kelas PTB-XL) boleh ada di
                              CSV dan akan diabaikan.

    Baris yang tidak punya satupun target_classes aktif DIBUANG, konsisten
    dengan perilaku labels.py untuk PTB-XL.
    """
    df = pd.read_csv(chapman_csv)

    missing_base = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"chapman_csv harus punya kolom {REQUIRED_BASE_COLUMNS}. "
            f"Kolom hilang: {missing_base}. Kolom yang ada: {list(df.columns)}"
        )

    missing_classes = [c for c in target_classes if c not in df.columns]
    if missing_classes:
        raise ValueError(
            f"chapman_csv tidak punya kolom label 0/1 untuk kelas: {missing_classes}. "
            f"Kolom yang ada: {list(df.columns)}. CSV ini harus sudah melalui mapping "
            "SNOMED-CT -> superclass (kolom 0/1 per kelas seperti NORM/CD/HYP/MI/STTC), "
            "bukan CSV Dx code SNOMED mentah tanpa mapping."
        )

    # pastikan label berupa int 0/1 bersih (jaga-jaga ada NaN/float dari CSV)
    for c in target_classes:
        df[c] = df[c].fillna(0).astype(int)

    n_before = len(df)
    df = df[df[target_classes].sum(axis=1) > 0].copy()
    n_after = len(df)
    print(f"[chapman_labels] Buang {n_before - n_after} record tanpa label target "
          f"({n_before} -> {n_after})")

    return df


def compute_boost_target(train_df: pd.DataFrame, target_classes, boost_class: str) -> int:
    """
    Hitung otomatis berapa record tambahan dibutuhkan supaya jumlah sample
    boost_class di train set sejajar dengan RATA-RATA jumlah kelas lain.
    Return 0 kalau boost_class sudah >= rata-rata (tidak perlu boost).
    """
    counts = {c: int(train_df[c].sum()) for c in target_classes}
    other = [v for c, v in counts.items() if c != boost_class]
    if not other:
        return 0
    avg_other = sum(other) / len(other)
    needed = max(0, int(round(avg_other - counts[boost_class])))
    return needed


def filter_boost_class(df: pd.DataFrame, boost_class: str, boost_target: int,
                        seed: int = 42) -> pd.DataFrame:
    """
    Ambil subset record Chapman yang punya label boost_class aktif, dibatasi
    sampai boost_target record (random sample kalau tersedia lebih banyak).
    Kalau tersedia LEBIH SEDIKIT dari boost_target, ambil semua yang ada
    (tidak error, cuma boost-nya jadi tidak penuh -> dicetak sebagai info).

    Kolom 'hea_path' dan 'record_id' ikut terbawa di subset hasil filter,
    dipakai belakangan oleh ChapmanBoostDataset untuk membaca file sinyal.
    """
    if boost_class not in df.columns:
        raise ValueError(f"boost_class '{boost_class}' tidak ada di kolom Chapman metadata")

    subset = df[df[boost_class] == 1].copy()
    n_available = len(subset)

    if n_available == 0:
        print(f"[chapman_labels] PERINGATAN: tidak ada record Chapman dengan label {boost_class}")
        return subset

    if boost_target <= 0:
        print(f"[chapman_labels] boost_target=0, tidak ada data Chapman yang ditambahkan")
        return subset.iloc[0:0]

    if n_available > boost_target:
        subset = subset.sample(n=boost_target, random_state=seed)
    elif n_available < boost_target:
        print(f"[chapman_labels] PERINGATAN: hanya {n_available} record {boost_class} tersedia "
              f"di Chapman, kurang dari target {boost_target}. Semua dipakai.")

    print(f"[chapman_labels] Boost kelas {boost_class}: {len(subset)}/{n_available} "
          f"record Chapman dipakai (target={boost_target})")
    return subset.reset_index(drop=True)