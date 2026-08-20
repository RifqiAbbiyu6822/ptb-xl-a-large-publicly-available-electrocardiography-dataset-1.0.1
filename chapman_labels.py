import ast
import pandas as pd


def _parse_superclass_string(raw) -> list:
    """Parse kolom superclasses ke list Python, robust terhadap beberapa
    format umum: literal list-string, dipisah ';'/','/'|' , atau NaN/kosong."""
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return raw
    raw = str(raw).strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except (ValueError, SyntaxError):
            pass
    for sep in [";", ",", "|"]:
        if sep in raw:
            return [x.strip().strip("'\"") for x in raw.split(sep) if x.strip()]
    return [raw] if raw else []


def load_chapman_metadata(chapman_csv: str, target_classes) -> pd.DataFrame:
    """
    Mengembalikan DataFrame dengan kolom: record_id, <target_classes> (0/1).
    Baris tanpa satupun target_classes DIBUANG (konsisten dengan labels.py PTB-XL).
    """
    df = pd.read_csv(chapman_csv)

    if "record_id" not in df.columns:
        raise ValueError(
            "chapman_csv harus punya kolom 'record_id'. Kolom yang ada: "
            f"{list(df.columns)}"
        )
    if "superclasses" not in df.columns:
        raise ValueError(
            "chapman_csv harus punya kolom 'superclasses'. Kolom yang ada: "
            f"{list(df.columns)}. Kalau CSV kamu masih pakai SNOMED-CT code "
            "mentah (kolom 'Dx'), lakukan mapping ke superclass dulu sebelum "
            "load_chapman_metadata() dipanggil."
        )

    df["superclass_list"] = df["superclasses"].apply(_parse_superclass_string)

    n_before = len(df)
    df = df[df["superclass_list"].apply(len) > 0].copy()
    n_after = len(df)
    print(f"[chapman_labels] Buang {n_before - n_after} record tanpa label target "
          f"({n_before} -> {n_after})")

    for c in target_classes:
        df[c] = df["superclass_list"].apply(lambda lst: 1 if c in lst else 0)

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
