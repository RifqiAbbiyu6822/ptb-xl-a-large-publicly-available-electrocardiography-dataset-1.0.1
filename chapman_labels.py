import ast
import re
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# Label mapping Chapman-Shaoxing -> superclass (sesuai tabel Label Mappings)
# ---------------------------------------------------------------------------
CHAPMAN_LABEL_MAP = {
    "NORM": ["NORM", "SB", "SR", "ST"],
    "CD": ["1AVB", "2AVB2", "AVB", "AVNRT", "AT", "CAVB", "CLBBB", "IIAVBI",
           "IVB", "JEB", "JPT", "NONSPECIFICBBB", "PRIE", "PRWP", "PWC",
           "SAAWR", "SVT", "VEB", "VET", "VPB", "VPE", "WAVN", "WPW"],
    "HYP": ["ALS", "ARS", "CR", "LVH", "LVHV", "RAH", "RAVC", "RVH"],
    "MI": ["MILW"],
    "STTC": ["STDD", "STE", "STTC", "STTU", "TTW", "TWO"],
    "AFIB": ["AF", "AFIB"],  # A. Fib/Aflutter: BUKAN target model
    "OTHER": ["ABI", "APB", "AQW", "ERV", "FQRS", "LVQRSCL", "LVQRSLL",
              "PTW", "UW", "VB"],  # BUKAN target model
}

CODE_TO_GROUP = {code: grp for grp, codes in CHAPMAN_LABEL_MAP.items() for code in codes}

# kolom kode mentah yang dicoba (berurutan)
RAW_LABEL_COLUMN_CANDIDATES = [["Rhythm", "Beat"], ["labels"], ["label"], ["Dx"]]


def tokenize_codes(raw) -> list:
    """String label mentah -> list kode UPPERCASE. 'Nonspecific BBB' dijaga
    sebagai satu token (mengandung spasi)."""
    if raw is None or (not isinstance(raw, (list, tuple)) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        raw = " ".join(str(x) for x in raw)
    s = str(raw).upper()
    s = re.sub(r"NONSPECIFIC\s*BBB", "NONSPECIFICBBB", s)
    return [t for t in re.split(r"[;,|\s\[\]'\"]+", s) if t]


def map_codes_to_targets(tokens: list, target_classes) -> list:
    """
    Kode Chapman -> list superclass target.
    Aturan NORM: hanya diberikan kalau TIDAK ada kode non-normal sama sekali
    (termasuk AFIB & OTHER), agar tidak muncul label kontradiktif NORM+HYP
    untuk rekaman 'SR + LVH'.
    """
    groups = {CODE_TO_GROUP[t] for t in tokens if t in CODE_TO_GROUP}
    abnormal = groups - {"NORM"}
    result = {g for g in abnormal if g in target_classes}
    if "NORM" in groups and not abnormal and "NORM" in target_classes:
        result.add("NORM")
    return sorted(result)


def _parse_superclass_string(raw) -> list:
    """Fallback: parse kolom 'superclasses' lama (list-string / dipisah ;,|)."""
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


def _read_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _find_raw_label_source(df: pd.DataFrame, label_cols=None):
    if label_cols:
        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"label_cols {missing} tidak ada. Kolom: {list(df.columns)}")
        return list(label_cols)
    for cols in RAW_LABEL_COLUMN_CANDIDATES:
        if all(c in df.columns for c in cols):
            return cols
    return None


def load_chapman_metadata(chapman_csv: str, target_classes, label_cols=None) -> pd.DataFrame:
    """
    Return DataFrame: record_id, <target_classes> (0/1).
    Label dibangun dari kode mentah Chapman lewat CHAPMAN_LABEL_MAP.
    Baris tanpa satupun target_classes DIBUANG (konsisten dengan labels.py).
    """
    df = _read_table(chapman_csv)

    if "record_id" not in df.columns:
        if "FileName" in df.columns:
            df["record_id"] = df["FileName"]
        else:
            raise ValueError("Butuh kolom 'record_id' (atau 'FileName'). "
                             f"Kolom yang ada: {list(df.columns)}")
    df["record_id"] = (df["record_id"].astype(str).str.strip()
                       .str.replace(r"\.(mat|hea)$", "", regex=True))

    src = _find_raw_label_source(df, label_cols)
    if src is not None:
        print(f"[chapman_labels] Membangun label dari kode mentah kolom {src} "
              f"lewat CHAPMAN_LABEL_MAP")
        df["_tokens"] = df[src].apply(
            lambda r: tokenize_codes(" ".join(str(r[c]) for c in src if pd.notna(r[c])))
            , axis=1)
        df["superclass_list"] = df["_tokens"].apply(
            lambda t: map_codes_to_targets(t, target_classes))

        unknown = Counter(t for toks in df["_tokens"] for t in toks
                          if t not in CODE_TO_GROUP)
        if unknown:
            print(f"[chapman_labels] PERINGATAN: kode tidak ada di mapping (diabaikan): "
                  f"{dict(unknown.most_common(15))}")

        if "superclasses" in df.columns:
            old = df["superclasses"].apply(_parse_superclass_string)
            for c in target_classes:
                new_c = df["superclass_list"].apply(lambda l: c in l)
                old_c = old.apply(lambda l: c in l)
                print(f"[chapman_labels] Cek vs kolom 'superclasses' lama - {c}: "
                      f"lama={int(old_c.sum())}, baru={int(new_c.sum())}, "
                      f"beda={int((new_c != old_c).sum())}")
    elif "superclasses" in df.columns:
        print("[chapman_labels] PERINGATAN: kolom kode mentah tidak ditemukan, memakai "
              "kolom 'superclasses' apa adanya (mapping TIDAK diverifikasi). "
              "Sebaiknya sediakan kolom Rhythm+Beat / labels berisi kode Chapman.")
        df["superclass_list"] = df["superclasses"].apply(_parse_superclass_string)
    else:
        raise ValueError("Tidak ada kolom label. Butuh salah satu dari "
                         f"{RAW_LABEL_COLUMN_CANDIDATES} atau 'superclasses'. "
                         f"Kolom yang ada: {list(df.columns)}")

    n_before = len(df)
    df = df[df["superclass_list"].apply(len) > 0].copy()
    print(f"[chapman_labels] Buang {n_before - len(df)} record tanpa label target "
          f"({n_before} -> {len(df)})")

    for c in target_classes:
        df[c] = df["superclass_list"].apply(lambda lst: 1 if c in lst else 0)

    n_conflict = int(((df["NORM"] == 1) & (df[[c for c in target_classes if c != "NORM"]].sum(axis=1) > 0)).sum()) \
        if "NORM" in target_classes else 0
    print(f"[chapman_labels] Record berlabel NORM + kelas lain (harus 0): {n_conflict}")
    print("[chapman_labels] Distribusi: " +
          ", ".join(f"{c}={int(df[c].sum())}" for c in target_classes))
    return df


def compute_boost_target(train_df: pd.DataFrame, target_classes, boost_class: str) -> int:
    """Jumlah record tambahan agar boost_class sejajar RATA-RATA kelas lain di train set."""
    counts = {c: int(train_df[c].sum()) for c in target_classes}
    other = [v for c, v in counts.items() if c != boost_class]
    if not other:
        return 0
    avg_other = sum(other) / len(other)
    return max(0, int(round(avg_other - counts[boost_class])))


def filter_boost_class(df: pd.DataFrame, boost_class: str, boost_target: int,
                       seed: int = 42) -> pd.DataFrame:
    """Ambil subset record dengan boost_class aktif, maksimal boost_target record."""
    if boost_class not in df.columns:
        raise ValueError(f"boost_class '{boost_class}' tidak ada di kolom Chapman metadata")

    subset = df[df[boost_class] == 1].copy()
    n_available = len(subset)

    if n_available == 0:
        print(f"[chapman_labels] PERINGATAN: tidak ada record Chapman dengan label {boost_class}")
        return subset
    if boost_target <= 0:
        print("[chapman_labels] boost_target=0, tidak ada data Chapman yang ditambahkan")
        return subset.iloc[0:0]

    if n_available > boost_target:
        subset = subset.sample(n=boost_target, random_state=seed)
    elif n_available < boost_target:
        print(f"[chapman_labels] PERINGATAN: hanya {n_available} record {boost_class} tersedia "
              f"di Chapman, kurang dari target {boost_target}. Semua dipakai.")

    print(f"[chapman_labels] Boost kelas {boost_class}: {len(subset)}/{n_available} "
          f"record Chapman dipakai (target={boost_target})")
    return subset.reset_index(drop=True)