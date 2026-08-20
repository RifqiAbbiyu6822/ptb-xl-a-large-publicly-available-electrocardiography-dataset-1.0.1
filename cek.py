import argparse
import random

import wfdb

from labels import load_ptbxl_metadata
from chapman_labels import load_chapman_metadata
from chapman_dataset import load_chapman_record


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", type=str, required=True)
    p.add_argument("--chapman_root", type=str, required=True)
    p.add_argument("--chapman_csv", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=== 1. Statistik amplitudo PTB-XL (referensi) ===")
    ptbxl_df = load_ptbxl_metadata(args.ptbxl_root, ["NORM", "MI", "STTC", "CD", "HYP"])
    ptbxl_sample_rows = ptbxl_df.sample(n=min(args.n_samples, len(ptbxl_df)), random_state=args.seed)

    ptbxl_stds = []
    for _, row in ptbxl_sample_rows.iterrows():
        import os
        path = os.path.join(args.ptbxl_root, row["filename_lr"])
        sig, meta = wfdb.rdsamp(path)
        print(f"  {row.name}: fs={meta['fs']}Hz  shape={sig.shape}  "
              f"mean={sig.mean():.4f}  std={sig.std():.4f}  "
              f"min={sig.min():.4f}  max={sig.max():.4f}")
        ptbxl_stds.append(sig.std())

    ptbxl_avg_std = sum(ptbxl_stds) / len(ptbxl_stds)
    print(f"\n  Rata-rata std PTB-XL: {ptbxl_avg_std:.4f}")

    print("\n=== 2. Statistik amplitudo Chapman-Shaoxing ===")
    chapman_df = load_chapman_metadata(args.chapman_csv, ["NORM", "MI", "STTC", "CD", "HYP"])
    hyp_rows = chapman_df[chapman_df["HYP"] == 1]
    if len(hyp_rows) == 0:
        print("  [PERINGATAN] Tidak ada record HYP di Chapman, cek CSV/mapping label.")
        return
    sample_rows = hyp_rows.sample(n=min(args.n_samples, len(hyp_rows)), random_state=args.seed)

    chapman_stds = []
    for _, row in sample_rows.iterrows():
        record_id = row["record_id"]
        try:
            signal, fs = load_chapman_record(record_id, args.chapman_root)
        except Exception as e:
            print(f"  [ERROR] Gagal load {record_id}: {e}")
            continue
        print(f"  {record_id}: fs={fs}Hz  shape={signal.shape}  "
              f"mean={signal.mean():.4f}  std={signal.std():.4f}  "
              f"min={signal.min():.4f}  max={signal.max():.4f}")
        chapman_stds.append(signal.std())

    if not chapman_stds:
        print("\n  [ERROR] Tidak ada record Chapman yang berhasil di-load. Cek path/nama file.")
        return

    chapman_avg_std = sum(chapman_stds) / len(chapman_stds)
    ratio = chapman_avg_std / ptbxl_avg_std if ptbxl_avg_std > 0 else float("inf")

    print(f"\n  Rata-rata std Chapman: {chapman_avg_std:.4f}")
    print(f"\n=== 3. Kesimpulan ===")
    print(f"  Rasio std Chapman/PTB-XL: {ratio:.2f}x")
    if 0.2 <= ratio <= 5:
        print("  -> AMAN. Skala amplitudo sebanding, kemungkinan besar gain/baseline "
              "parsing sudah benar. Boleh lanjut training dengan --chapman_root.")
    else:
        print("  -> PERINGATAN: skala amplitudo beda jauh (>5x atau <0.2x). Kemungkinan "
              "gain/baseline di header .hea salah di-parse untuk sebagian/semua lead. "
              "JANGAN training dulu dengan data ini -- laporkan angka di atas untuk "
              "diperbaiki parser-nya.")


if __name__ == "__main__":
    main()