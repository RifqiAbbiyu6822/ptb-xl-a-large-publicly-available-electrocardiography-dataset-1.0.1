"""
train.py
Script training SE-ConvNeXt1D untuk klasifikasi multi-label PTB-XL.

v2 (imbalance-focused): loss function sekarang bisa dipilih lewat --loss_fn
(bce / bce_pos_weight / focal / asl), dan WeightedRandomSampler bisa
dinyalakan/dimatikan lewat --use_sampler/--no_sampler. Ini supaya kamu bisa
ablasi dengan jelas kombinasi mana yang paling membantu kelas sulit (HYP),
bukan cuma ganti-ganti tanpa bukti.

Rekomendasi default (sudah di-set): --loss_fn asl --no_sampler
  ASL menangani imbalance secara dinamis di level gradient, jadi oversampling
  eksplisit biasanya tidak perlu lagi (lihat losses.py untuk penjelasan).

Contoh pakai:
    # Rekomendasi utama: ASL, tanpa sampler
    python train.py --ptbxl_root "C:\\path\\ke\\ptb-xl" --loss_fn asl

    # Ablasi 1: baseline BCE polos (tanpa penanganan imbalance sama sekali)
    python train.py --ptbxl_root "C:\\path\\ke\\ptb-xl" --loss_fn bce --no_sampler

    # Ablasi 2: BCE + pos_weight (setup lama)
    python train.py --ptbxl_root "C:\\path\\ke\\ptb-xl" --loss_fn bce_pos_weight --no_sampler

    # Ablasi 3: ASL + sampler (cek apakah kombinasi over-correct atau membantu)
    python train.py --ptbxl_root "C:\\path\\ke\\ptb-xl" --loss_fn asl --use_sampler

Fitur:
  - AdamW + cosine LR scheduler
  - Mixed precision (AMP) otomatis kalau ada GPU CUDA
  - Gradient clipping
  - Early stopping berdasarkan macro-F1 validasi
  - Simpan checkpoint terbaik (best_model.pt) + checkpoint terakhir (last_model.pt)
  - Evaluasi akhir di test set memakai checkpoint terbaik
  - history.json + run_config.json (semua hyperparameter loss/sampler dicatat
    supaya tiap run bisa dibandingkan/direproduksi)
"""

import argparse
import os
import json
import time

import pandas as pd
import torch

from config import PTBXLConfig
from labels import load_ptbxl_metadata, split_by_fold, class_distribution
from dataset import build_dataloaders
from utils import set_seed
from model import build_model
from engine import train_one_epoch, evaluate
from losses import build_criterion
from chapman_labels import load_chapman_metadata, compute_boost_target, filter_boost_class
from chapman_dataset import ChapmanBoostDataset


def compute_pos_weight(train_df, target_classes, clip_max: float = 10.0) -> torch.Tensor:
    """
    pos_weight untuk BCEWithLogitsLoss: rasio (jumlah negative / jumlah positive)
    per kelas. Hanya dipakai kalau --loss_fn bce_pos_weight.
    clip_max membatasi rasio supaya tidak terlalu ekstrem (training tetap stabil).
    """
    n_total = len(train_df)
    weights = []
    for c in target_classes:
        n_pos = train_df[c].sum()
        n_neg = n_total - n_pos
        w = n_neg / max(n_pos, 1)
        w = min(w, clip_max)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--ptbxl_root", type=str, required=True)
    p.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # model
    p.add_argument("--model_variant", type=str, default="tiny",
                    choices=["nano", "tiny", "small"])
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--se_reduction", type=int, default=16)
    # optimisasi
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=15,
                    help="Early stopping: berhenti kalau macro-F1 val tidak naik selama N epoch. "
                         "Dinaikkan dari 10 -> 15: dengan CosineAnnealingLR(T_max=epochs), val_macro_f1 "
                         "bisa oscillating cukup lama selama LR masih tinggi (awal-tengah schedule). "
                         "patience terlalu pendek relatif ke T_max bisa memicu stop SEBELUM LR sempat "
                         "turun ke fase konvergen -> lihat --lr_scheduler plateau untuk alternatif yang "
                         "lebih robust terhadap masalah ini.")
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["cosine", "plateau"],
                    help="'cosine': LR turun mengikuti jadwal tetap (T_max=epochs), tidak peduli apakah "
                         "model sudah plateau atau belum -> rawan early-stop prematur kalau training "
                         "berhenti jauh sebelum T_max (LR masih tinggi saat berhenti). "
                         "'plateau' (REKOMENDASI BARU): LR baru diturunkan kalau val_macro_f1 stagnan "
                         "beberapa epoch -> lebih adaptif, tidak terikat asumsi jumlah epoch total.")
    p.add_argument("--plateau_factor", type=float, default=0.5,
                    help="[--lr_scheduler plateau] LR dikali faktor ini tiap kali plateau terdeteksi")
    p.add_argument("--plateau_patience", type=int, default=4,
                    help="[--lr_scheduler plateau] jumlah epoch tanpa peningkatan sebelum LR diturunkan")

    # --- Penanganan imbalance: loss function ---
    p.add_argument("--loss_fn", type=str, default="asl",
                    choices=["bce", "bce_pos_weight", "focal", "asl"],
                    help="Pilihan loss function untuk multi-label imbalance. "
                         "'asl' (Asymmetric Loss) direkomendasikan sebagai default: "
                         "lebih stabil dari pos_weight dan fokus dinamis ke hard/rare sample.")
    p.add_argument("--pos_weight_clip", type=float, default=10.0,
                    help="[hanya untuk --loss_fn bce_pos_weight] batas maksimum pos_weight per kelas")
    p.add_argument("--asl_gamma_neg", type=float, default=2.0,
                    help="[hanya untuk --loss_fn asl] focusing untuk easy negative, makin besar makin ditekan. "
                         "Default diturunkan ke 2.0 (dari rekomendasi paper 4.0): dataset ini cuma 5 kelas "
                         "dengan imbalance moderat (~3.6:1), bukan extreme multi-label tagging (50:1+) yang "
                         "jadi basis default asli. gamma_neg=4 empiris terbukti menurunkan F1 di SEMUA kelas, "
                         "bukan cuma kelas mayoritas -> indikasi terlalu agresif membuang gradient.")
    p.add_argument("--asl_gamma_pos", type=float, default=1.0,
                    help="[hanya untuk --loss_fn asl] focusing untuk sample positif, dibuat kecil "
                         "supaya sample langka (HYP) tetap dapat gradient penuh")
    p.add_argument("--asl_clip", type=float, default=0.05,
                    help="[hanya untuk --loss_fn asl] probability margin, buang easy negative "
                         "yang confidence-nya sudah sangat tinggi")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                    help="[hanya untuk --loss_fn focal] focusing parameter")
    p.add_argument("--focal_alpha", type=float, default=None,
                    help="[hanya untuk --loss_fn focal] weighting statis pos vs neg (opsional)")

    # --- Penanganan imbalance: sampler ---
    p.add_argument("--use_sampler", dest="use_sampler", action="store_true",
                    help="Aktifkan WeightedRandomSampler (oversampling eksplisit kelas minor). "
                         "Default OFF karena loss_fn asl/focal sudah menangani imbalance "
                         "secara dinamis; kombinasi keduanya bisa over-correct/tidak stabil. "
                         "Nyalakan untuk ablasi.")
    p.add_argument("--no_sampler", dest="use_sampler", action="store_false")
    p.set_defaults(use_sampler=False)

    # --- Override augmentasi lead-dropout dari CLI (tanpa perlu edit config.py) ---
    # supaya ablasi "apakah lead_dropout yang bikin turun?" gampang dites terpisah.
    p.add_argument("--use_lead_dropout", dest="use_lead_dropout", action="store_true")
    p.add_argument("--no_lead_dropout", dest="use_lead_dropout", action="store_false")
    p.set_defaults(use_lead_dropout=None)  # None = pakai default dari config.py, tidak di-override
    p.add_argument("--lead_dropout_prob", type=float, default=None,
                    help="Override config.lead_dropout_prob. None = pakai default config.py")
    p.add_argument("--normalize_mode", type=str, default=None, choices=["per_lead", "global"],
                    help="Override config.normalize_mode. 'global' (default config.py) menjaga rasio "
                         "amplitudo antar-lead -> penting untuk HYP (Sokolow-Lyon/Cornell voltage "
                         "criteria berbasis perbandingan antar-lead). 'per_lead' (perilaku lama) "
                         "menghapus informasi ini. None = pakai default config.py.")

    # --- Boost kelas minoritas dengan data eksternal Chapman-Shaoxing ---
    p.add_argument("--chapman_root", type=str, default=None,
                    help="Folder root Chapman-Shaoxing (berisi <record_id>.hea + <record_id>.mat, "
                         "satu folder datar). Kalau diisi, dipakai untuk menambah data kelas "
                         "--boost_class di TRAIN SET SAJA (val/test tetap murni PTB-XL).")
    p.add_argument("--chapman_csv", type=str, default=None,
                    help="Path ke CSV metadata Chapman (kolom minimal: record_id, superclasses). "
                         "Wajib diisi kalau --chapman_root diisi.")
    p.add_argument("--boost_class", type=str, default="HYP",
                    help="Kelas target yang mau ditambah datanya dari Chapman-Shaoxing")
    p.add_argument("--boost_target", type=int, default=None,
                    help="Jumlah record boost yang mau ditambahkan. None (default) = dihitung "
                         "otomatis dari train_df supaya jumlah --boost_class sejajar RATA-RATA "
                         "jumlah kelas lain di train set (lihat chapman_labels.compute_boost_target).")

    # lain-lain
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = PTBXLConfig(ptbxl_root=args.ptbxl_root, sampling_rate=args.sampling_rate,
                          seed=args.seed)
    if args.use_lead_dropout is not None:
        config.use_lead_dropout = args.use_lead_dropout
    if args.lead_dropout_prob is not None:
        config.lead_dropout_prob = args.lead_dropout_prob
    if args.normalize_mode is not None:
        config.normalize_mode = args.normalize_mode
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    print(f"[train] loss_fn={args.loss_fn}  use_sampler={args.use_sampler}  "
          f"lr_scheduler={args.lr_scheduler}  normalize_mode={config.normalize_mode}  "
          f"lead_dropout={config.use_lead_dropout} (p={config.lead_dropout_prob})")

    # --- 1. Data ---
    print("\n=== Load & split data ===")
    df = load_ptbxl_metadata(config.ptbxl_root, config.target_classes)
    train_df, val_df, test_df = split_by_fold(df, config.test_fold, config.val_fold)
    class_distribution(train_df, config.target_classes, name="Train (PTB-XL)")
    class_distribution(val_df, config.target_classes, name="Val")

    # --- 1b. Boost kelas minoritas dengan Chapman-Shaoxing (opsional, TRAIN SET SAJA) ---
    boost_dataset = None
    boost_df = None
    if args.chapman_root:
        if not args.chapman_csv:
            raise ValueError("--chapman_root diisi tapi --chapman_csv kosong")
        print(f"\n=== Load data boost Chapman-Shaoxing (kelas: {args.boost_class}) ===")
        chapman_df = load_chapman_metadata(args.chapman_csv, config.target_classes)

        boost_target = args.boost_target
        if boost_target is None:
            boost_target = compute_boost_target(train_df, config.target_classes, args.boost_class)
            print(f"[train] boost_target dihitung otomatis: {boost_target} record "
                  f"(supaya {args.boost_class} sejajar rata-rata kelas lain di train set)")

        boost_df = filter_boost_class(chapman_df, args.boost_class, boost_target, seed=config.seed)
        if len(boost_df) > 0:
            boost_dataset = ChapmanBoostDataset(boost_df, config, args.chapman_root)
            combined_label_df = pd.concat(
                [train_df[config.target_classes], boost_df[config.target_classes]],
                ignore_index=True,
            )
            class_distribution(combined_label_df, config.target_classes,
                                name="Train (PTB-XL + Chapman boost)")
        else:
            print("[train] Tidak ada data boost yang dipakai (boost_df kosong)")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_sampler=args.use_sampler,
        boost_dataset=boost_dataset, boost_df=boost_df,
    )

    # --- 2. Model ---
    print(f"\n=== Bangun model SE-ConvNeXt1D (variant={args.model_variant}) ===")
    model = build_model(
        variant=args.model_variant, in_channels=config.n_leads,
        num_classes=len(config.target_classes),
        drop_path_rate=args.drop_path_rate, se_reduction=args.se_reduction,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Jumlah parameter: {n_params:,}")

    # --- 3. Loss, optimizer, scheduler ---
    pos_weight = None
    if args.loss_fn == "bce_pos_weight":
        pos_weight_df = train_df
        if boost_df is not None and len(boost_df) > 0:
            pos_weight_df = pd.concat(
                [train_df[config.target_classes], boost_df[config.target_classes]],
                ignore_index=True,
            )
        pos_weight = compute_pos_weight(pos_weight_df, config.target_classes,
                                         clip_max=args.pos_weight_clip).to(device)
        print(f"\npos_weight per kelas ({config.target_classes}), dihitung dari "
              f"{'train+boost' if boost_df is not None and len(boost_df) > 0 else 'train'} set: "
              f"{[round(w, 2) for w in pos_weight.tolist()]}")

    criterion = build_criterion(
        args.loss_fn, pos_weight=pos_weight,
        asl_gamma_neg=args.asl_gamma_neg, asl_gamma_pos=args.asl_gamma_pos, asl_clip=args.asl_clip,
        focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
    )
    print(f"[train] criterion: {criterion.__class__.__name__}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor,
            patience=args.plateau_patience,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # --- 4. Training loop ---
    best_macro_f1 = -1.0
    epochs_no_improve = 0
    history = []

    # catat konfigurasi run supaya tiap eksperimen bisa dibandingkan/direproduksi
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n=== Mulai training ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, scaler=scaler, grad_clip=args.grad_clip)
        val_metrics = evaluate(model, val_loader, criterion, device,
                                target_classes=config.target_classes,
                                threshold=args.threshold)

        if args.lr_scheduler == "plateau":
            scheduler.step(val_metrics["macro_f1"])
        else:
            scheduler.step()
        dt = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_macro_auroc={val_metrics['macro_auroc']:.4f}  "
              f"f1_HYP={val_metrics.get('f1_HYP', float('nan')):.4f}  "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # checkpoint terakhir (selalu ditimpa)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        }, os.path.join(args.output_dir, "last_model.pt"))

        # checkpoint terbaik berdasarkan macro-F1 validasi
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"  -> checkpoint terbaik disimpan (macro_f1={best_macro_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[train] Early stopping di epoch {epoch} "
                      f"(tidak ada peningkatan selama {args.patience} epoch)")
                break

    # simpan history training
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- 5. Evaluasi akhir di test set memakai checkpoint terbaik ---
    print("\n=== Evaluasi test set (checkpoint terbaik) ===")
    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device,
                             target_classes=config.target_classes,
                             threshold=args.threshold)
    print(json.dumps(test_metrics, indent=2))

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
