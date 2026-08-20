import argparse

from config import PTBXLConfig
from labels import load_ptbxl_metadata, split_by_fold, class_distribution
from dataset import build_dataloaders
from utils import set_seed, validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl_root", type=str, required=True,
                         help="Folder root PTB-XL (berisi ptbxl_database.csv, records100/, records500/)")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    config = PTBXLConfig(ptbxl_root=args.ptbxl_root, sampling_rate=args.sampling_rate)
    set_seed(config.seed)

    print("=== 1. Load & agregasi label multi-label ===")
    df = load_ptbxl_metadata(config.ptbxl_root, config.target_classes)
    class_distribution(df, config.target_classes, name="Seluruh dataset")

    print("\n=== 2. Split train/val/test sesuai strat_fold resmi PTB-XL ===")
    train_df, val_df, test_df = split_by_fold(df, config.test_fold, config.val_fold)
    class_distribution(train_df, config.target_classes, name="Train")
    class_distribution(val_df, config.target_classes, name="Val")
    class_distribution(test_df, config.target_classes, name="Test")

    print("\n=== 3. Bangun Dataset & DataLoader (preprocessing on-the-fly) ===")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config, batch_size=args.batch_size
    )

    print("\n=== 4. Validasi cepat dataset (shape, NaN, label kosong) ===")
    validate_dataset(train_loader.dataset, n_samples=20)

    print("\n=== 5. Cek satu batch ===")
    signals, labels = next(iter(train_loader))
    print(f"  Batch sinyal shape: {tuple(signals.shape)}  (batch, n_leads, target_length)")
    print(f"  Batch label shape : {tuple(labels.shape)}  (batch, n_classes)")
    print(f"  Contoh label baris pertama: {labels[0].tolist()} -> kelas: {config.target_classes}")


if __name__ == "__main__":
    main()
