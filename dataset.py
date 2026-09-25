import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def filter_missing_files(df, data_dir, filename_col='filename', ext='.npy'):
    """
    Buang record yang file fisiknya nggak ada di disk buat mencegah crash saat iterasi DataLoader.
    """
    valid_indices = []
    for idx, row in df.iterrows():
        file_path = os.path.join(data_dir, f"{row[filename_col]}{ext}")
        if os.path.exists(file_path):
            valid_indices.append(idx)
        else:
            print(f"[WARNING] File hilang/nggak lengkap: {file_path}. Baris ini di-drop.")
            
    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    print(f"[INFO] Dataset difilter dari {len(df)} jadi {len(filtered_df)} sampel valid.")
    return filtered_df

def apply_chapman_boost(ptbxl_df, chapman_df, target_hyp_count, boost_neg_ratio, label_cols):
    """
    Injeksi data Chapman ke PTB-XL dengan rasio seimbang agar model tidak belajar
    domain shift (ciri khas RS/alat) sebagai fitur label HYP.
    """
    # 1. Ambil sampel HYP dari Chapman sesuai target_hyp_count yang masuk akal
    chapman_hyp = chapman_df[chapman_df['HYP'] == 1]
    if len(chapman_hyp) > target_hyp_count:
        chapman_hyp = chapman_hyp.sample(n=target_hyp_count, random_state=42)
        
    # 2. Ambil sampel negatif (non-HYP) dari Chapman
    # Hitung jumlah record non-HYP yang mau diambil
    chapman_neg = chapman_df[chapman_df['HYP'] == 0]
    target_neg_count = int(target_hyp_count * boost_neg_ratio)
    
    if len(chapman_neg) > target_neg_count:
        # Opsional: Bisa distratifikasi berdasar label lain kalau mau lebih rapi, 
        # tapi random sample juga udah cukup buat ngasih variasi noise domain.
        chapman_neg = chapman_neg.sample(n=target_neg_count, random_state=42)
        
    # 3. Gabungkan PTB-XL dengan subset Chapman yang sudah diseimbangkan
    combined_df = pd.concat([ptbxl_df, chapman_hyp, chapman_neg])
    
    # Shuffle dataset
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"\n[INFO] Chapman Boost Applied:")
    print(f"       + {len(chapman_hyp)} Chapman HYP records")
    print(f"       + {len(chapman_neg)} Chapman Non-HYP records")
    print(f"       Total Train Data: {len(combined_df)} records")
    
    return combined_df

class ECGDataset(Dataset):
    def __init__(self, df, data_dir, label_cols, ext='.npy', transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_cols = label_cols
        self.ext = ext
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, f"{row['filename']}{self.ext}")
        
        # Load sinyal
        if self.ext == '.npy':
            signal = np.load(file_path)
        else:
            raise ValueError(f"Ekstensi {self.ext} belum disupport, tambahin parser-nya bro.")
            
        # Transpose/reshape jika perlu agar formatnya (Channels, Length)
        # Sesuai standar Conv1D PyTorch
        if signal.shape[0] > signal.shape[1]:
            signal = np.transpose(signal)
            
        if self.transform:
            signal = self.transform(signal)
            
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        
        # Ekstrak label dari kolom (multi-label)
        labels = row[self.label_cols].values.astype(np.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return signal_tensor, label_tensor