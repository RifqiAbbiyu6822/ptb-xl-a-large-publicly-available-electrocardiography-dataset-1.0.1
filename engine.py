import math
import sys
import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, criterion, device, scaler, ema_model=None, max_norm=5.0):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    # Bungkus loader dengan tqdm biar progress training gampang dipantau
    pbar = tqdm(data_loader, desc="Training", leave=False, file=sys.stdout)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = inputs.size(0)
        optimizer.zero_grad()
        
        # Mixed-precision training context
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        # GUARD: Cek apakah loss finite (bukan NaN atau Inf)
        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"\n[WARNING] Loss is {loss_val}, skipping batch. Silakan cek learning rate atau data input.")
            continue
            
        # Scaling backward pass
        scaler.scale(loss).backward()
        
        # Unscale sebelum gradient clipping biar clipping dihitung di skala asli
        scaler.unscale_(optimizer)
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
        scaler.step(optimizer)
        scaler.update()
        
        # Update Exponential Moving Average (EMA) model kalau dipakai
        if ema_model is not None:
            ema_model.update(model)
            
        # Akumulasi loss berdasar jumlah sampel aktual di batch
        total_loss += loss_val * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': f"{loss_val:.4f}"})
        
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(data_loader, desc="Evaluating", leave=False, file=sys.stdout)
    
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        batch_size = inputs.size(0)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        loss_val = loss.item()
        if math.isfinite(loss_val):
            total_loss += loss_val * batch_size
            total_samples += batch_size
            
        # Simpan probabilitas (sigmoid) dan target asli untuk kalkulasi metrik AUROC/AUPRC
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    # Gabungkan semua tensor dari semua batch
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return avg_loss, all_preds, all_targets