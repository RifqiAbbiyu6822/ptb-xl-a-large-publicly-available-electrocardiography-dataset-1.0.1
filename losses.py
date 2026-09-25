import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        """
        Asymmetric Loss untuk Multi-Label Classification.
        Sudah diperbaiki untuk stabilitas Mixed-Precision Training (AMP / fp16).
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # FIX STABILITAS: Paksa casting logits dan target ke float32 (fp32)
        # Menghindari efek samping autocast fp16 yang bikin clamp(min=1e-8) jadi 0
        x = x.float()
        y = y.float()
        
        # Kalkulasi probabilitas
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping untuk membuang negative sample yang terlalu mudah
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Guard tambahan sebelum masuk logaritma biar aman dari -inf
        xs_pos = xs_pos.clamp(min=self.eps)
        xs_neg = xs_neg.clamp(min=self.eps)

        # Hitung Asymmetric Focal Loss
        loss_pos = y * torch.log(xs_pos) * (1 - xs_pos) ** self.gamma_pos
        loss_neg = (1 - y) * torch.log(xs_neg) * (1 - xs_neg) ** self.gamma_neg
        
        # Gabung dan rata-rata
        loss = loss_pos + loss_neg
        return -torch.mean(loss)

class AsymmetricLossOptimized(nn.Module):
    """
    Versi alternatif kalau lo mau iterasi lebih rapi untuk per-kelas, 
    biasanya dipakai kalau ada class-weighting spesifik. 
    Secara default pakai AsymmetricLoss di atas sudah cukup.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        x = x.float()
        y = y.float()

        self.targets = y
        self.anti_targets = 1 - y

        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1 - self.xs_pos

        if self.clip is not None and self.clip > 0:
            self.xs_neg = (self.xs_neg + self.clip).clamp(max=1)

        self.xs_pos = self.xs_pos.clamp(min=self.eps)
        self.xs_neg = self.xs_neg.clamp(min=self.eps)

        self.loss = self.targets * torch.log(self.xs_pos)
        self.loss = self.loss.add(self.anti_targets * torch.log(self.xs_neg))
        
        self.asymmetric_w = torch.pow(1 - self.xs_pos, self.gamma_pos) * self.targets
        self.asymmetric_w = self.asymmetric_w.add(torch.pow(1 - self.xs_neg, self.gamma_neg) * self.anti_targets)

        self.loss = self.loss * self.asymmetric_w
        return -self.loss.mean()