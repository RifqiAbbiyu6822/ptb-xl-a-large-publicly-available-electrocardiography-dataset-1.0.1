import torch
import torch.nn as nn


class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss (ASL) untuk multi-label classification.
    Referensi: Ben-Baruch et al., "Asymmetric Loss For Multi-Label
    Classification", ICCV 2021.

    Parameter:
      gamma_neg : focusing untuk sample NEGATIF (kelas yang tidak aktif).
                  Makin besar -> makin menekan kontribusi easy negative.
      gamma_pos : focusing untuk sample POSITIF (kelas yang aktif).
                  Dibuat kecil/nol supaya sample positif (langka, mis. HYP)
                  tidak ikut ditekan gradiennya.
      clip      : probability margin. Negative logit dengan prob < clip
                  dianggap "sudah pasti negatif", dibuang total dari loss.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8,
                 disable_torch_grad_focal_loss: bool = True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric probability shifting: buang kontribusi easy negative
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Binary cross-entropy dasar
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing (dinamis per sample, beda gamma untuk pos/neg)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    one_sided_w = self._focal_weight(xs_pos, xs_neg, targets)
            else:
                one_sided_w = self._focal_weight(xs_pos, xs_neg, targets)
            loss = loss * one_sided_w

        return -loss.sum(dim=1).mean()

    def _focal_weight(self, xs_pos, xs_neg, targets):
        pt0 = xs_pos * targets
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        return torch.pow(1 - pt, one_sided_gamma)


class FocalLossMultiLabel(nn.Module):
    """
    Sigmoid Focal Loss (Lin et al., 2017) untuk multi-label.
    Alternatif yang lebih sederhana dari ASL: satu gamma untuk pos & neg
    (simetris), plus alpha skalar opsional untuk weighting statis tambahan
    antara kelas positif vs negatif secara keseluruhan.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t).clamp(min=self.eps) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.sum(dim=1).mean()


def build_criterion(loss_fn: str, pos_weight: torch.Tensor = None,
                     asl_gamma_neg: float = 4.0, asl_gamma_pos: float = 1.0,
                     asl_clip: float = 0.05,
                     focal_gamma: float = 2.0, focal_alpha: float = None) -> nn.Module:
    """Factory: bangun loss module sesuai pilihan --loss_fn di train.py."""
    loss_fn = loss_fn.lower()

    if loss_fn == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_fn == "bce_pos_weight":
        if pos_weight is None:
            raise ValueError("loss_fn='bce_pos_weight' butuh pos_weight (dihitung dari train_df)")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if loss_fn == "focal":
        return FocalLossMultiLabel(gamma=focal_gamma, alpha=focal_alpha)

    if loss_fn == "asl":
        return AsymmetricLossMultiLabel(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos,
                                         clip=asl_clip)

    raise ValueError(f"loss_fn '{loss_fn}' tidak dikenal, pilih dari "
                      f"['bce', 'bce_pos_weight', 'focal', 'asl']")
