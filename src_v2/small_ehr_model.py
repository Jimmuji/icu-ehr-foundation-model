"""Small EHR Transformer (~2M params) — masked-feature autoencoder.

Design choices vs v1 (228M EHRFormer):
- DROP the VAE branch entirely (root cause of v1's posterior collapse)
- Smaller architecture: 4-layer transformer, hidden 128, 4 heads
- Cleaner MAE-style pretraining (masked-feature reconstruction only)
- Token+value embedding instead of per-timestep BERT pooling
- Per-feature linear heads for reconstruction (cat → CE, cont → MSE)

Input per stay (matching v1 tokenization, so we can reuse the existing
ehr_cache):
  tokenized_category_feats : (F_cat,   T) int64  (-1 = missing)
  tokenized_float_feats    : (F_float, T) int64  (-1 = missing)
  float_feats              : (F_float, T) float  (raw, for regression target)
  valid_mask               : (T,)        bool
  time_index               : (T,)        int

Forward returns:
  cls_logits  : list[(B, T, n_cat_vals)] × F_cat   (cat reconstruction logits)
  reg_preds   : list[(B, T)]              × F_float (continuous reconstruction values)
  hidden      : (B, T, D)                          (mean-poolable encoder output)

Param count target: ~2M.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallEHRTransformer(nn.Module):
    def __init__(
        self,
        n_cat_feats: int,
        n_float_feats: int,
        n_cat_values: int,
        n_float_bins: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 256,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        missing_idx: int = -1,
        float_mode: str = "mse",   # "mse" = regression head; "ce" = predict quantile bin (APOLLO/MOTOR style)
    ):
        super().__init__()
        self.cfg = dict(
            n_cat_feats=n_cat_feats, n_float_feats=n_float_feats,
            n_cat_values=n_cat_values, n_float_bins=n_float_bins,
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            ff_dim=ff_dim, max_seq_len=max_seq_len, dropout=dropout,
            pad_idx=pad_idx, missing_idx=missing_idx, float_mode=float_mode,
        )
        self.float_mode = float_mode
        D = hidden_dim
        self.D = D
        self.n_cat = n_cat_feats
        self.n_float = n_float_feats

        # ---- Token embeddings (SHARED across features + per-feature offset) ----
        # Idea: one shared value-embedding for cat tokens, one for float bins;
        # add a learned per-feature offset embedding. Cuts ~6M params.
        self.cat_value_emb = nn.Embedding(n_cat_values + 1, D, padding_idx=0)   # shared
        self.cat_feat_emb = nn.Embedding(n_cat_feats, D)                        # per-cat-feature offset
        self.float_value_emb = nn.Embedding(n_float_bins + 1, D, padding_idx=0) # shared
        self.float_feat_emb = nn.Embedding(n_float_feats, D)                    # per-float-feature offset

        # ---- Time + position embedding ----
        self.time_emb = nn.Embedding(max_seq_len, D)

        # ---- Transformer encoder ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(D)

        # ---- Reconstruction heads ----
        # Cat: one logit head per cat feature, output n_cat_values classes
        self.cat_heads = nn.ModuleList([nn.Linear(D, n_cat_values) for _ in range(n_cat_feats)])
        if float_mode == "mse":
            # Float: shared decoder + per-feature linear (predict raw value, standardized)
            self.float_decoder = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, n_float_feats))
        elif float_mode == "ce":
            # Float-as-classification (APOLLO / MOTOR style): predict the quantile BIN
            # instead of regressing the raw value. Param-efficient: one SHARED bin head
            # + a per-feature decoder offset (mirrors the input-embedding trick above).
            self.dec_float_feat_emb = nn.Embedding(n_float_feats, D)
            self.float_bin_head = nn.Linear(D, n_float_bins)
        else:
            raise ValueError(f"float_mode must be 'mse' or 'ce', got {float_mode!r}")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed_step(
        self,
        cat_in: torch.Tensor,   # (B, F_cat, T) int64, with -1 = missing
        float_in: torch.Tensor, # (B, F_float, T) int64, with -1 = missing
    ) -> torch.Tensor:
        """Sum embeddings over all features per timestep.
        Each (feature_idx, value) pair contributes (shared_value_emb + per_feature_offset).
        """
        B, _, T = cat_in.shape
        device = cat_in.device

        # ---- Cat ----
        # Shift -1 → 0 ; clamp to embedding range
        cat_safe = (cat_in + 1).clamp(min=0, max=self.cat_value_emb.num_embeddings - 1)  # (B, F_cat, T)
        cat_val = self.cat_value_emb(cat_safe)                  # (B, F_cat, T, D)
        feat_ids = torch.arange(self.n_cat, device=device)      # (F_cat,)
        cat_off = self.cat_feat_emb(feat_ids).view(1, self.n_cat, 1, -1)  # (1, F_cat, 1, D)
        # Mask missing positions (where original was -1) to zero contribution from offset
        missing_cat = (cat_in == -1).unsqueeze(-1)              # (B, F_cat, T, 1)
        cat_total = (cat_val + cat_off).masked_fill(missing_cat, 0.0).sum(dim=1)  # (B, T, D)

        # ---- Float ----
        float_safe = (float_in + 1).clamp(min=0, max=self.float_value_emb.num_embeddings - 1)
        float_val = self.float_value_emb(float_safe)
        feat_ids = torch.arange(self.n_float, device=device)
        float_off = self.float_feat_emb(feat_ids).view(1, self.n_float, 1, -1)
        missing_float = (float_in == -1).unsqueeze(-1)
        float_total = (float_val + float_off).masked_fill(missing_float, 0.0).sum(dim=1)

        return cat_total + float_total  # (B, T, D)

    def forward(
        self,
        cat_in: torch.Tensor,    # (B, F_cat, T) — masked input
        float_in: torch.Tensor,  # (B, F_float, T) — masked input (bin idx)
        time_index: torch.Tensor,  # (B, T)
        valid_mask: torch.Tensor,  # (B, T) bool
    ):
        # Embed step
        h = self.embed_step(cat_in, float_in)            # (B, T, D)
        h = h + self.time_emb(time_index.clamp(min=0))   # add time emb

        # Build attention mask: True where INVALID (transformer ignores those)
        src_key_padding_mask = ~valid_mask  # (B, T)

        z = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        z = self.norm(z)  # (B, T, D)

        # Heads
        cls_logits = [head(z) for head in self.cat_heads]           # each (B, T, n_cat_values)
        if self.float_mode == "mse":
            reg_all = self.float_decoder(z)                          # (B, T, F_float)
            float_out = [reg_all[..., f] for f in range(self.n_float)]   # each (B, T)
        else:  # "ce" — per-feature bin logits via shared head + per-feature offset
            feat_ids = torch.arange(self.n_float, device=z.device)
            offs = self.dec_float_feat_emb(feat_ids)                 # (F_float, D)
            float_out = [self.float_bin_head(z + offs[f]) for f in range(self.n_float)]  # each (B, T, n_bins)
        return cls_logits, float_out, z


def loss_pretrain(
    cls_logits: list[torch.Tensor],     # each (B, T, n_cat_values)
    float_out: list[torch.Tensor],      # mse: each (B, T); ce: each (B, T, n_bins)
    cat_target: torch.Tensor,           # (B, F_cat, T) int64 — unmasked target tokens
    cat_mask_loss: torch.Tensor,        # (B, F_cat, T) bool — True at positions we masked & should predict
    float_mask_loss: torch.Tensor,      # (B, F_float, T) bool
    float_target_raw: torch.Tensor | None = None,   # (B, F_float, T) float — for mse
    float_target_bin: torch.Tensor | None = None,   # (B, F_float, T) int  — for ce (-1 = missing)
    float_mode: str = "mse",
    label_pad_idx: int = -1,
) -> tuple[torch.Tensor, dict]:
    """Reconstruction loss across all masked positions.

    Notes:
      - cat_target uses -1 to signal "no signal here" (skipped via mask_loss)
      - mse mode: float_target_raw is the (z-normalized) continuous value
      - ce  mode: float_target_bin is the quantile-bin index (-1 = missing → skipped)
    """
    B, F_cat, T = cat_target.shape
    cls_loss = 0.0
    n_cls = 0
    for f in range(F_cat):
        logits = cls_logits[f]                  # (B, T, V)
        tgt = cat_target[:, f, :]               # (B, T) int
        m = cat_mask_loss[:, f, :]              # (B, T) bool
        if m.sum() == 0:
            continue
        L = F.cross_entropy(logits[m], tgt[m].clamp(min=0), reduction="mean")
        cls_loss = cls_loss + L
        n_cls += 1
    cls_loss = cls_loss / max(n_cls, 1) if n_cls > 0 else logits.new_zeros(())

    F_float = len(float_out)
    reg_loss = 0.0
    n_reg = 0
    for f in range(F_float):
        m = float_mask_loss[:, f, :]
        if float_mode == "mse":
            pred = float_out[f]                  # (B, T)
            tgt = float_target_raw[:, f, :]      # (B, T)
            m = m & ~torch.isnan(tgt)            # drop NaN targets
            if m.sum() == 0:
                continue
            L = F.mse_loss(pred[m], tgt[m], reduction="mean")
        else:  # ce
            logits_f = float_out[f]              # (B, T, n_bins)
            tgt = float_target_bin[:, f, :]      # (B, T) int, -1 = missing
            m = m & (tgt >= 0)                   # drop missing-value targets
            if m.sum() == 0:
                continue
            L = F.cross_entropy(logits_f[m], tgt[m], reduction="mean")
        reg_loss = reg_loss + L
        n_reg += 1
    reg_loss = reg_loss / max(n_reg, 1) if n_reg > 0 else cls_loss.new_zeros(())

    total = cls_loss + reg_loss
    return total, dict(cls=cls_loss.detach().item() if torch.is_tensor(cls_loss) else cls_loss,
                       reg=reg_loss.detach().item() if torch.is_tensor(reg_loss) else reg_loss,
                       total=total.detach().item())


if __name__ == "__main__":
    # Quick smoke test + param count for BOTH float modes (forward + loss + backward)
    B, T = 2, 64
    cat_in = torch.randint(-1, 50, (B, 11, T))
    float_in = torch.randint(-1, 256, (B, 199, T))
    time_index = torch.arange(T).unsqueeze(0).repeat(B, 1)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    # Targets (unmasked) + a fake loss mask (predict ~half the positions)
    cat_tgt = torch.randint(0, 50, (B, 11, T))
    float_bin_tgt = torch.randint(-1, 256, (B, 199, T))
    float_raw_tgt = torch.randn(B, 199, T)
    cat_mask = torch.rand(B, 11, T) < 0.5
    float_mask = torch.rand(B, 199, T) < 0.5

    for mode in ["mse", "ce"]:
        model = SmallEHRTransformer(
            n_cat_feats=11, n_float_feats=199,
            n_cat_values=50, n_float_bins=256,
            hidden_dim=192, n_layers=6, n_heads=4, ff_dim=384, max_seq_len=512,
            float_mode=mode,
        )
        cls_logits, float_out, z = model(cat_in, float_in, time_index, valid_mask)
        loss, parts = loss_pretrain(
            cls_logits, float_out,
            cat_target=cat_tgt, cat_mask_loss=cat_mask, float_mask_loss=float_mask,
            float_target_raw=float_raw_tgt, float_target_bin=float_bin_tgt, float_mode=mode,
        )
        loss.backward()
        shp = tuple(float_out[0].shape)
        print(f"[{mode}] params={model.num_params()/1e6:.2f}M | z={tuple(z.shape)} "
              f"float_out[0]={shp} | loss={parts['total']:.3f} (cls={parts['cls']:.3f}, reg={parts['reg']:.3f}) | backward OK")
