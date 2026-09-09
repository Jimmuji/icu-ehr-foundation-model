# EHRFormer hyperparameters — from the upstream repo

Source: [`github.com/kaiwang13/EHRFormer`](https://github.com/kaiwang13/EHRFormer) (`configs/pretrain.json`, `pretrain.py`, `ehrformer.py`, `ehr_model_module_pretrain.py`).

## Architecture (fixed; do not change)

| Block | Type | Details |
|---|---|---|
| Per-timestep feature embedding | `BertModel` (1 layer) | hidden=768, heads=12, FFN=3072, no position emb, vocab=`n_cat_values + n_float_values + 2`, type_vocab=`n_cat_feats + n_float_feats + 1`, max_pos=8192 |
| Temporal trunk | `GPT2Model` (base "gpt2") | 12 layers, 12 heads, hidden=768, n_positions extended to **8192**, time_index used as position_ids |
| VAE `μ` encoder | `BertEncoder` 2 layers | 12 heads, hidden=768 |
| VAE `log σ` encoder | `BertEncoder` 2 layers | 12 heads, hidden=768 |
| VAE decoder | `BertEncoder` 2 layers | 12 heads, hidden=768 |
| Pretrain head | MLP per feature | Linear(768→256)→ReLU→Linear(256→`{2 or 1}`), one head per cat / float feature |

**This is a VAE-Transformer, not a plain MLM**. Loss has 3 terms (see below). Total params ~150M with default config.

## Loss

```python
elbo_loss = cls_loss + reg_loss + kl_loss
# cls_loss: CE over masked categorical features
# reg_loss: MSE over masked float features (quantized values treated as labels, not regression target)
# kl_loss : -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

KL weight = 1.0 (no β-VAE annealing in upstream).

## Optimization

| Param | Value | Notes |
|---|---|---|
| Optimizer | AdamW | |
| Learning rate | **1e-4** | |
| Weight decay | **0.05** | |
| Scheduler | CosineAnnealingWarmupRestarts | from `cosine_annealing_warmup` package |
| Warmup | **10% of n_epoch** | |
| min_lr | 1e-8 | |
| Epochs | **300** (default) | reduce for MIMIC; see "Adaptations" below |
| Batch size | 200 | per device |
| Precision | **bf16-mixed** | H100 native |
| Strategy | `ddp_find_unused_parameters_true` | multi-GPU |
| sync_batchnorm | True | |

## Data hyperparameters (sample data → MIMIC adaptation)

| Field | Sample config | MIMIC target | Why |
|---|---|---|---|
| `n_category_feats` | 1 | ~50-100 | from our `feature_index.parquet` |
| `n_float_feats` | 4 | ~200-400 | from our `feature_index.parquet` |
| `n_category_values` | 2 | max per-feature vocab size (cap?) | from `vocab_cat.json` |
| `n_float_values` | **256** | **256** | EHRFormer default |
| `seq_max_len` | 16 | **512** | matches our `time_binning.max_seq_len` |
| `mask_ratio` | **0.15** | 0.15 | BERT default, NOT 50% as README claims |

## Cross-validation setup

Upstream uses 10-fold CV (`dataset_col: "dataset_fold10"`).
For MIMIC pretraining we use a 3-way split (train/val/test by patient) — simpler and standard.

## Adaptations recommended for MIMIC (~50k stays, ~30k patients)

| Default | Recommendation | Why |
|---|---|---|
| `n_epoch: 300` | **30-60** | data is ~100x smaller than production. Monitor val_loss, early-stop. |
| `batch_size: 200` | 64-128 per GPU | seq_max_len went 16→512 = 32× more tokens per sample. |
| `n_gpus: [7]` | 4 (negotiable up to 8) | the bottleneck is data not compute. |
| `mask_ratio: 0.15` | 0.15 | keep default; literature confirms. |
| Effective LR | scale linearly with batch | new_lr = 1e-4 × (effective_batch / 200) |

## Compute budget estimate

With 4× H100, bf16, 50k stays × T=512 × ~F=300:
- ~50 epochs × ~100 steps/epoch (depends on batch) ≈ **1-2 hours pretraining**

## Files in upstream

```
EHRFormer/
├── ehrformer.py                    # model (EHREmbedding + GPT2 + VAE + head)
├── pretrain.py                     # PL Trainer entry, sets GPT2Config + n_positions=8192
├── finetune.py                     # downstream tasks
├── ehr_model_module_pretrain.py    # LightningModule with ELBO loss
├── ehr_model_module_finetune.py
├── ehr_dataset_chunk.py            # data loader with chunked .pt caching
├── configs/{pretrain,finetune}.json
├── Utils.py, ddp_Utils.py, lr_monitor2.py
└── sample_data/ (placeholder)
```
