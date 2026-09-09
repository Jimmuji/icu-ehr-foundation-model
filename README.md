# EHR Temporal Foundation Model (MIMIC-IV CLIF)

A small temporal Transformer foundation model for ICU electronic health records, trained
with masked reconstruction on MIMIC-IV CLIF, plus a downstream evaluation suite (linear
probes, an MLP head, Cox survival analysis, clustering, retrieval, and leave-one-feature-out).

This repository contains **code only**. No patient-level data, result files or trained
checkpoints are included; the numbers live in the report. See [Data access](#data-access).

## What is here

- **`src/`** — Step 1 preprocessing: build the ICU cohort from MIMIC-IV, extract events from
  CLIF (vitals, labs, assessments) plus medications and diagnosis chapters, build the vocab
  and 256-bin quantile edges, tokenize each stay, and chunk to tensors.
- **`src_v2/`** — the v2 model and evaluation:
  - `small_ehr_model.py` — the ~2.16M-parameter Transformer (masked-feature autoencoder).
  - `train_small.py` — pretraining loop.
  - `03_linear_probe.py`, `05_finetune.py` — frozen-embedding linear probe and MLP head.
  - `09_survival_eval.py`, `12_km_survival.py` — Cox model and Kaplan–Meier by phenotype.
  - `11_cluster_v2.py`, `15_retrieval_precision.py`, `16_loto.py`, `21_first48h.py`, and the
    atlas scripts — the sanity checks and the first-48h re-evaluation.
- **`configs/`** — preprocessing and pretraining configuration.
- **`scripts/`** — environment setup and run helpers.
- **`REPORT_v5.md` / `REPORT_v5.pdf`** — the full technical write-up (architecture,
  pretraining, downstream tuning, and a task-by-task analysis of the performance gap).

## Model at a glance

| | |
|---|---|
| Input | ICU stay as an hourly sequence, up to 512 timesteps; 210 variables (11 categorical, 199 continuous) |
| Encoder | 6-layer Transformer, hidden 192, 4 heads, FF 384, pre-LayerNorm, ~2.16M params |
| Pretraining | masked-feature reconstruction (categorical CE + continuous 256-bin CE), 50% masking |
| Downstream | freeze the encoder, mean-pool valid timesteps to one patient embedding, then probe |
| Split | patient-level temporal: train 2008–2016, validation 2017–2019, test 2020–2022 |

See `REPORT_v5.md` for details and results.

## Data access

This project uses restricted data that **cannot be redistributed** here:

- **MIMIC-IV v3.1** and **MIMIC-IV-Ext-CLIF v1.1.0**, available from PhysioNet to
  credentialed users under a Data Use Agreement.

To reproduce, obtain access yourself (credentialing + DUA at PhysioNet), then point the
config at your local copies:

- `configs/preprocess.yaml`: set `paths.mimic_root` and `paths.clif_root`.

Note that several scripts contain absolute paths from the original training server (for
example under `/data/mimic_data/...`); adjust these to your environment.

## Usage

```bash
pip install -r requirements.txt

# 1. preprocess (needs local MIMIC-IV + CLIF)
bash scripts/run_all.sh

# 2. pretrain the small model
python src_v2/train_small.py --epochs 15 --batch-size 32 --lr 1e-3 --float-mode ce

# 3. downstream evaluation on the frozen embedding
python src_v2/03_linear_probe.py
python src_v2/05_finetune.py
python src_v2/09_survival_eval.py
```

## Status and limitations

This is a research work in progress, not a finished or production model. The model is small
and trained on a subset of stays, and the downstream numbers mostly use frozen embeddings,
which are a lower bound and are not directly comparable to fully fine-tuned models. The
results are best read as a check that the pretrained representation carries useful clinical
signal, strongest for mortality and survival and weaker for readmission and diagnosis tasks.
See the report for a fuller discussion.

## Acknowledgements

Preprocessing follows Apollo's discretise-and-reconstruct protocol, with 256
quantile bins on an hourly grid rather than Apollo's ten log-scale bins on raw
timestamps. The tensor format and the architecture follow
[EHRFormer](https://github.com/kaiwang13/EHRFormer), scaled down and without its VAE.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
