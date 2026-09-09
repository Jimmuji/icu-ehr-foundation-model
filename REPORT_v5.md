# EHR temporal foundation model on MIMIC-IV CLIF

## 1. Introduction

This project checks whether a small self supervised model can learn a useful representation of ICU time series. I pretrain a Transformer encoder on MIMIC-IV CLIF with masked reconstruction, freeze it, pool each stay into a single embedding, and measure how much clinical signal that embedding carries through linear probes, a survival model, clustering, and retrieval.

The current model, called v2 below, is a reconstruction only Transformer with about 2.16M parameters. The first attempt (v1) was a 228M EHRFormer with a VAE branch. The VAE collapsed during training (KL near zero by epoch 9) and the embedding compressed onto almost one axis, so in v2 I removed the VAE and kept the model small on purpose. The goal at this stage is a clean recipe with careful checks, not a final task model.

## 2. Dataset

**Source.** MIMIC-IV v3.1 plus the MIMIC-IV-Ext-CLIF v1.1.0 extension from PhysioNet (credentialed access). I used CLIF as the single event source because it already harmonises vital and lab names and units across the raw ICU tables.

**Cohort.** Adult ICU stays with length of stay between 24 hours and 30 days, first stay per patient: 50,022 stays from about 30,000 patients, 56.4M clinical events, 11% in hospital mortality.

**Representation.** Each stay is an hourly sequence, up to 512 timesteps, with 210 variables (11 categorical, 199 continuous): vitals, labs, nurse assessments (GCS, CAM, RASS, Braden, breathing trial status), medication administration, demographics, and the ICD chapter as a static feature at the first timestep. Continuous values are mapped to 256 quantile bins, with edges fit on the training split only; missing values are marked and contribute nothing to the input. So the tokenization covers structured variables and broad diagnosis chapters, but not the fine grained ICD codes, individual treatments, procedures, notes, or images that Apollo uses. This matters for the task analysis in Section 10.

**Downstream labels.** Seven binary tasks (in hospital mortality, stay over 7 days, 30 day readmission, celiac, MASLD, heart attack, stroke), three regressions (platelets, creatinine, SpO2), and one time to event task (in hospital death). Several diagnosis labels are rare in the test set (9 celiac, 125 MASLD, 332 heart attack positives), which limits what those tasks can show.

**Split.** Patients are split by first admission year using anchor_year_group, the variable MIMIC provides for real time analysis: train 2008-2016 (~42k stays), validation 2017-2019 (~5k), test 2020-2022 (~3k). No patient appears in two splits. I chose a temporal split over a random one because it matches how the model would actually be used, learning from past patients and predicting on new ones.

## 3. Network architecture

![v2 network architecture](outputs/final_results/network_architecture.png)

At each hour, categorical tokens and continuous value bins are embedded and summed into one vector, a learned admission aligned time embedding is added, and the sequence goes through the encoder with padded hours masked out. During pretraining, reconstruction heads predict the masked values. For downstream work I freeze the encoder and mean pool valid timesteps into one embedding per stay.

Two choices keep the model near 2M parameters. Values use a shared embedding table plus a small per feature offset, instead of 210 separate tables. On the output side, continuous features are predicted as quantile bins (shared bin head plus per feature offset, CE over 256 bins) rather than regressed as raw values, following Apollo and MOTOR. With this factorization about 82% of the parameters sit in the encoder layers, with roughly 9% in the embeddings and 9% in the heads; with one table per feature the embeddings alone would be near 10M and dominate the model.

| Component | Setting |
|---|---|
| Encoder | 6 layers, hidden 192, 4 heads, FF 384, pre LayerNorm, GELU, dropout 0.1 |
| Parameters | about 2.16M |
| Value embedding | shared value table + per feature offset (categorical and continuous separately) |
| Categorical heads | per feature linear, CE |
| Continuous head | shared bin head + per feature offset, CE over 256 bins |
| Patient embedding | mean pool over valid timesteps, 192-d |

## 4. Pretraining

![v2 CE train and validation loss](outputs/final_results/train_val_loss.png)

The objective is masked reconstruction: mask 50% of valid feature and timestep positions, replace them with the missing token, and predict the originals. The loss is the mean categorical CE plus the mean continuous bin CE, over masked positions only. Two simplifications are worth noting: the two loss terms are averaged within each group and then added, so each of the 11 categorical features carries more weight than each of the 199 continuous ones; and the input mask and the loss mask are the same positions, without the BERT style corruption scheme.

| Item | Setting |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 0.05, grad clip 1.0 |
| Schedule | 10% warmup, then cosine decay |
| Batch / precision | 32, bf16 |
| Epochs | 15, checkpoint at best validation loss |

Train loss falls from about 4.3 to 2.87 over the 15 epochs and validation tracks it closely the whole way. I read this as stable optimization with no clear sign of overfitting. The absolute value of a masked reconstruction loss is not very interpretable, so the shape of the curve is the point here. This is the contrast with v1, where the KL term collapsed and took the representation with it.

## 5. Downstream protocol

At inference there is no masking. I encode the observed values of a stay, mean pool over valid timesteps, and get one 192 dimensional embedding, which every evaluation below consumes. For the first 48h comparison I truncate each stay to its first 48 hours before encoding.

| Evaluation | Setup |
|---|---|
| Linear probe | logistic regression on frozen embeddings |
| MLP head | Linear(D→256), GELU, dropout 0.3, Linear(256→1), where D is the embedding dimension |
| Survival | Cox model on frozen embeddings |
| Clustering | UMAP plus k-means, no labels used |
| Retrieval | nearest neighbours in embedding space |
| Leave one feature out | drop one variable group, re-score mortality |

Probe settings are fixed on purpose: logistic regression with balanced class weights (C=1.0) on embeddings standardized with train statistics, no hyperparameter search, so the score reflects the embedding rather than probe tuning. Regression tasks use ridge (alpha 1.0) with targets winsorized to the 0.5 to 99.5 percentile range. The MLP head is trained with AdamW (lr 1e-3, batch 256) and early stopping on validation loss, with patience 5, up to 50 epochs, and a fixed seed. The whole protocol is conservative: a frozen probe measures what is already accessible in the representation, so the numbers are a lower bound and not directly comparable with models fine tuned per task.

Of the two head levels, the linear probe is the one I have run on the v2 embeddings, and its results are in Section 7. The MLP head comparison on these embeddings and end to end fine tuning with the encoder unfrozen are the next steps (fine tuning is implemented but waits on GPU availability), so the reported numbers are a lower bound on task performance.

## 6. Embedding sanity check

![Patient atlas, UMAP coloured by unsupervised phenotype with clinical annotation](outputs/final_results/annotated_phenotype_atlas.png)

UMAP of the patient embeddings, coloured by unsupervised phenotypes, with a short clinical annotation for each group. Similar patients fall into clinically distinct regions, and the embedding uses about 65 dimensions rather than collapsing onto one. The groups range from low risk (P0 at 1% mortality and P2 at <1% mortality) to critically ill (P1 at 26%), and two are disease enriched without using labels for clustering: P2 is a low risk cardiac group (AMI 1.4x) and P5 a stroke group (stroke 2.8x, 12% mortality). The survival differences between these groups are shown in Section 8.

![Nearest neighbour retrieval vs baseline rate](outputs/final_results/retrieval_clean.png)

Nearest neighbour retrieval: nearby patients in the embedding space share the same condition above baseline, especially for mortality (5.1x), stroke (3.5x), and heart attack (1.9x). This is a useful sanity check that distance in the embedding space reflects clinical similarity.

![Leave one feature out drivers of predicted mortality](outputs/final_results/loto_mortality.png)

Leave one feature out: the mortality signal is driven by clinically reasonable variables, such as delirium, age, heart rate, mental status, blood pressure, SpO2, GCS, and breathing trial status. I take this as a sign that the model is not relying on an obvious shortcut or artefact.

## 7. Downstream tasks

![Linear probe test AUROC per task](outputs/final_results/downstream_auroc_clean.png)

A logistic regression on top of the frozen embedding, evaluated on the test set. Mortality (0.92) and stroke (0.84) come out strongest, long stay and heart attack are moderate, and rare or weak signal tasks like celiac and readmission sit close to chance. One frozen representation already carries signal for a range of outcomes, though the strength varies a lot by task.

## 8. Time to event task

![Kaplan Meier survival by unsupervised phenotype](outputs/final_results/km_survival_by_phenotype.png)

This part uses the temporal nature of the data. A Cox model on the embedding ranks which patients die sooner reasonably well (C index 0.86). When I cluster the embedding without using any labels, the groups it finds have quite different survival, from about 54% to 91% at 30 days, and the difference is clearly significant (log rank p well below 0.001). The groups also line up with recognisable types: a stroke heavy group, a low risk cardiac group, and a sickest group.

## 9. Comparison with existing MIMIC work

I also looked at several studies that use MIMIC data and are close to this project. Harutyunyan et al. is the classic supervised MIMIC-III ICU benchmark. EHRMamba is a recent self supervised sequence model on MIMIC-IV, with task specific fine tuning. ETHOS uses a GPT style generative model on MIMIC-IV plus ED timelines. They differ from this work in cohort, tokenization, input window, labels, and evaluation protocol, so I used AUROC on roughly similar tasks as a coarse reference, not a strict comparison.

| | This work | EHRMamba | ETHOS | Harutyunyan 2019 |
|---|---|---|---|---|
| Data | MIMIC-IV CLIF (ICU) | MIMIC-IV | MIMIC-IV + ED | MIMIC-III |
| Model | 2.16M transformer | Mamba state space FM | GPT style decoder FM | LSTM (supervised) |
| Input tokens | 210 variables, binned values, 1h bins | code level events | code level timeline | engineered features |
| Pretraining | masked reconstruction | self supervised | next token (generative) | none |
| Downstream use | frozen embedding + linear probe | fine tune per task | zero shot generation | supervised per task |

AUROC score:

| Task | This work | EHRMamba | ETHOS | Harutyunyan |
|---|---|---|---|---|
| Mortality | 0.89 (in hospital, first 48h) | 0.98* (post discharge, 1 mo) | 0.92 (in hospital, at admission) | ~0.86 (in hospital, first 48h) |
| 30 day readmission | 0.54 | 0.68* (1 month) | 0.75 (hospital) | - |
| Prolonged LOS | 0.78 | 0.92* (>1 wk, first 24h) | - | - |

The closest comparison is early in hospital mortality, where this work reaches 0.89 against the classic benchmark's ~0.86 under the same first 48h setup; the whole stay probe in Section 7 is 0.92 but uses the full stay. Readmission and LOS are weaker than the larger or task specific models. Part of that difference is protocol: the cited numbers mostly come from task specific fine tuning, while ours is a frozen probe.

## 10. Performance gap

Mortality and survival work well because they are driven by acute physiology (vitals, labs, GCS, delirium), which is exactly the hourly signal the model reconstructs during pretraining, and the leave one feature out result supports this. The tasks that lag fail for three different reasons, so I would not treat them as one gap:

1. **Rare labels.** Celiac (9 test positives), MASLD (125), and heart attack (332) are dominated by sampling noise at this size; no head change fixes 9 positives. These tasks are underpowered here rather than evidence the model failed them.
2. **Input coverage.** Readmission and long stay depend heavily on post discharge and social factors that are not in the intra ICU stream. A model cannot recover information its inputs do not contain, so scaling the encoder alone is unlikely to close this.
3. **Tokenization.** Disease specific evidence often lives in fine grained ICD codes, treatments, procedures, and notes, which the current variable level tokenization does not keep. For the diagnosis tasks, richer event level tokenization in the Apollo style looks like the more promising change, more than adding parameters.

Two evaluation caveats also widen the apparent gap: long stays are truncated to the last 512 hours, which helps whole stay mortality but is unfair against first 24h or 48h benchmarks; and the frozen probe protocol is a lower bound against fine tuned baselines.

| Gap source | What to try next |
|---|---|
| Task setup / input window | report first 24h, first 48h, and whole stay separately; make the early window the main literature comparison |
| Tokenization | add finer diagnosis, medication, procedure, and treatment tokens |
| Model scale | larger hidden size and more layers once the pipeline is stable |
| Modality | clinical text or imaging if available |
| Tuning protocol | end to end fine tuning after the representation checks are stable |

## 11. Limitations and next steps

The model is small and trained on a subset of stays, so these numbers are a pipeline check, not a final result. The downstream results use frozen embeddings, which is a lower bound. The rare diagnosis tasks have too few positives to be conclusive, and the whole stay evaluation is not comparable to early prediction benchmarks.

Overall, the current results show that the pretrained representation contains useful information for mortality and survival prediction, while its performance is less consistent for readmission and diagnosis related tasks. Future work can explore richer tokenization of diagnoses and treatments similar to Apollo, a larger model and training cohort, and multimodal inputs such as clinical text or medical images.
