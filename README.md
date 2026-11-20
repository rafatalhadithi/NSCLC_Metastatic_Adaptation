# NSCLC Metastatic Adaptation: Multi-Modal Transcriptomic Pipeline

This repository contains the reproducible computational pipeline for ***"Machine Learning and Spatial Transcriptomics Identify a MUC1/SCGB3A2 Lineage Plasticity Signature Associated with Metastatic Adaptation in NSCLC."***

## Overview
By integrating single-cell RNA sequencing (scRNA-seq), 10x Visium spatial transcriptomics, and donor-aware machine learning, this hypothesis-generating framework characterizes epithelial lineage plasticity across primary and metastatic lung adenocarcinoma (LUAD). A Nested Patient-Level 5-Fold `GroupKFold` Random Forest primary-versus-metastatic classifier highlights **SCGB3A2**, **HP**, **MUC1**, and **SFTPB** among top discriminators, generalizing across sequencing platforms (10x Genomics to Smart-seq2) in 41 held-out biopsies from `PRJNA591860`. Biopsy-level pseudo-bulk GSEA, biopsy-level predicted extracellular ligand–receptor co-expression, Visium physical spatial block analyses, and multivariable survival modeling in resected neoadjuvant-naïve *EGFR*-mutant TCGA-LUAD are implemented end-to-end.

---

## Data Availability
Due to GitHub file size limits, raw sequencing archives and large `.h5ad` single-cell objects are hosted externally.

**Pre-Processed Single-Cell Objects (Zenodo):**
1. Download `GSE131907_FINAL_ANNOTATED.h5ad` (Discovery) and `maynard_annotated.h5ad` (External Validation) from Zenodo:  
   **https://zenodo.org/records/23512899?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjExYzFlMTBmLTdmYmQtNGVhNS1iYmNmLTIzODkzYTBhNzJiMCIsImRhdGEiOnt9LCJyYW5kb20iOiIyYjFlZGVmMzhmM2U0NTc3Zjc2MjcwZmFiNmZhMjNmYiJ9.swjgBGwqWAyDnL36unSsdy-QABep7NpuI5vfd9EXKQFYFfd85yO8PhYS5ApYlXvnYG4aS6M1-jUZCCu5Soljhw**
2. Place both `.h5ad` files into the `data/` directory of this repository before running the pipeline.

**Public Accessions:**
* **Discovery scRNA-seq Atlas:** `GSE131907` (Kim et al., 2020)
* **External Multi-Biopsy scRNA-seq Cohort:** `PRJNA591860` (Maynard et al., 2020)
* **10x Visium Spatial LUAD Dataset:** `GSE267960` / `GSM8282531` (Hong et al., 2025; place SpaceRanger output under `data/validation_cohorts/GSE267960/GSM8282531_S20-12254-L9/S20-12254-L9/`)
* **TCGA-LUAD Clinical & RSEM Cohort:** Pan-Cancer Atlas 2018 via cBioPortal (place under `data/tcga_luad/luad_tcga_pan_can_atlas_2018/`)

---

## Supplementary Data File Mapping

| Manuscript File Name | Description |
| :--- | :--- |
| `Supplementary_Data_1.csv` | Biopsy-level pseudo-bulk differential gene expression (`GSE131907`, Metastatic vs. Primary, Wilcoxon with BH adjustment). |
| `Supplementary_Data_2.csv` | Preranked MSigDB Hallmark GSEA summary for the Discovery cohort (`GSE131907`, Metastatic vs. Primary biopsies). |
| `Supplementary_Data_3.csv` | Unsupervised PAGA connectivity matrix across non-brain anatomical sites (`tLung`, `tL/B`, `mLN`, `PE`). |
| `Supplementary_Data_3B_Sensitivity.csv` | DPT and PAGA hyperparameter sensitivity grid (`n_neighbors` = 10, 15, 30; `n_pcs` = 20, 30, 40). |
| `Supplementary_Data_3C_Node_Sample_Counts.csv` | Epithelial cell counts and unique patient biopsy counts per anatomical node in `GSE131907`. |
| `Supplementary_Data_4.csv` | Nested Patient-Level 5-Fold `GroupKFold` Random Forest cross-validation metrics. |
| `Supplementary_Data_4B_LR_Biopsy_Stats.csv` | Biopsy-level predicted extracellular ligand–receptor co-expression SMDs, simultaneous 95% bootstrap CIs, and BH-FDR *q*-values. |
| `Supplementary_Data_5.csv` | Consensus Random Forest primary-versus-metastatic feature importance weights. |
| `Supplementary_Data_6.csv` | Exploratory Visium spot-level Spearman correlations with summed *HOXC* locus expression (`GSM8282531`). |
| `Supplementary_Data_7.csv` | Single-spot Visium correlation (with 2D physical block-bootstrap 95% CI) vs. 12×12 Spatial Grid Block Aggregation. |
| `Supplementary_Data_8.csv` | Held-out 41-biopsy external validation matrix (`PRJNA591860`: 14 Primary vs. 27 Metastatic biopsies) and ROC predictions. |
| `Supplementary_Data_9.csv` | Preranked MSigDB Hallmark GSEA summary for the External cohort (`PRJNA591860`, Metastatic vs. Primary biopsies). |
| `Supplementary_Data_10.csv` | Complete 49-biopsy / 33-patient clinical, genomic, anatomical, therapy, and cell-count manifest (`PRJNA591860`). |
| `Supplementary_Data_11.csv` | TCGA-LUAD univariable/multivariable Cox regressions (continuous & median-split) and Schoenfeld PH diagnostics. |
| `Supplementary_Data_12.csv` | Auditable patient-level neoadjuvant-naïve primary *EGFR*-mutant TCGA-LUAD dataset (*n* = 80). |

---

## Sequential Pipeline Execution (`notebooks/`)

1. **`01_preprocess_GSE131907.py`** — Quality control, normalization, PCA/UMAP/Leiden clustering, and multi-marker lineage validation (Outputs: Figure 1, Supp Figs S1–S2).
2. **`02_preprocess_Maynard.py`** — Audits `PRJNA591860` metadata and exports the 49-biopsy clinical/cellular manifest (`Supplementary_Data_10.csv`).
3. **`03_ML_Patient_CV.py`** — Nested Patient-Level 5-Fold `GroupKFold` Random Forest classifier (`GSE131907`, 25 biopsies) and held-out 41-biopsy external validation on `PRJNA591860` (`Supplementary_Data_4.csv`, `5.csv`, `8.csv`; Figures 5–6).
4. **`04_PseudoBulk_GSEA.py`** — Biopsy-level pseudo-bulk DGE and continuous preranked Hallmark GSEA across both cohorts (`Supplementary_Data_1.csv`, `2.csv`, `9.csv`; Figure 2).
5. **`05_PAGA_Trajectory.py`** — Exploratory cross-sectional DPT (anchored at the `tLung` PCA medoid) and PAGA continuum plus 3×3 parameter sensitivity (`Supplementary_Data_3.csv`, `3B`, `3C`; Figure 3).
6. **`06_Biopsy_Ligand_Receptor.py`** — Biopsy-level predicted extracellular ligand–receptor co-expression with simultaneous multiplicity-adjusted 95% bootstrap CIs (`Supplementary_Data_4B_LR_Biopsy_Stats.csv`; Figure 4).
7. **`07_Spatial_Simpson_Paradox.py`** — Exploratory physical slide Visium correlation (`GSM8282531`), 2D spatial block bootstrapping, and 12×12 Spatial Grid Block Aggregation (`Supplementary_Data_6.csv`, `7.csv`; Supp Figs S3–S4).
8. **`08_PPI_Network_Generator.py`** — Medium-confidence STRING (combined score >= 0.400) predicted functional association network and `.graphml` export (Figure 8).
9. **`09_TCGA_Survival_Prep.py`** — Neoadjuvant-naïve *EGFR*-mutant TCGA-LUAD cohort construction, `log2(RSEM + 1)` composite Z-scoring, univariable/multivariable Cox models, PH diagnostics, and DPT cross-correlation (`Supplementary_Data_11.csv`, `12.csv`; Figure 7A).
10. **`10_Plot_Survival_R.R`** — R `survival` / `survminer` Kaplan–Meier plots and risk tables for Overall Survival (Figure 7B) and Progression-Free Survival (Figure 9).

---

## Installation & Dependencies

```bash
conda create -n nsclc-plasticity python=3.11
conda activate nsclc-plasticity

pip install scanpy>=1.9.0 pandas>=1.5.0 numpy>=1.23.0 scikit-learn>=1.2.0 decoupler-py>=1.4.0 gseapy>=1.0.0 networkx>=2.8.0 lifelines>=0.27.0 statsmodels>=0.14.0 matplotlib seaborn requests

For Script 10 in R >= 4.3: install.packages(c("survival", "survminer", "ggplot2", "readr"))
