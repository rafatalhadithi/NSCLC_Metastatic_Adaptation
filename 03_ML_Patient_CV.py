import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def robust_patient_cv(X, y, groups, n_splits=5, n_features=50):
    gkf = GroupKFold(n_splits=n_splits)
    metrics = {'accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []}
    importances = []
    for tr, te in gkf.split(X, y, groups=groups):
        fs = SelectKBest(score_func=f_classif, k=n_features)
        X_tr = fs.fit_transform(X.iloc[tr], y.iloc[tr])
        X_te = fs.transform(X.iloc[te])
        genes = X.columns[fs.get_support()]

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_tr, y.iloc[tr])
        preds = rf.predict(X_te)

        importances.append(pd.Series(rf.feature_importances_, index=genes))
        metrics['accuracy'].append(accuracy_score(y.iloc[te], preds))
        metrics['f1_macro'].append(f1_score(y.iloc[te], preds, average='macro'))
        metrics['precision_macro'].append(precision_score(y.iloc[te], preds, average='macro', zero_division=0))
        metrics['recall_macro'].append(recall_score(y.iloc[te], preds, average='macro', zero_division=0))

    mean_m = pd.DataFrame(metrics).mean().to_frame(name='Mean_Value')
    mean_imp = pd.concat(importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False).to_frame(name='Mean_Importance')
    return mean_m, mean_imp

def main():
    print("--- SCRIPT 03: NESTED GROUPKFOLD CLASSIFIER & HELD-OUT 41-BIOPSY EXTERNAL VALIDATION ---")
    adata_d = sc.read_h5ad(os.path.join(DATA_DIR, 'GSE131907_FINAL_ANNOTATED.h5ad'))
    tum_d = adata_d[(adata_d.obs['cell_type'].astype(str).str.contains('Epithelial|Tumor', case=False, na=False)) &
                    (adata_d.obs['Sample_Origin'].isin(['tLung', 'tL/B', 'PE', 'mLN']))].copy()
    tum_d.obs['Target'] = np.where(tum_d.obs['Sample_Origin'].isin(['PE', 'mLN']), 1, 0)

    X_d = pd.DataFrame(tum_d.X.toarray() if hasattr(tum_d.X, 'toarray') else tum_d.X,
                       index=tum_d.obs_names, columns=tum_d.var_names)
    metrics, importances = robust_patient_cv(X_d, tum_d.obs['Target'], tum_d.obs['Sample'])
    metrics.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_4.csv'))
    importances.index.name = 'Gene'
    importances.reset_index().to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_5.csv'), index=False)

    # Figure 5
    top10 = importances.head(10).reset_index()
    top10.columns = ['Gene', 'Importance_Score']
    fig5, ax5 = plt.subplots(figsize=(7.5, 5.5))
    sns.barplot(data=top10, x='Importance_Score', y='Gene', hue='Gene', palette='mako', edgecolor='black', legend=False, ax=ax5)
    ax5.set_title('Primary-vs.-Metastatic Lineage Classifier Feature Importance\n(Nested Patient-Level GroupKFold CV)', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Mean Decrease in Impurity (Importance Weight)', fontweight='bold')
    ax5.set_ylabel('')
    sns.despine(ax=ax5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(FIG_DIR, 'Figure_5_ML_Features.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig5)

    # Held-Out 41-Biopsy External Validation on PRJNA591860
    raw_d = adata_d.raw.to_adata()
    raw_d.obs = adata_d.obs.copy()
    tum_raw = raw_d[(raw_d.obs['cell_type'].astype(str).str.contains('Epithelial|Tumor', case=False, na=False)) &
                    (raw_d.obs['Sample_Origin'].isin(['tLung', 'tL/B', 'PE', 'mLN']))].copy()
    tum_raw.obs['Is_Met'] = np.where(tum_raw.obs['Sample_Origin'].isin(['PE', 'mLN']), 1, 0)

    adata_m = sc.read_h5ad(os.path.join(DATA_DIR, 'maynard_annotated.h5ad'))
    obs_m = adata_m.obs.copy()
    if adata_m.raw is not None:
        adata_m = adata_m.raw.to_adata()
        adata_m.obs = obs_m
    sc.pp.normalize_total(adata_m, target_sum=1e4)
    sc.pp.log1p(adata_m)

    epi_m = adata_m[(adata_m.obs['general_annotation'].astype(str).str.lower().str.contains('epi')) &
                    (adata_m.obs['primary_or_metastaic'].isin(['Primary', 'Metastatic']))].copy()
    epi_m.obs['Is_Met'] = np.where(epi_m.obs['primary_or_metastaic'] == 'Metastatic', 1, 0)

    shared = [g for g in importances.head(50).index.tolist() if g in tum_raw.var_names and g in epi_m.var_names]
    b_tr = pd.DataFrame(tum_raw[:, shared].X.toarray() if hasattr(tum_raw[:, shared].X, 'toarray') else tum_raw[:, shared].X, columns=shared)
    b_tr['Sample'] = tum_raw.obs['Sample'].astype(str).values
    b_tr['Is_Met'] = tum_raw.obs['Is_Met'].values
    b_tr = b_tr.groupby(['Sample', 'Is_Met'])[shared].mean().reset_index()

    b_te = pd.DataFrame(epi_m[:, shared].X.toarray() if hasattr(epi_m[:, shared].X, 'toarray') else epi_m[:, shared].X, columns=shared)
    b_te['Sample'] = epi_m.obs['sample_name'].astype(str).values
    b_te['Is_Met'] = epi_m.obs['Is_Met'].values
    b_te = b_te.groupby(['Sample', 'Is_Met'])[shared].mean().reset_index()

    rf_lock = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced').fit(
        b_tr[shared].rank(axis=1, pct=True).values, b_tr['Is_Met'].values
    )
    rf_probs = rf_lock.predict_proba(b_te[shared].rank(axis=1, pct=True).values)[:, 1]
    auc_rf = roc_auc_score(b_te['Is_Met'].values, rf_probs)

    alv_genes = ['SCGB3A2', 'SFTPB', 'MUC1', 'SCGB3A1']
    alv_score = -b_te[alv_genes].mean(axis=1).values
    auc_alv = roc_auc_score(b_te['Is_Met'].values, alv_score)
    _, p_alv = mannwhitneyu(alv_score[b_te['Is_Met']==1], alv_score[b_te['Is_Met']==0])

    b_te['Site_Group'] = np.where(b_te['Is_Met'] == 1, 'Metastatic\n(n=27 biopsies)', 'Primary\n(n=14 biopsies)')
    b_te['Locked_RF_Met_Prob'] = rf_probs
    b_te['Alveolar_Depletion_Score'] = alv_score
    b_te.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_8.csv'), index=False)

    # Figure 6
    val_genes = ['SCGB3A2', 'SFTPB', 'MUC1', 'HP']
    p_raw = [mannwhitneyu(b_te[b_te['Is_Met']==1][g], b_te[b_te['Is_Met']==0][g])[1] for g in val_genes]
    q_adj = multipletests(p_raw, method='fdr_bh')[1]

    fig6 = plt.figure(figsize=(13.5, 8.2))
    gs = fig6.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.35, wspace=0.32)
    ax_roc = fig6.add_subplot(gs[:, 0])
    fpr1, tpr1, _ = roc_curve(b_te['Is_Met'].values, rf_probs)
    fpr2, tpr2, _ = roc_curve(b_te['Is_Met'].values, alv_score)
    ax_roc.plot(fpr1, tpr1, color='#1D3557', lw=2.3, label=f"Locked 47-Gene RF\n(AUC = {auc_rf:.3f})")
    ax_roc.plot(fpr2, tpr2, color='#E63946', lw=2.3, label=f"Core Alveolar-Secretory Module\n(AUC = {auc_alv:.3f}, p = {p_alv:.4f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.1, alpha=0.6)
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax_roc.set_title('A. Held-Out Biopsy External Validation\n(PRJNA591860, N = 41 Biopsies)', fontweight='bold')
    ax_roc.legend(loc='lower right', frameon=True, fontsize=10)
    sns.despine(ax=ax_roc)

    for idx, (g, pv, qv, (r_i, c_i)) in enumerate(zip(val_genes, p_raw, q_adj, [(0, 1), (0, 2), (1, 1), (1, 2)])):
        ax_g = fig6.add_subplot(gs[r_i, c_i])
        sns.boxplot(data=b_te, x='Site_Group', y=g, order=['Primary\n(n=14 biopsies)', 'Metastatic\n(n=27 biopsies)'],
                    palette=['#457B9D', '#E63946'], width=0.48, fliersize=0, ax=ax_g, boxprops=dict(alpha=0.85))
        sns.stripplot(data=b_te, x='Site_Group', y=g, order=['Primary\n(n=14 biopsies)', 'Metastatic\n(n=27 biopsies)'],
                      color='black', size=5.5, jitter=0.15, alpha=0.75, ax=ax_g)
        ax_g.set_xlabel('')
        ax_g.set_ylabel('Biopsy Mean Log1p Expression', fontsize=10)
        ax_g.set_title(f"{['B','C','D','E'][idx]}. {g}\n(p = {pv:.4f}, FDR q = {qv:.4f})", fontweight='bold', fontsize=11)
        sns.despine(ax=ax_g)

    fig6.savefig(os.path.join(FIG_DIR, 'Figure_6_External_Biopsy_Validation.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig6)
    print("--- SCRIPT 03 COMPLETE ---")

if __name__ == "__main__":
    main()
