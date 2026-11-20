import os
import pandas as pd
import numpy as np
import scanpy as sc
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5

def run_biopsy_gsea(adata_ep, sample_col, met_mask):
    samples = adata_ep.obs[sample_col].unique()
    rows, flags = [], []
    for s in samples:
        m = (adata_ep.obs[sample_col] == s).values
        rows.append(np.asarray(adata_ep.X[m].mean(axis=0)).flatten())
        flags.append(int(met_mask[m][0]))
    pb = pd.DataFrame(rows, columns=adata_ep.var_names)
    flags = np.array(flags)

    pb_adata = sc.AnnData(X=pb.values, var=pd.DataFrame(index=pb.columns),
                          obs=pd.DataFrame({'Group': np.where(flags == 1, 'Metastatic', 'Primary')}))
    sc.tl.rank_genes_groups(pb_adata, groupby='Group', reference='Primary', method='wilcoxon')
    dge = sc.get.rank_genes_groups_df(pb_adata, group='Metastatic')

    lfc = (pb[flags == 1].mean(axis=0) - pb[flags == 0].mean(axis=0)).dropna()
    lfc = lfc[lfc != 0].sort_values(ascending=False)
    res = gp.prerank(rnk=pd.DataFrame({'g': lfc.index, 'v': lfc.values}),
                     gene_sets='MSigDB_Hallmark_2020', threads=4, min_size=5, max_size=500,
                     permutation_num=1000, seed=42).res2d.copy()
    res['Clean_Term'] = (res['Term'].astype(str)
                         .str.replace('Hallmark ', '', regex=False)
                         .str.replace('_', ' ', regex=False)
                         .str.replace('  ', ' ', regex=False)
                         .str.replace('heme Metabolism', 'Heme Metabolism', regex=False))
    for c in ['NES', 'NOM p-val', 'FDR q-val']:
        res[c] = pd.to_numeric(res[c], errors='coerce')
    return dge, res

def main():
    print("--- SCRIPT 04: BIOPSY-LEVEL PSEUDO-BULK DGE & CROSS-COHORT GSEA ---")
    adata_d = sc.read_h5ad(os.path.join(DATA_DIR, 'GSE131907_FINAL_ANNOTATED.h5ad'))
    raw_d = adata_d.raw.to_adata()
    raw_d.obs = adata_d.obs.copy()
    tum_d = raw_d[(raw_d.obs['cell_type'].astype(str).str.contains('Epithelial|Tumor', case=False, na=False)) &
                  (raw_d.obs['Sample_Origin'].isin(['tLung', 'tL/B', 'PE', 'mLN']))].copy()
    dge_d, disc_df = run_biopsy_gsea(tum_d, 'Sample', tum_d.obs['Sample_Origin'].isin(['PE', 'mLN']).values)
    dge_d.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_1.csv'), index=False)
    disc_df.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_2.csv'), index=False)

    adata_m = sc.read_h5ad(os.path.join(DATA_DIR, 'maynard_annotated.h5ad'))
    obs_m = adata_m.obs.copy()
    if adata_m.raw is not None:
        adata_m = adata_m.raw.to_adata()
        adata_m.obs = obs_m
    sc.pp.normalize_total(adata_m, target_sum=1e4)
    sc.pp.log1p(adata_m)
    epi_m = adata_m[(adata_m.obs['general_annotation'].astype(str).str.lower().str.contains('epi')) &
                    (adata_m.obs['primary_or_metastaic'].isin(['Primary', 'Metastatic']))].copy()
    _, val_df = run_biopsy_gsea(epi_m, 'sample_name', (epi_m.obs['primary_or_metastaic'] == 'Metastatic').values)
    val_df.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_9.csv'), index=False)

    merged = pd.merge(disc_df[['Clean_Term', 'NES', 'NOM p-val', 'FDR q-val']],
                      val_df[['Clean_Term', 'NES', 'NOM p-val', 'FDR q-val']],
                      on='Clean_Term', suffixes=('_Disc', '_Val'))
    top_plot = merged.sort_values('FDR q-val_Disc').head(10).sort_values('NES_Disc', ascending=True)

    y_pos, h = np.arange(len(top_plot)), 0.36
    fig2, ax2 = plt.subplots(figsize=(11.2, 6.3))
    ax2.barh(y_pos + h/2, top_plot['NES_Disc'], height=h, color='#3B75AF', edgecolor='black', label='Discovery Cohort (GSE131907: Met vs. Prim)')
    ax2.barh(y_pos - h/2, top_plot['NES_Val'], height=h, color='#D63B44', edgecolor='black', label='External Cohort (PRJNA591860: Met vs. Prim)')
    for idx, (_, r) in enumerate(top_plot.iterrows()):
        if r['FDR q-val_Disc'] < 0.05:
            ax2.text(r['NES_Disc'] + (0.05 if r['NES_Disc'] >= 0 else -0.05), idx + h/2, '*',
                     va='center', ha='left' if r['NES_Disc'] >= 0 else 'right', fontweight='bold', fontsize=13)
        if r['FDR q-val_Val'] < 0.05:
            ax2.text(r['NES_Val'] + (0.05 if r['NES_Val'] >= 0 else -0.05), idx - h/2, '*',
                     va='center', ha='left' if r['NES_Val'] >= 0 else 'right', fontweight='bold', fontsize=13)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_plot['Clean_Term'], fontsize=11)
    ax2.axvline(0, color='black', lw=1.2)
    ax2.set_xlim(-2.3, 2.5)
    ax2.set_xlabel('Normalized Enrichment Score (NES) [Metastatic vs. Primary Biopsies]', fontsize=12, fontweight='bold')
    ax2.set_title('Cross-Cohort Hallmark Pathway Enrichment in Metastatic vs. Primary LUAD\n(* denotes Benjamini-Hochberg FDR q < 0.05)', fontsize=12.5, fontweight='bold', pad=12)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 0.02), frameon=True, framealpha=0.95, fontsize=10)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, 'Figure_2_CrossCohort_GSEA.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig2)
    print("--- SCRIPT 04 COMPLETE ---")

if __name__ == "__main__":
    main()
