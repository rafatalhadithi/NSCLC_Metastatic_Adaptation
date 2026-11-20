import os
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5

def main():
    print("--- SCRIPT 01: GSE131907 PREPROCESSING & MULTI-MARKER VALIDATION ---")
    raw_path = os.path.join(DATA_DIR, 'GSE131907_RAW_merged.h5ad')
    annot_path = os.path.join(DATA_DIR, 'GSE131907_FINAL_ANNOTATED.h5ad')

    if os.path.exists(raw_path) and not os.path.exists(annot_path):
        adata = sc.read_h5ad(raw_path)
        adata.X = sp.csr_matrix(adata.X)
        adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(adata, min_genes=200)
        adata = adata[(adata.obs.n_genes_by_counts < 5000) & (adata.obs.pct_counts_mt < 10), :].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata.raw = adata

        adata_hvg = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, svd_solver='arpack', random_state=42)
        sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=40, random_state=42)
        sc.tl.umap(adata_hvg, random_state=42)
        sc.tl.leiden(adata_hvg, resolution=0.5, random_state=42)

        adata.obs['leiden'] = adata_hvg.obs['leiden']
        adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']

        cluster_map = {'1': 'T-Cells', '2': 'Macrophages', '3': 'NK / Cytotoxic Cells', '4': 'B-Cells'}
        for c in ['9', '11', '12', '13', '20', '21', '24', '25', '26']:
            cluster_map[c] = 'Epithelial / Tumor Cells'
        adata.obs['cell_type'] = adata.obs['leiden'].map(lambda x: cluster_map.get(x, f"Cluster {x}")).astype('category')
        adata.write(annot_path)
    else:
        adata = sc.read_h5ad(annot_path)

    raw_d = adata.raw.to_adata()
    raw_d.obs = adata.obs.copy()
    raw_d.var['mt'] = raw_d.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(raw_d, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    valid_main = {'T-Cells', 'Macrophages', 'NK / Cytotoxic Cells', 'B-Cells'}
    def map_ct(x):
        s = str(x).strip()
        if 'Tumor' in s or 'Epithelial' in s:
            return 'Epithelial / Tumor Cells'
        return s if s in valid_main else 'Other / Unassigned'

    adata.obs['Clean_Cell_Type'] = pd.Categorical(
        adata.obs['cell_type'].apply(map_ct),
        categories=['Epithelial / Tumor Cells', 'T-Cells', 'Macrophages', 'NK / Cytotoxic Cells', 'B-Cells', 'Other / Unassigned']
    )
    anat_map = {
        'nLung': 'Normal Tissue (nLung/nLN)', 'nLN': 'Normal Tissue (nLung/nLN)',
        'tLung': 'Primary LUAD (tLung/tL/B)', 'tL/B': 'Primary LUAD (tLung/tL/B)',
        'mLN': 'Metastatic LUAD (mLN/PE/mBrain)', 'PE': 'Metastatic LUAD (mLN/PE/mBrain)',
        'mBrain': 'Metastatic LUAD (mLN/PE/mBrain)'
    }
    adata.obs['Anatomical_Category'] = pd.Categorical(
        adata.obs['Sample_Origin'].astype(str).map(lambda x: anat_map.get(x, 'Normal Tissue (nLung/nLN)')),
        categories=['Normal Tissue (nLung/nLN)', 'Primary LUAD (tLung/tL/B)', 'Metastatic LUAD (mLN/PE/mBrain)']
    )

    # Figure 1
    fig1, axes1 = plt.subplots(1, 2, figsize=(14.8, 5.8))
    sc.pl.umap(adata, color='Clean_Cell_Type',
               palette=['#D62828', '#1D3557', '#2A9D8F', '#F4A261', '#6A4C93', '#D3D3D3'],
               ax=axes1[0], show=False, title='A. Microenvironmental Cell-Type Compartments', frameon=True)
    sc.pl.umap(adata, color='Anatomical_Category',
               palette=['#2A9D8F', '#457B9D', '#E63946'],
               ax=axes1[1], show=False, title='B. Anatomical Tissue Origin (GSE131907)', frameon=True)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIG_DIR, 'Figure_1_Microenvironment_UMAP.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig1)

    # Supplementary Figure S1
    rng = np.random.default_rng(42)
    qc_sub = raw_d.obs.iloc[rng.choice(raw_d.n_obs, size=min(15000, raw_d.n_obs), replace=False)]
    fig_s1, axes_s1 = plt.subplots(1, 3, figsize=(14.0, 4.8))
    for ax_i, (col_q, tit_q, clr_q, ylab_q) in zip(axes_s1, [
        ('n_genes_by_counts', 'A. Detected Genes per Cell', '#457B9D', 'Number of Genes'),
        ('total_counts', 'B. Total Transcripts (UMI/TPM)', '#2A9D8F', 'Total Expression Signal'),
        ('pct_counts_mt', 'C. Mitochondrial Transcript Fraction', '#E63946', 'Mitochondrial (%)')
    ]):
        sns.violinplot(y=qc_sub[col_q], color=clr_q, ax=ax_i, inner='quartile', cut=0)
        ax_i.set_title(tit_q, fontweight='bold', fontsize=12)
        ax_i.set_ylabel(ylab_q, fontsize=11.5, fontweight='bold')
        sns.despine(ax=ax_i)
    fig_s1.tight_layout()
    fig_s1.savefig(os.path.join(FIG_DIR, 'Supplementary_Figure_S1_QC_Metrics.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig_s1)

    # Supplementary Figure S2
    raw_d.obs['Clean_Cell_Type'] = adata.obs['Clean_Cell_Type']
    raw_lab = raw_d[raw_d.obs['Clean_Cell_Type'] != 'Other / Unassigned'].copy()
    raw_lab.obs['Clean_Cell_Type'] = raw_lab.obs['Clean_Cell_Type'].cat.remove_unused_categories()
    markers = [g for g in ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'PTPRC', 'CD3D', 'CD3E',
                           'NKG7', 'GNLY', 'TYROBP', 'CD68', 'LYZ', 'CD79A', 'MS4A1'] if g in raw_lab.var_names]
    dp = sc.pl.dotplot(raw_lab, markers, groupby='Clean_Cell_Type', standard_scale='var',
                       swap_axes=True, show=False, return_fig=True, figsize=(6.8, 8.8))
    dp.savefig(os.path.join(FIG_DIR, 'Supplementary_Figure_S2_MultiMarker_Dotplot.pdf'), dpi=600, bbox_inches='tight')
    plt.close('all')
    print("--- SCRIPT 01 COMPLETE ---")

if __name__ == "__main__":
    main()
