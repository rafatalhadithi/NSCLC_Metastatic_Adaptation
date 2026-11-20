import os
import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    print("--- SCRIPT 05: EXPLORATORY CROSS-SECTIONAL CONTINUUM, PAGA & SENSITIVITY ---")
    adata = sc.read_h5ad(os.path.join(DATA_DIR, 'GSE131907_FINAL_ANNOTATED.h5ad'))
    tum = adata[(adata.obs['cell_type'].astype(str).str.contains('Epithelial|Tumor', case=False, na=False)) &
                (adata.obs['Sample_Origin'].isin(['tLung', 'tL/B', 'PE', 'mLN']))].copy()
    tum.obs['Anatomical_Site'] = pd.Categorical(tum.obs['Sample_Origin'].astype(str), categories=['PE', 'mLN', 'tL/B', 'tLung'])

    if 'X_pca' not in tum.obsm:
        sc.tl.pca(tum, svd_solver='arpack', random_state=42)
    sc.pp.neighbors(tum, n_neighbors=15, n_pcs=30, random_state=42)
    sc.tl.diffmap(tum, random_state=42)

    tlung_idx = np.where(tum.obs['Anatomical_Site'] == 'tLung')[0]
    tlung_pca = tum.obsm['X_pca'][tlung_idx, :30]
    medoid = int(tlung_idx[np.argmin(np.linalg.norm(tlung_pca - tlung_pca.mean(axis=0), axis=1))])
    tum.uns['iroot'] = medoid
    sc.tl.dpt(tum)
    sc.tl.paga(tum, groups='Anatomical_Site')

    cats = list(tum.obs['Anatomical_Site'].cat.categories)
    conn_df = pd.DataFrame(tum.uns['paga']['connectivities'].todense(), index=cats, columns=cats)
    conn_df.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_3.csv'))

    counts_df = tum.obs.groupby('Anatomical_Site', observed=True).agg(
        N_Cells=('Sample', 'count'), N_Unique_Biopsies=('Sample', 'nunique')
    ).reset_index()
    counts_df.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_3C_Node_Sample_Counts.csv'), index=False)

    base_dpt = tum.obs['dpt_pseudotime'].values.copy()
    sens = []
    for k_nn in [10, 15, 30]:
        for n_pc in [20, 30, 40]:
            tmp = tum.copy()
            sc.pp.neighbors(tmp, n_neighbors=k_nn, n_pcs=n_pc, random_state=42)
            sc.tl.diffmap(tmp, random_state=42)
            tmp.uns['iroot'] = medoid
            sc.tl.dpt(tmp)
            sc.tl.paga(tmp, groups='Anatomical_Site')
            cm = pd.DataFrame(tmp.uns['paga']['connectivities'].todense(), index=cats, columns=cats)
            r_s, _ = spearmanr(base_dpt[np.isfinite(base_dpt)], tmp.obs['dpt_pseudotime'].values[np.isfinite(base_dpt)])
            sens.append({
                'n_neighbors': k_nn, 'n_pcs': n_pc,
                'DPT_Spearman_vs_Baseline': round(r_s, 4),
                'PAGA_tLB_to_PE': round(float(cm.loc['tL/B', 'PE']), 4),
                'PAGA_tLung_to_mLN': round(float(cm.loc['tLung', 'mLN']), 4)
            })
    pd.DataFrame(sens).to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_3B_Sensitivity.csv'), index=False)

    # Export Cytoscape GraphML for Panel 3C & Plot 3-Panel Figure 3
    site_colors = {'PE': '#1f77b4', 'mLN': '#ff7f0e', 'tL/B': '#2ca02c', 'tLung': '#d62728'}
    G_paga = nx.Graph()
    c_idx = counts_df.set_index('Anatomical_Site')
    for s in cats:
        G_paga.add_node(s, display_label=s, n_cells=int(c_idx.loc[s, 'N_Cells']),
                        n_biopsies=int(c_idx.loc[s, 'N_Unique_Biopsies']), hex_color=site_colors[s])
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            w = float(conn_df.iloc[i, j])
            if w > 0:
                G_paga.add_edge(cats[i], cats[j], weight=round(w, 6), connectivity_label=f"{w:.3f}")
    nx.write_graphml(G_paga, os.path.join(FIG_DIR, 'Figure_3C_PAGA_Connectivity.graphml'), named_key_ids=True)

    fig3, axes3 = plt.subplots(1, 3, figsize=(16.0, 4.9))
    sc.pl.umap(tum, color='Anatomical_Site', palette=[site_colors[k] for k in cats],
               ax=axes3[0], show=False, title='A. Epithelial Anatomical Origin', frameon=True)
    sc.pl.umap(tum, color='dpt_pseudotime', cmap='viridis',
               ax=axes3[1], show=False, title='B. Cross-Sectional DPT Continuum', frameon=True)

    ax_p = axes3[2]
    pos_p = {'tLung': (-0.75, -0.65), 'tL/B': (0.75, -0.65), 'mLN': (-0.55, 0.72), 'PE': (0.65, 0.62)}
    for u, v, d in G_paga.edges(data=True):
        w = d['weight']
        if w > 0.01:
            ax_p.plot([pos_p[u][0], pos_p[v][0]], [pos_p[u][1], pos_p[v][1]],
                      color='#333333', lw=max(1.0, w * 14.0), alpha=0.8, zorder=1)
            ax_p.text((pos_p[u][0] + pos_p[v][0]) / 2, (pos_p[u][1] + pos_p[v][1]) / 2, f"{w:.2f}",
                      fontsize=9, fontweight='bold', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='gray', alpha=0.9), zorder=3)
    for n_k, (px, py) in pos_p.items():
        ax_p.scatter(px, py, s=1050, color=site_colors[n_k], edgecolors='black', linewidths=1.5, zorder=4)
        ax_p.text(px, py, n_k, fontsize=10.5, fontweight='bold', color='white', ha='center', va='center', zorder=5)
    ax_p.set_xlim(-1.15, 1.15)
    ax_p.set_ylim(-1.05, 1.05)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    ax_p.set_title('C. Unsupervised PAGA Connectivity', fontweight='bold', fontsize=12)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIG_DIR, 'Figure_3_Transcriptomic_Continuum.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig3)
    print("--- SCRIPT 05 COMPLETE ---")

if __name__ == "__main__":
    main()
