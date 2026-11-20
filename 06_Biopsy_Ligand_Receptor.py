import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5

def bootstrap_ci(x, y, alpha=0.05, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(x, len(x), True).mean() - rng.choice(y, len(y), True).mean() for _ in range(n_boot)]
    return np.percentile(diffs, [100 * (alpha / 2), 100 * (1 - alpha / 2)])

def main():
    print("--- SCRIPT 06: BIOPSY-LEVEL PREDICTED EXTRACELLULAR L-R CO-EXPRESSION ---")
    pairs = [('MIF', 'CD74'), ('LGALS9', 'HAVCR2'), ('SPP1', 'CD44'),
             ('TGFB1', 'TGFBR2'), ('HLA-E', 'KLRC1'), ('THBS1', 'CD47'), ('CD274', 'PDCD1')]

    adata_d = sc.read_h5ad(os.path.join(DATA_DIR, 'GSE131907_FINAL_ANNOTATED.h5ad'))
    raw_d = adata_d.raw.to_adata()
    raw_d.obs = adata_d.obs.copy()
    sub_d = raw_d[raw_d.obs['Sample_Origin'].isin(['tLung', 'tL/B', 'PE', 'mLN'])].copy()
    dg = list({g for p in pairs for g in p if g in sub_d.var_names})
    df_d = pd.DataFrame(sub_d[:, dg].X.toarray() if hasattr(sub_d[:, dg].X, 'toarray') else sub_d[:, dg].X, columns=dg)
    df_d['Sample'] = sub_d.obs['Sample'].astype(str).values
    df_d['Is_Met'] = sub_d.obs['Sample_Origin'].isin(['PE', 'mLN']).values
    df_d['CT'] = sub_d.obs['cell_type'].astype(str).values

    adata_m = sc.read_h5ad(os.path.join(DATA_DIR, 'maynard_annotated.h5ad'))
    obs_m = adata_m.obs.copy()
    if adata_m.raw is not None:
        adata_m = adata_m.raw.to_adata()
        adata_m.obs = obs_m
    sc.pp.normalize_total(adata_m, target_sum=1e4)
    sc.pp.log1p(adata_m)
    sub_m = adata_m[adata_m.obs['primary_or_metastaic'].isin(['Primary', 'Metastatic'])].copy()
    mg = list({g for p in pairs for g in p if g in sub_m.var_names})
    df_m = pd.DataFrame(sub_m[:, mg].X.toarray() if hasattr(sub_m[:, mg].X, 'toarray') else sub_m[:, mg].X, columns=mg)
    df_m['Sample'] = sub_m.obs['sample_name'].astype(str).values
    df_m['Is_Met'] = (sub_m.obs['primary_or_metastaic'] == 'Metastatic').values
    df_m['CT'] = sub_m.obs['general_annotation'].astype(str).values

    out_dfs = []
    for coh, dfc, t_pat, i_pat in [('Discovery (GSE131907)', df_d, 'Epithelial|Tumor', 'Macrophage|T-Cell|NK'),
                                   ('External (PRJNA591860)', df_m, 'epi', 'immune')]:
        valid = [(l, r) for l, r in pairs if l in dfc.columns and r in dfc.columns and dfc[l].sum() > 0 and dfc[r].sum() > 0]
        rows = []
        for l, r in valid:
            tb = dfc[dfc['CT'].str.contains(t_pat, case=False, na=False)].groupby(['Sample', 'Is_Met'])[l].mean()
            ib = dfc[dfc['CT'].str.contains(i_pat, case=False, na=False)].groupby(['Sample', 'Is_Met'])[r].mean()
            cb = pd.concat([tb.rename('L'), ib.rename('R')], axis=1).dropna()
            cb['Prod'] = cb['L'] * cb['R']
            cb['Z'] = (cb['Prod'] - cb['Prod'].mean()) / (cb['Prod'].std() or 1.0)
            mv = cb[cb.index.get_level_values('Is_Met') == True]['Z'].values
            pv = cb[cb.index.get_level_values('Is_Met') == False]['Z'].values
            _, pval = mannwhitneyu(mv, pv)
            lo, hi = bootstrap_ci(mv, pv, alpha=0.05 / len(valid))
            rows.append({
                'Cohort': coh, 'Pair': f"{l}–{r}",
                'N_Primary_Biopsies': len(pv), 'N_Metastatic_Biopsies': len(mv),
                'Effect_Size_SMD': mv.mean() - pv.mean(),
                'Simultaneous_95CI_Low': lo, 'Simultaneous_95CI_High': hi, 'Nominal_P': pval
            })
        tdf = pd.DataFrame(rows)
        tdf['BH_FDR_Q'] = multipletests(tdf['Nominal_P'], method='fdr_bh')[1]
        out_dfs.append(tdf)

    lr_all = pd.concat(out_dfs, ignore_index=True)
    lr_all.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_4B_LR_Biopsy_Stats.csv'), index=False)

    fig4, ax4 = plt.subplots(figsize=(9.8, 5.6))
    order = ['MIF–CD74', 'LGALS9–HAVCR2', 'SPP1–CD44', 'TGFB1–TGFBR2', 'HLA-E–KLRC1', 'THBS1–CD47', 'CD274–PDCD1']
    ymap = {p: i for i, p in enumerate(reversed(order))}
    for coh, off, clr in [('Discovery (GSE131907)', 0.16, '#3B75AF'), ('External (PRJNA591860)', -0.16, '#D63B44')]:
        sc_df = lr_all[lr_all['Cohort'] == coh]
        ys = [ymap[p] + off for p in sc_df['Pair']]
        xs = sc_df['Effect_Size_SMD'].values
        ax4.errorbar(xs, ys, xerr=[np.maximum(0, xs - sc_df['Simultaneous_95CI_Low'].values),
                                   np.maximum(0, sc_df['Simultaneous_95CI_High'].values - xs)],
                     fmt='o', color=clr, ecolor=clr, elinewidth=1.8, capsize=4, markersize=7, label=coh)
        for x_v, y_v, q_v in zip(xs, ys, sc_df['BH_FDR_Q']):
            if q_v < 0.05:
                ax4.text(x_v, y_v + 0.11, f"FDR q = {q_v:.3f}*", fontsize=9.5, fontweight='bold', color=clr, ha='center')

    ax4.text(0.05, ymap['MIF–CD74'] - 0.18, "(MIF unquantified in PRJNA591860 Smart-seq2 matrix)",
             fontsize=9, style='italic', color='#7A2025', va='center', ha='left')
    ax4.axvline(0, color='black', ls='--', lw=1.1)
    ax4.set_yticks(range(len(order)))
    ax4.set_yticklabels(list(reversed(order)), fontsize=11, fontweight='medium')
    ax4.set_xlabel('Standardized Mean Difference (Metastatic - Primary Biopsies)\nwith Multiplicity-Adjusted Simultaneous 95% CIs', fontsize=11.5, fontweight='bold')
    ax4.set_title('Biopsy-Level Predicted Extracellular Ligand–Receptor Co-Expression', fontsize=13, fontweight='bold', pad=12)
    ax4.legend(loc='lower right', frameon=True)
    sns.despine(ax=ax4)
    fig4.tight_layout()
    fig4.savefig(os.path.join(FIG_DIR, 'Figure_4_BiopsyLevel_LR_Coexpression.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig4)
    print("--- SCRIPT 06 COMPLETE ---")

if __name__ == "__main__":
    main()
