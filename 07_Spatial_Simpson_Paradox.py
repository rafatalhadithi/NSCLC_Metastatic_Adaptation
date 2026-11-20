import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5

def main():
    print("--- SCRIPT 07: VISIUM PHYSICAL SPATIAL CORRELATION & GRID BLOCK AGGREGATION ---")
    sp_candidates = [
        os.path.join(DATA_DIR, 'validation_cohorts', 'GSE267960', 'GSM8282531_S20-12254-L9', 'S20-12254-L9'),
        os.path.join(DATA_DIR, 'GSM8282531')
    ]
    sp_dir = next((p for p in sp_candidates if os.path.exists(os.path.join(p, 'filtered_feature_bc_matrix.h5'))), sp_candidates[0])

    adata_sp = sc.read_10x_h5(os.path.join(sp_dir, 'filtered_feature_bc_matrix.h5'))
    adata_sp.var_names_make_unique()
    sc.pp.normalize_total(adata_sp, target_sum=1e4)
    sc.pp.log1p(adata_sp)

    tp = pd.read_csv(os.path.join(sp_dir, 'tissue_positions_list.csv'), header=None)
    if str(tp.iloc[0, 0]).lower() == 'barcode':
        tp = tp.iloc[1:].copy()
    tp.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
    tp['pxl_row'] = pd.to_numeric(tp['pxl_row'], errors='coerce')
    tp['pxl_col'] = pd.to_numeric(tp['pxl_col'], errors='coerce')
    tp = tp.set_index('barcode')

    shared = adata_sp.obs_names.intersection(tp.index)
    adata_sp = adata_sp[shared].copy()
    ezh2 = np.asarray(adata_sp[:, 'EZH2'].X.todense()).flatten()
    hoxc = np.asarray(adata_sp[:, ['HOXC9', 'HOXC10', 'HOXC11']].X.todense()).sum(axis=1).flatten()
    expr_mask = (ezh2 > 0) | (hoxc > 0)

    # Supplementary Data 6: Spot-level epigenetic regulator screen on expressed spots
    epi_genes = [g for g in ['EZH2', 'SUZ12', 'EED', 'DNMT1', 'KDM6A', 'KDM6B', 'EP300', 'SMARCA4', 'BRD4'] if g in adata_sp.var_names]
    epi_rows = []
    for eg in epi_genes:
        vec = np.asarray(adata_sp[:, eg].X.todense()).flatten()[expr_mask]
        rv, pv = spearmanr(vec, hoxc[expr_mask])
        if np.isfinite(rv):
            epi_rows.append({'Epigenetic_Regulator': eg, 'Spearman_R': rv, 'P_Value': pv})
    epi_df = pd.DataFrame(epi_rows).sort_values('Spearman_R', ascending=True)
    epi_df.to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_6.csv'), index=False)

    # Supplementary Data 7: 2D Physical Spatial Block Bootstrap & 12x12 Grid Block Aggregation
    phys_df = pd.DataFrame({'X_px': tp.loc[shared, 'pxl_col'].values, 'Y_px': tp.loc[shared, 'pxl_row'].values,
                            'EZH2_Log1p': ezh2, 'HOXC_Log1p': hoxc}).dropna()
    expr_sp = phys_df[(phys_df['EZH2_Log1p'] > 0) | (phys_df['HOXC_Log1p'] > 0)].copy()
    r_spot, p_spot = spearmanr(expr_sp['EZH2_Log1p'], expr_sp['HOXC_Log1p'])

    expr_sp['block_id'] = pd.cut(expr_sp['X_px'], 5, labels=False).astype(str) + "_" + pd.cut(expr_sp['Y_px'], 5, labels=False).astype(str)
    blocks = [g for _, g in expr_sp.groupby('block_id')]
    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(1000):
        s_df = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))], ignore_index=True)
        rb, _ = spearmanr(s_df['EZH2_Log1p'], s_df['HOXC_Log1p'])
        if np.isfinite(rb):
            boot_rs.append(rb)
    sb_lo, sb_hi = np.percentile(boot_rs, [2.5, 97.5])

    phys_df['gx'] = pd.cut(phys_df['X_px'], 12, labels=False)
    phys_df['gy'] = pd.cut(phys_df['Y_px'], 12, labels=False)
    grid = phys_df.groupby(['gx', 'gy'], observed=True)[['EZH2_Log1p', 'HOXC_Log1p']].sum().reset_index()
    grid = grid[(grid['EZH2_Log1p'] + grid['HOXC_Log1p']) > 0]
    r_grid, p_grid = spearmanr(grid['EZH2_Log1p'], grid['HOXC_Log1p'])

    pd.DataFrame([
        {'Spatial_Resolution': 'Single-Spot Physical Visium Resolution (Expressed Spots)', 'N_Spatial_Units': len(expr_sp),
         'Spearman_R': r_spot, 'P_Value': p_spot, 'Physical_Spatial_Block_Bootstrap_95CI': f"[{sb_lo:.3f}, {sb_hi:.3f}]"},
        {'Spatial_Resolution': 'Physical Spatial Grid Block Aggregation (12x12 Slide Macro-Blocks)', 'N_Spatial_Units': len(grid),
         'Spearman_R': r_grid, 'P_Value': p_grid, 'Physical_Spatial_Block_Bootstrap_95CI': 'Spatially Aggregated Slide Blocks'}
    ]).to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_7.csv'), index=False)

    # Supplementary Figure S3
    fig_s3, axes_s3 = plt.subplots(1, 3, figsize=(17.2, 5.3))
    for ax_i, col_i, cm_i, tit_i, lbl_i in [
        (axes_s3[0], 'EZH2_Log1p', 'Reds', 'A. Tissue Spatial EZH2 Expression\n(10x Visium Log1p Counts, GSM8282531)', 'EZH2 Log1p'),
        (axes_s3[1], 'HOXC_Log1p', 'Blues', 'B. Tissue Spatial HOXC Locus Expression\n(Summed HOXC9/10/11 Log1p Counts)', 'HOXC Log1p')
    ]:
        sc_p = ax_i.scatter(phys_df['X_px'], phys_df['Y_px'], c=phys_df[col_i], cmap=cm_i, s=10, edgecolors='none')
        ax_i.invert_yaxis()
        ax_i.set_aspect('equal')
        ax_i.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax_i.set_title(tit_i, fontweight='bold', fontsize=12)
        ax_i.set_xlabel('Tissue Slide X Coordinate (px)', fontsize=11.5)
        ax_i.set_ylabel('Tissue Slide Y Coordinate (px)', fontsize=11.5)
        plt.colorbar(sc_p, ax=ax_i, fraction=0.046, pad=0.04, label=lbl_i)

    bar_clrs = ['#1D3557' if g == 'EZH2' else '#457B9D' for g in epi_df['Epigenetic_Regulator']]
    axes_s3[2].barh(epi_df['Epigenetic_Regulator'], epi_df['Spearman_R'], color=bar_clrs, edgecolor='black')
    axes_s3[2].axvline(0, color='black', lw=1.1)
    axes_s3[2].set_title('C. Exploratory Spot-Level Correlation\nwith HOXC Locus Expression (n = 2,305)', fontweight='bold', fontsize=12)
    axes_s3[2].set_xlabel('Spearman Rank Correlation (R)', fontsize=11.5)
    sns.despine(ax=axes_s3[2])
    fig_s3.tight_layout()
    fig_s3.savefig(os.path.join(FIG_DIR, 'Supplementary_Figure_S3_Spatial_Screen.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig_s3)

    # Supplementary Figure S4
    fig_s4, ax_s4 = plt.subplots(figsize=(7.6, 5.0))
    lbls = [f'Single-Spot Visium Resolution\n(n = {len(expr_sp)} Expressed Spots)',
            f'Spatial Grid Block Aggregation\n(n = {len(grid)} Slide Macro-Blocks)']
    vals, pvals = [r_spot, r_grid], [p_spot, p_grid]
    bars = ax_s4.bar(lbls, vals, color=['#1D3557', '#E63946'], edgecolor='black', width=0.46)
    ax_s4.axhline(0, color='black', lw=1.2)
    for b_i, v_i, p_i in zip(bars, vals, pvals):
        yt = v_i + (0.04 if v_i >= 0 else -0.06)
        ptxt = "p < 0.0001" if p_i < 1e-4 else f"p = {p_i:.4f}"
        ax_s4.text(b_i.get_x() + b_i.get_width()/2, yt, f"R = {v_i:.3f}\n({ptxt})",
                   ha='center', va='bottom' if v_i >= 0 else 'top', fontweight='bold', fontsize=11)
    ax_s4.set_ylim(min(vals) - 0.22, max(vals) + 0.22)
    ax_s4.set_ylabel('Spearman Correlation (R): EZH2 vs. HOXC Locus', fontsize=11.5, fontweight='bold')
    ax_s4.set_title('Resolution-Dependent Spatial Aggregation Masking', fontsize=12.5, fontweight='bold', pad=12)
    sns.despine(ax=ax_s4)
    fig_s4.tight_layout()
    fig_s4.savefig(os.path.join(FIG_DIR, 'Supplementary_Figure_S4_Aggregation_Masking.pdf'), dpi=600, bbox_inches='tight')
    plt.close(fig_s4)
    print("--- SCRIPT 07 COMPLETE ---")

if __name__ == "__main__":
    main()
