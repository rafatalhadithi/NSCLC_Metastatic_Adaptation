import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11.5

def main():
    print("--- SCRIPT 09: AUDITABLE TCGA-LUAD COHORT, COX MODELS & FIGURE 7A ---")
    tcga_sub = os.path.join(DATA_DIR, 'tcga_luad', 'luad_tcga_pan_can_atlas_2018')
    tcga_dir = tcga_sub if os.path.exists(os.path.join(tcga_sub, 'data_mrna_seq_v2_rsem.txt')) else DATA_DIR

    top_genes = pd.read_csv(os.path.join(DATA_DIR, 'Supplementary_Data_5.csv')).iloc[:, 0].head(10).tolist()
    df_clin_all = pd.read_csv(os.path.join(tcga_dir, 'data_clinical_patient.txt'), sep='\t', skiprows=4)
    df_egfr = pd.read_csv(os.path.join(DATA_DIR, 'egfr_status.txt'), sep='\t')
    egfr_col = [c for c in df_egfr.columns if 'EGFR' in c.upper()][0]
    egfr_ids = df_egfr[pd.to_numeric(df_egfr[egfr_col], errors='coerce') == 1].iloc[:, 0].astype(str).str.split(':').str[-1].str[:12].unique()

    df_clin = df_clin_all[df_clin_all['PATIENT_ID'].isin(egfr_ids)].copy()
    df_clin = df_clin[df_clin['HISTORY_NEOADJUVANT_TRTYN'].astype(str).str.strip().str.upper() != 'YES'].copy()

    df_clin['OS_Event'] = df_clin['OS_STATUS'].apply(lambda x: 1 if '1' in str(x) else 0)
    df_clin['OS_Time'] = pd.to_numeric(df_clin['OS_MONTHS'], errors='coerce')
    df_clin['PFS_Event'] = df_clin['PFS_STATUS'].apply(lambda x: 1 if '1' in str(x) else 0)
    df_clin['PFS_Time'] = pd.to_numeric(df_clin['PFS_MONTHS'], errors='coerce')
    df_clin['Sex_Male'] = np.where(df_clin['SEX'].astype(str).str.strip().str.upper() == 'MALE', 1,
                          np.where(df_clin['SEX'].astype(str).str.strip().str.upper() == 'FEMALE', 0, np.nan))
    df_clin['Late_Stage'] = df_clin['AJCC_PATHOLOGIC_TUMOR_STAGE'].apply(
        lambda s: np.nan if str(s).upper().strip() in ['NAN', '', '[NOT AVAILABLE]', '[DISCREPANCY]']
        else (1 if ('III' in str(s).upper() or 'IV' in str(s).upper()) else 0))
    df_clin['Age'] = pd.to_numeric(df_clin['AGE'], errors='coerce')

    df_expr = pd.read_csv(os.path.join(tcga_dir, 'data_mrna_seq_v2_rsem.txt'), sep='\t').set_index('Hugo_Symbol')
    matched = [g for g in top_genes if g in df_expr.index]
    df_t = df_expr.loc[matched].T.reset_index()
    df_t = df_t[df_t['index'].str[13:15] == '01'].copy()
    df_t['PATIENT_ID'] = df_t['index'].str[:12]
    df_t = df_t.drop_duplicates(subset=['PATIENT_ID'])

    df = pd.merge(df_clin, df_t, on='PATIENT_ID', how='inner')
    for g in matched:
        df[g] = np.log2(pd.to_numeric(df[g], errors='coerce') + 1.0)
    df = df.dropna(subset=matched).copy()

    raw_s = df[matched].apply(lambda x: (x - x.mean()) / x.std(), axis=0).mean(axis=1)
    df['Composite_Signature_Z'] = (raw_s - raw_s.mean()) / raw_s.std()
    med = df['Composite_Signature_Z'].median()
    df['Group'] = np.where(df['Composite_Signature_Z'] >= med, 'High', 'Low')
    df['High_Group_Binary'] = np.where(df['Group'] == 'High', 1, 0)

    stats_rows = []
    for ep, t_c, e_c, out_f in [('Overall Survival (OS)', 'OS_Time', 'OS_Event', 'Figure_7B_Survival_Data.csv'),
                                ('Progression-Free Survival (PFS)', 'PFS_Time', 'PFS_Event', 'Figure_9_Survival_Data.csv')]:
        su = df[['PATIENT_ID', t_c, e_c, 'Group', 'High_Group_Binary', 'Composite_Signature_Z']].dropna()
        su = su[su[t_c] > 0].copy()
        su[['PATIENT_ID', t_c, e_c, 'Group', 'Composite_Signature_Z']].rename(
            columns={t_c: 'Time', e_c: 'Event'}).to_csv(os.path.join(DATA_DIR, out_f), index=False)

        lr = logrank_test(su[su['High_Group_Binary']==1][t_c], su[su['High_Group_Binary']==0][t_c],
                          event_observed_A=su[su['High_Group_Binary']==1][e_c], event_observed_B=su[su['High_Group_Binary']==0][e_c])

        for m_name, v_col, lr_p in [('Univariable (Median Split: High vs Low [Ref])', 'High_Group_Binary', lr.p_value),
                                    ('Univariable (Continuous per 1-SD)', 'Composite_Signature_Z', np.nan)]:
            c_fit = CoxPHFitter().fit(su[[t_c, e_c, v_col]], duration_col=t_c, event_col=e_c)
            ph = proportional_hazard_test(c_fit, su[[t_c, e_c, v_col]], time_transform='rank')
            stats_rows.append({
                'Endpoint': ep, 'Model': m_name, 'Variable': v_col,
                'N_Patients': len(su), 'N_Events': int(su[e_c].sum()),
                'HR': c_fit.hazard_ratios_[v_col],
                'CI_95_Lower': np.exp(c_fit.confidence_intervals_.loc[v_col].iloc[0]),
                'CI_95_Upper': np.exp(c_fit.confidence_intervals_.loc[v_col].iloc[1]),
                'Cox_P_Value': c_fit.summary.loc[v_col, 'p'],
                'LogRank_P_Value': lr_p,
                'PH_Assumption_P_Value': ph.summary.loc[v_col, 'p']
            })

        sm = df[[t_c, e_c, 'High_Group_Binary', 'Composite_Signature_Z', 'Age', 'Sex_Male', 'Late_Stage']].dropna()
        sm = sm[sm[t_c] > 0].copy()
        cm_cont = CoxPHFitter().fit(sm[[t_c, e_c, 'Composite_Signature_Z', 'Age', 'Sex_Male', 'Late_Stage']], duration_col=t_c, event_col=e_c)
        ph_cont = proportional_hazard_test(cm_cont, sm[[t_c, e_c, 'Composite_Signature_Z', 'Age', 'Sex_Male', 'Late_Stage']], time_transform='rank')
        for v in ['Composite_Signature_Z', 'Age', 'Sex_Male', 'Late_Stage']:
            stats_rows.append({
                'Endpoint': ep, 'Model': 'Multivariable Adjusted (Continuous Score)', 'Variable': v,
                'N_Patients': len(sm), 'N_Events': int(sm[e_c].sum()),
                'HR': cm_cont.hazard_ratios_[v],
                'CI_95_Lower': np.exp(cm_cont.confidence_intervals_.loc[v].iloc[0]),
                'CI_95_Upper': np.exp(cm_cont.confidence_intervals_.loc[v].iloc[1]),
                'Cox_P_Value': cm_cont.summary.loc[v, 'p'],
                'LogRank_P_Value': np.nan,
                'PH_Assumption_P_Value': ph_cont.summary.loc[v, 'p']
            })

        cm_med = CoxPHFitter().fit(sm[[t_c, e_c, 'High_Group_Binary', 'Age', 'Sex_Male', 'Late_Stage']], duration_col=t_c, event_col=e_c)
        ph_med = proportional_hazard_test(cm_med, sm[[t_c, e_c, 'High_Group_Binary', 'Age', 'Sex_Male', 'Late_Stage']], time_transform='rank')
        stats_rows.append({
            'Endpoint': ep, 'Model': 'Multivariable Adjusted (Median Split)', 'Variable': 'High_Group_Binary',
            'N_Patients': len(sm), 'N_Events': int(sm[e_c].sum()),
            'HR': cm_med.hazard_ratios_['High_Group_Binary'],
            'CI_95_Lower': np.exp(cm_med.confidence_intervals_.loc['High_Group_Binary'].iloc[0]),
            'CI_95_Upper': np.exp(cm_med.confidence_intervals_.loc['High_Group_Binary'].iloc[1]),
            'Cox_P_Value': cm_med.summary.loc['High_Group_Binary', 'p'],
            'LogRank_P_Value': np.nan,
            'PH_Assumption_P_Value': ph_med.summary.loc['High_Group_Binary', 'p']
        })

    pd.DataFrame(stats_rows).to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_11.csv'), index=False)
    df[['PATIENT_ID', 'index', 'AGE', 'SEX', 'AJCC_PATHOLOGIC_TUMOR_STAGE', 'HISTORY_NEOADJUVANT_TRTYN',
        'OS_Time', 'OS_Event', 'PFS_Time', 'PFS_Event', 'Composite_Signature_Z', 'Group'] + matched].rename(
        columns={'index': 'SAMPLE_ID'}).to_csv(os.path.join(DATA_DIR, 'Supplementary_Data_12.csv'), index=False)

    # Figure 7A: Pseudotime-Ordered Expression Cross-Correlation
    tlcc_path = os.path.join(DATA_DIR, 'Figure_6A_Causality_Data.csv')
    if os.path.exists(tlcc_path):
        tlcc_df = pd.read_csv(tlcc_path)
        lags, corrs = tlcc_df.iloc[:, 0], tlcc_df.iloc[:, 1]
        opt_lag = int(round(lags.iloc[corrs.idxmax()]))

        fig7a, ax7a = plt.subplots(figsize=(7.2, 5.0))
        ax7a.plot(lags, corrs, color='#3B0F70', lw=2.8, zorder=3)
        ax7a.axvline(0, color='#777777', ls='--', lw=1.5, label='Zero Lag (Lag = 0)', zorder=2)
        ax7a.axvline(opt_lag, color='#E63946', ls='-', lw=1.8, label=f'Optimal Lag ({opt_lag:+d})', zorder=4)
        ax7a.set_xlabel('Lag (Pseudotime Units)', fontsize=12, fontweight='bold')
        ax7a.set_ylabel('Cross-Correlation (TBX3 vs HOXC10)', fontsize=12, fontweight='bold')
        ax7a.legend(loc='upper left', bbox_to_anchor=(0.08, 0.96), frameon=False, fontsize=10.5)
        sns.despine(ax=ax7a)
        fig7a.tight_layout()
        fig7a.savefig(os.path.join(FIG_DIR, 'Figure_7A_Pseudotime_CrossCorrelation.pdf'), dpi=600, bbox_inches='tight')
        plt.close(fig7a)

    print("--- SCRIPT 09 COMPLETE ---")

if __name__ == "__main__":
    main()
