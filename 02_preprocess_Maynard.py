import os
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    print("--- SCRIPT 02: PRJNA591860 COHORT AUDIT & SAMPLE MANIFEST EXPORT ---")
    val_path = os.path.join(DATA_DIR, 'maynard_annotated.h5ad')
    adata = sc.read_h5ad(val_path)
    obs_m = adata.obs.copy()

    obs_m['Is_Epi'] = obs_m['general_annotation'].astype(str).str.lower().str.contains('epi')
    obs_m['Is_Imm'] = obs_m['general_annotation'].astype(str).str.lower().str.contains('immune')

    manifest = obs_m.groupby(
        ['patient_id', 'sample_name', 'driver_gene', 'driver_mutation', 'biopsy_site',
         'primary_or_metastaic', 'analysis', 'treatment_type', 'treatment', 'line_of_therapy'],
        dropna=False, observed=True
    ).agg(
        Total_Retained_Cells=('Is_Epi', 'count'),
        Epithelial_Cells=('Is_Epi', 'sum'),
        Immune_Cells=('Is_Imm', 'sum')
    ).reset_index()

    manifest = manifest[manifest['Total_Retained_Cells'] > 0].sort_values(['patient_id', 'sample_name'])
    out_csv = os.path.join(DATA_DIR, 'Supplementary_Data_10.csv')
    manifest.to_csv(out_csv, index=False)
    print(f"Exported {len(manifest)} biopsies across {manifest['patient_id'].nunique()} patient records to {out_csv}")
    print("--- SCRIPT 02 COMPLETE ---")

if __name__ == "__main__":
    main()
