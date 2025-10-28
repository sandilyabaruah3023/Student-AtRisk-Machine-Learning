# src/preprocess.py
import pandas as pd
from pathlib import Path

def create_target(student_info: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target column 'at_risk' in student_info:
    at_risk = 1 if final_result in ['Withdrawn','Fail'] else 0
    If no 'final_result' column is present, tries to fall back to 'code_module' check (minimal).
    """
    df = student_info.copy()
    if 'final_result' in df.columns:
        df['at_risk'] = df['final_result'].isin(['Withdrawn', 'Fail']).astype(int)
    else:
        raise KeyError("'final_result' not found in studentInfo.csv. Inspect your file.")
    return df

def merge_core(dfs: dict) -> pd.DataFrame:
    """
    Do a conservative merge to create a single table per student-module pair:
    - Start with studentRegistration (student-module enrollments)
    - Join studentInfo (student demographics) on 'id_student'
    - Aggregate VLE interactions for that student-module
    - Return a merged dataframe
    """
    reg = dfs['studentRegistration'].copy()
    info = dfs['studentInfo'].copy()
    vle = dfs['studentVle'].copy()
    vle_meta = dfs['vle'].copy()
    # create target
    info = create_target(info)

    # Merge registration with student info on id_student
    merged = reg.merge(info, on='id_student', how='left', suffixes=('', '_info'))

    # Basic VLE aggregation per (id_student, code_module)
    vle_agg = (
        vle
        .groupby(['id_student', 'code_module'])
        .agg(
            vle_events=('id_site', 'count'),
            vle_sum_clicks=('sum_click', 'sum') if 'sum_click' in vle.columns else ('id_site', 'count')
        )
        .reset_index()
    )
    merged = merged.merge(vle_agg, on=['id_student', 'code_module'], how='left')

    # Fill NaNs with zeros for aggregations
    merged['vle_events'] = merged['vle_events'].fillna(0)
    if 'vle_sum_clicks' in merged.columns:
        merged['vle_sum_clicks'] = merged['vle_sum_clicks'].fillna(0)

    # Some courses-level features (num_assessments etc.) could be added later
    return merged

if __name__ == "__main__":
    import sys
    from load_data import load_oulad
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../OULAD"
    dfs = load_oulad(data_dir)
    merged = merge_core(dfs)
    print("Merged shape:", merged.shape)
    print(merged.columns.tolist())
