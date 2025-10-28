# src/features.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def simple_feature_engineer(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Create simple features and return X, y.
    - numeric columns: age_band -> ordinal mapping if possible
    - categorical: gender, region, disability (one-hot)
    - fill NaNs
    """
    df = df.copy()
    if 'at_risk' not in df.columns:
        raise KeyError("Target column 'at_risk' not found. Run preprocess.create_target.")
    y = df['at_risk'].astype(int)

    # Simple numeric features
    # If age_band exists, map it to ordinal
    if 'age_band' in df.columns:
        age_order = {
            '0-35': 0, '35-55': 1, '55<=': 2,
            '18-20': 0, '21-25': 0, '26-30': 0, # alternate labels sometimes present
        }
        df['age_ord'] = df['age_band'].map(lambda x: age_order.get(x, -1))

    numeric_cols = ['num_of_prev_attempts', 'vle_events']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    cat_cols = [c for c in ['gender', 'region', 'highest_education', 'imd_band'] if c in df.columns]

    # Impute numerics
    X_num = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    if not X_num.empty:
        imp_num = SimpleImputer(strategy='median')
        X_num = pd.DataFrame(imp_num.fit_transform(X_num), columns=X_num.columns, index=df.index)

    # One-hot encode categorical
    X_cat = pd.DataFrame(index=df.index)
    if cat_cols:
        enc = OneHotEncoder(handle_unknown='ignore')
        # Compatibility fix for sklearn versions
    try:
        Xc = enc.fit_transform(df[cat_cols].fillna('missing')).toarray()
    except AttributeError:
        Xc = enc.fit_transform(df[cat_cols].fillna('missing'))
    cols = enc.get_feature_names_out(cat_cols)
    X_cat = pd.DataFrame(Xc, columns=cols, index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    return X, y
