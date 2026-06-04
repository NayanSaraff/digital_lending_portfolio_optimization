import json
import sys
from pathlib import Path
import pandas as pd

EXPECTED_SHEETS = ['customer_profile','loan_details','loan_outcome','repayment_behavior','behavioral_signals']

def normalize_sheet_name(name):
    return name.strip().lower().replace(' ', '_')


def read_excel(path):
    xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
    norm = {}
    for k, df in xls.items():
        norm_name = normalize_sheet_name(k)
        norm[norm_name] = df
    return norm


def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing


def to_flag(s):
    if pd.isna(s):
        return None
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().lower()
    if s in ('1','yes','y','true','t'):
        return 1
    if s in ('0','no','n','false','f'):
        return 0
    try:
        return int(float(s))
    except Exception:
        return None


def credit_band(score):
    try:
        s = float(score)
    except Exception:
        return 'unknown'
    if s < 300:
        return '<300'
    if s <= 550:
        return '300-550'
    if s <= 650:
        return '550-650'
    if s <= 750:
        return '650-750'
    if s <= 900:
        return '750-900'
    return '>900'


def analyze(path):
    path = Path(path)
    data = read_excel(path)
    out = {'file': str(path), 'notes': [], 'results': {}}

    for s in EXPECTED_SHEETS:
        if s not in data:
            out['notes'].append(f"Missing sheet: {s}")

    cp = data.get('customer_profile', pd.DataFrame())
    ld = data.get('loan_details', pd.DataFrame())
    lo = data.get('loan_outcome', pd.DataFrame())

    # Basic counts
    customers = int(cp['customer_id'].nunique()) if 'customer_id' in cp.columns else int(cp.shape[0])
    loans = int(ld['loan_id'].nunique()) if 'loan_id' in ld.columns else int(ld.shape[0])
    out['results']['total_customers'] = customers
    out['results']['total_loans'] = loans

    # Normalize flags
    if 'default_flag' in lo.columns:
        lo['default_flag_norm'] = lo['default_flag'].apply(to_flag)
    elif 'default' in lo.columns:
        lo['default_flag_norm'] = lo['default'].apply(to_flag)
    else:
        lo['default_flag_norm'] = None

    # Merge loan outcome with loan details and customer profile where possible
    merged = lo.copy()
    if not ld.empty:
        common = [c for c in ['loan_id','customer_id','product_type','acquisition_channel','city_tier'] if c in ld.columns]
        if 'loan_id' in merged.columns:
            merged = merged.merge(ld, on='loan_id', how='left', suffixes=('', '_ld'))
        else:
            # attempt merge on customer_id
            if 'customer_id' in merged.columns and 'customer_id' in ld.columns:
                merged = merged.merge(ld, on='customer_id', how='left', suffixes=('', '_ld'))

    if not cp.empty and 'customer_id' in merged.columns and 'customer_id' in cp.columns:
        merged = merged.merge(cp, on='customer_id', how='left', suffixes=('', '_cp'))

    # Compute overall default rate
    if 'default_flag_norm' in merged.columns and merged['default_flag_norm'].notna().any():
        total = merged['default_flag_norm'].notna().sum()
        defaults = merged['default_flag_norm'].fillna(0).sum()
        overall_rate = defaults / (total if total>0 else 1)
        out['results']['overall_default_rate'] = round(float(overall_rate), 4)
        out['results']['total_defaulted_loans'] = int(defaults)
    else:
        out['notes'].append('No default_flag column found or it is empty in loan_outcome sheet')

    # By employment type
    if 'employment_type' in merged.columns or 'employent_type' in merged.columns:
        emp_col = 'employment_type' if 'employment_type' in merged.columns else 'employent_type'
        grp = merged.dropna(subset=['default_flag_norm', emp_col]).groupby(emp_col)['default_flag_norm']
        out['results']['by_employment'] = {k: float(v.sum()/v.count()) for k,v in grp}
    else:
        out['notes'].append('No employment_type column available in merged data')

    # By product type
    if 'product_type' in merged.columns:
        grp = merged.dropna(subset=['default_flag_norm','product_type']).groupby('product_type')['default_flag_norm']
        out['results']['by_product'] = {k: float(v.sum()/v.count()) for k,v in grp}
    else:
        out['notes'].append('No product_type column')

    # By city tier
    if 'city_tier' in merged.columns:
        grp = merged.dropna(subset=['default_flag_norm','city_tier']).groupby('city_tier')['default_flag_norm']
        out['results']['by_city_tier'] = {k: float(v.sum()/v.count()) for k,v in grp}
    else:
        out['notes'].append('No city_tier column')

    # By acquisition channel
    if 'acquisition_channel' in merged.columns:
        grp = merged.dropna(subset=['default_flag_norm','acquisition_channel']).groupby('acquisition_channel')['default_flag_norm']
        out['results']['by_channel'] = {k: float(v.sum()/v.count()) for k,v in grp}
    else:
        out['notes'].append('No acquisition_channel column')

    # By credit score bands
    if 'credit_score' in merged.columns:
        merged['credit_band'] = merged['credit_score'].apply(credit_band)
        grp = merged.dropna(subset=['default_flag_norm','credit_band']).groupby('credit_band')['default_flag_norm']
        out['results']['by_credit_band'] = {k: float(v.sum()/v.count()) for k,v in grp}
    else:
        out['notes'].append('No credit_score column')

    # Top risk segments summary
    try:
        # find combinations with highest default rates (employment x product)
        if 'employment_type' in merged.columns and 'product_type' in merged.columns:
            combo = merged.dropna(subset=['default_flag_norm','employment_type','product_type']).groupby(['employment_type','product_type'])['default_flag_norm']
            combos = {f"{a}||{b}": float(v.sum()/v.count()) for (a,b),v in combo}
            # sort and keep top 5
            top5 = dict(sorted(combos.items(), key=lambda x: x[1], reverse=True)[:5])
            out['results']['top_risky_combinations'] = top5
    except Exception as e:
        out['notes'].append(f'Error computing top combinations: {e}')

    # write out
    out_path = path.with_suffix('.analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'Analysis written to: {out_path}')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python analyze_xlsx.py <path-to-xlsx>')
        sys.exit(1)
    analyze(sys.argv[1])
