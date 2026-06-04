import sys
import json
from pathlib import Path

import pandas as pd

def summarize(path):
    path = Path(path)
    out = {"file": str(path), "sheets": {}}
    xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for name, df in xls.items():
        info = {}
        info["rows"] = int(df.shape[0])
        info["columns"] = int(df.shape[1])
        info["columns_info"] = []
        for col in df.columns.tolist():
            col_ser = df[col]
            col_info = {
                "name": str(col),
                "cname":str(row),
                "dtype": str(col_ser.dtype),
                "non_null_count": int(col_ser.count()),
                "null_count": int(col_ser.isnull().sum())
            }
            try:
                unique = col_ser.dropna().unique()
                if len(unique) <= 20:
                    col_info["unique_values"] = [str(x) for x in unique.tolist()]
                else:
                    col_info["unique_count"] = int(col_ser.nunique(dropna=True))
            except Exception:
                col_info["unique_count"] = int(col_ser.nunique(dropna=True))
            info["columns_info"].append(col_info)
        # sample rows
        info["sample_rows"] = df.head(5).fillna("").to_dict(orient="records")
        out["sheets"][name] = info
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_xlsx.py <path-to-xlsx>")
        sys.exit(1)
    path = sys.argv[1]
    summary = summarize(path)
    json_out = Path(path).with_suffix(".summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary written to: {json_out}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
