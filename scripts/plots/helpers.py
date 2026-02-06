import pandas as pd
import numpy as np
BOOL_TRUE = {"true", "1", "yes", "y", "ok", "success", "t"}
BOOL_FALSE = {"false", "0", "no", "n", "fail", "error", "f", "", "none", "nan"}


def as_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="boolean")
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Float64").fillna(0).astype(int).astype(bool).astype("boolean")
    s2 = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[s2.isin(BOOL_TRUE)] = True
    out[s2.isin(BOOL_FALSE)] = False
    return out

def _dynamic_ylim_percent(ax, y_values: np.ndarray) -> None:
    vals = np.asarray(y_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(vals.max()) if vals.size else 0.0
    pad = max(2.0, 0.10 * vmax)
    ax.set_ylim(0.0, min(100.0, vmax + pad))