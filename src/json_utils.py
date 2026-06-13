"""
JSON serialization helpers shared by every scripts/*_run.py.

Why this exists
---------------
`json.dump(..., default=str)` silently stringifies anything it can't natively
encode. That includes:

  * numpy.bool_  -> "True"   (looks like a real bool, isn't)
  * numpy.float64 NaN/Inf  -> "nan"/"inf"  (not even valid JSON for tools)
  * pandas.Series -> str(series)  (an enormous junk string)

`safe_json_default` collapses those to honest Python values so the on-disk
artifacts the project advertises as authoritative are actually trustworthy.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def safe_json_default(o: Any) -> Any:
    """Use as ``json.dump(..., default=safe_json_default)``.

    Order matters: ``numpy.bool_`` subclasses ``int`` in older numpys, so
    bool must be checked before integer.
    """
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(o, (pd.Series, pd.DataFrame)):
        # Don't silently dump a huge tabular blob into a metrics summary.
        return None
    return str(o)
