"""Documentation of Estimate Seasonality Period

# Estimate Seasonality Period

## Description

Estimate a plausible season length (seasonal_periods) for a given time series using the
autocorrelation function (ACF). This is helpful when you want to apply seasonal models but
do not know the correct period a priori.

## Inputs

- **series** (Pandas Series):
    Time series with a Datetime index.
\- **acf_threshold** (Float, default value: 0.3):
    Minimum ACF value required for a lag to be considered as season length.

## Outputs

- **seasonal_periods** (Integer):
    Estimated season length. Returns `None` if no suitable seasonality is detected.

## Details

The component performs light preprocessing:
- Sort by timestamp and aggregate duplicate timestamps by mean.
- If timestamp gaps are irregular, resample to the most common interval and interpolate.
- Replace ±Inf with NaN, interpolate missing values (time-based), and drop remaining NaNs.

ACF is computed up to a maximum lag determined by data length and (if available) inferred
frequency. The returned lag is the one with the highest ACF above a fixed threshold (0.3)
in the range [2, max_period]. If no lag exceeds the threshold, `None` is returned. As an
additional sanity check, the detected period is only returned if at least two full
seasons fit into the data (len(series) >= 2 * period).

## Example

```
{
    "series": {
        "2023-09-04T00:00:00.000Z": 201,
        "2023-09-05T00:00:00.000Z": 194,
        "2023-09-06T00:00:00.000Z": 281,
        "2023-09-07T00:00:00.000Z": 279,
        "2023-09-08T00:00:00.000Z": 375,
        "2023-09-09T00:00:00.000Z": 393,
        "2023-09-10T00:00:00.000Z": 390,
        "2023-09-11T00:00:00.000Z": 220,
        "2023-09-12T00:00:00.000Z": 222,
        "2023-09-13T00:00:00.000Z": 312,
        "2023-09-14T00:00:00.000Z": 277,
        "2023-09-15T00:00:00.000Z": 332,
        "2023-09-16T00:00:00.000Z": 401,
        "2023-09-17T00:00:00.000Z": 400,
        "2023-09-18T00:00:00.000Z": 291,
        "2023-09-19T00:00:00.000Z": 282,
        "2023-09-20T00:00:00.000Z": 316,
        "2023-09-21T00:00:00.000Z": 305,
        "2023-09-22T00:00:00.000Z": 333,
        "2023-09-23T00:00:00.000Z": 398,
        "2023-09-24T00:00:00.000Z": 414
    },
    "acf_threshold": 0.3
}
```
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as sm_acf

from hdutils import ComponentInputValidationException


def _resample_if_needed(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        raise ComponentInputValidationException(
            "The input data must not be empty!",
            error_code="EmptyDataFrame",
            invalid_component_inputs=["series"],
        )
    if pd.api.types.is_datetime64_any_dtype(series.index.dtype) is False:
        raise ComponentInputValidationException(
            "Indices of series must be datetime, but are of type "
            + str(series.index.dtype),
            error_code=422,
            invalid_component_inputs=["series"],
        )

    # sort and aggregate duplicates
    series = series.sort_index()
    if not series.index.is_unique:
        series = series.groupby(level=0).mean()

    if len(series) < 2:
        return series

    time_diffs = series.index.to_series().diff().dropna()
    if time_diffs.empty:
        return series
    normal_diff = time_diffs.value_counts().idxmax()

    if normal_diff > pd.Timedelta(0) and not all(time_diffs == normal_diff):
        new_index = pd.date_range(
            start=series.index.min(), end=series.index.max(), freq=normal_diff
        )
        series = series.reindex(new_index).interpolate()

    return series


def _clean_series(series: pd.Series, min_required_points: int = 6) -> pd.Series:
    series = series.replace([np.inf, -np.inf], np.nan)
    method = (
        "time" if pd.api.types.is_datetime64_any_dtype(series.index.dtype) else "linear"
    )
    series = series.interpolate(method=method).dropna()
    if len(series) < min_required_points:
        # Too short for a meaningful ACF based detection
        return pd.Series(dtype=float)
    return series


def _max_period_from_freq(series: pd.Series) -> int:
    n = len(series)
    max_by_len = max(2, n // 2)
    max_by_freq = max_by_len
    if isinstance(series.index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(series.index)
        except Exception:
            freq = None
        if freq:
            f = freq.upper()
            if f.startswith("H"):
                max_by_freq = min(max_by_len, 24 * 7)
            elif f.startswith("D"):
                max_by_freq = min(max_by_len, 365)
            elif f.startswith("W"):
                max_by_freq = min(max_by_len, 52)
            elif f.startswith("M"):
                max_by_freq = min(max_by_len, 12)
    return max(2, max_by_freq)


def estimate_seasonal_periods(series: pd.Series, threshold: float = 0.3) -> int | None:
    series = _resample_if_needed(series)
    series = _clean_series(series)
    if len(series) == 0:
        return None

    max_period = _max_period_from_freq(series)
    if max_period <= 2:
        return None

    try:
        acf_vals = sm_acf(series.values, nlags=max_period, fft=True)
    except Exception:
        return None

    best_lag = None
    best_val = -1.0
    for lag in range(2, len(acf_vals)):
        val = acf_vals[lag]
        if val >= threshold and val > best_val:
            best_val = val
            best_lag = lag

    if best_lag is None:
        return None
    if len(series) < 2 * best_lag:
        return None
    return int(best_lag)


# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "series": {"data_type": "SERIES"},
        "acf_threshold": {"data_type": "FLOAT", "default_value": 0.3},
    },
    "outputs": {
        "seasonal_periods": {"data_type": "INT"},
    },
    "name": "Estimate Seasonality Period",
    "category": "Time Series Analysis",
    "description": "Estimate season length (seasonal_periods) via ACF",
    "version_tag": "1.0.0",
    "id": "b6c3f1d1-4e0b-4a2a-9b6c-7f0a1e3c9f21",
    "revision_group_id": "c5a0e7a2-6f8c-4b8d-9cc0-3c1c2b2e1a11",
    "state": "DRAFT",
}

from hdutils import parse_default_value  # noqa: E402, F401


def main(*, series, acf_threshold=0.3):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    # write your function code here.
    sp = estimate_seasonal_periods(series=series, threshold=acf_threshold)
    return {"seasonal_periods": sp}


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "series",
            "filters": {
                "value": '{\n    "2023-09-04T00:00:00.000Z": 201,\n    "2023-09-05T00:00:00.000Z": 194,\n    "2023-09-06T00:00:00.000Z": 281,\n    "2023-09-07T00:00:00.000Z": 279,\n    "2023-09-08T00:00:00.000Z": 375,\n    "2023-09-09T00:00:00.000Z": 393,\n    "2023-09-10T00:00:00.000Z": 390,\n    "2023-09-11T00:00:00.000Z": 220,\n    "2023-09-12T00:00:00.000Z": 222,\n    "2023-09-13T00:00:00.000Z": 312,\n    "2023-09-14T00:00:00.000Z": 277,\n    "2023-09-15T00:00:00.000Z": 332,\n    "2023-09-16T00:00:00.000Z": 401,\n    "2023-09-17T00:00:00.000Z": 400,\n    "2023-09-18T00:00:00.000Z": 291,\n    "2023-09-19T00:00:00.000Z": 282,\n    "2023-09-20T00:00:00.000Z": 316,\n    "2023-09-21T00:00:00.000Z": 305,\n    "2023-09-22T00:00:00.000Z": 333,\n    "2023-09-23T00:00:00.000Z": 398,\n    "2023-09-24T00:00:00.000Z": 414\n}\n'
            },
        },
        {"workflow_input_name": "acf_threshold", "filters": {"value": "0.3"}},
    ]
}
RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "series",
            "filters": {
                "value": '{\n    "2023-09-04T00:00:00.000Z": 201,\n    "2023-09-05T00:00:00.000Z": 194,\n    "2023-09-06T00:00:00.000Z": 281,\n    "2023-09-07T00:00:00.000Z": 279,\n    "2023-09-08T00:00:00.000Z": 375,\n    "2023-09-09T00:00:00.000Z": 393,\n    "2023-09-10T00:00:00.000Z": 390,\n    "2023-09-11T00:00:00.000Z": 220,\n    "2023-09-12T00:00:00.000Z": 222,\n    "2023-09-13T00:00:00.000Z": 312,\n    "2023-09-14T00:00:00.000Z": 277,\n    "2023-09-15T00:00:00.000Z": 332,\n    "2023-09-16T00:00:00.000Z": 401,\n    "2023-09-17T00:00:00.000Z": 400,\n    "2023-09-18T00:00:00.000Z": 291,\n    "2023-09-19T00:00:00.000Z": 282,\n    "2023-09-20T00:00:00.000Z": 316,\n    "2023-09-21T00:00:00.000Z": 305,\n    "2023-09-22T00:00:00.000Z": 333,\n    "2023-09-23T00:00:00.000Z": 398,\n    "2023-09-24T00:00:00.000Z": 414\n}\n'
            },
        },
        {"workflow_input_name": "acf_threshold", "filters": {"value": "0.3"}},
    ]
}
