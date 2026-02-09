"""Documentation for Jump Detection

# Jump Detection

## Description
Component to detect jumps in time series data.

## Inputs
- **timeseries** (Pandas Series):
    The input time series. Index must be datetime, values numeric.
- **method** (String, default value: "threshold_on_derivative"):
    Jump detection method. One of "threshold_on_derivative",
    "window_mean_shift", "robust_zscore_on_diff".
- **derivative** (Pandas Series, default value: null):
    Optional externally computed derivative signal.
- **derivative_method** (String, default value: "difference"):
    Method for internal derivative calculation. One of "difference",
    "quotient", "gradient".
- **normalize_by_time** (Boolean, default value: True):
    If True and datetime index is available, divides by delta time.
- **time_unit** (String, default value: "s"):
    Time unit for normalization. One of "s", "min", "h", "d".
- **fill_edge** (String, default value: "nan"):
    Edge handling for derivative output. One of "nan", "zero", "ffill".
- **smoothing_before** (String, default value: "none"):
    Optional pre-smoothing. One of "none", "moving_average".
- **smoothing_window** (String, default value: null):
    Window for optional moving average smoothing.
- **threshold** (String, default value: "auto"):
    Detection threshold. Either "auto" or numeric value.
- **window** (String, default value: null):
    Window for method "window_mean_shift".
- **min_distance** (String, default value: "1"):
    Minimum distance between two events (samples or time string).
- **direction** (String, default value: "both"):
    Event direction filter. One of "both", "up", "down".
- **merge_close_events** (Boolean, default value: True):
    Merge events that are closer than merge_distance.
- **merge_distance** (String, default value: null):
    Merge distance (samples or time string). If null, min_distance is used.
- **output_segments** (Boolean, default value: True):
    If True, additionally returns piecewise segment means.

## Outputs
- **jump_mask** (Pandas Series):
    Boolean mask with detected jump positions.
- **jump_events** (Pandas DataFrame):
    Event table with timestamp, magnitude, direction, confidence, method.

## Details
1. The input series is sorted by timestamp, duplicate timestamps are merged by mean.
2. Optional smoothing is applied to stabilize derivative-based detection.
3. A derivative signal is created (or an external derivative is aligned to input index).
4. Depending on `method`, a detection score is calculated for each timestamp.
5. A threshold is resolved (`auto` via robust MAD estimate or fixed numeric value).
6. Candidate jumps are filtered by direction and minimum event distance.
7. Nearby events can be merged and are exported as event table plus mask.
8. Optional segment means are calculated between detected jump boundaries.

## Example
```json
{
  "timeseries": {
    "2025-01-12T00:00:00Z": 10,
    "2025-01-12T00:05:00Z": 10,
    "2025-01-12T00:10:00Z": 11,
    "2025-01-12T00:15:00Z": 12,
    "2025-01-12T00:20:00Z": 26,
    "2025-01-12T00:25:00Z": 27,
    "2025-01-12T00:30:00Z": 27,
    "2025-01-12T00:35:00Z": 13,
    "2025-01-12T00:40:00Z": 13,
    "2025-01-12T00:45:00Z": 12,
    "2025-01-12T00:50:00Z": 12,
    "2025-01-12T00:55:00Z": 20,
    "2025-01-12T01:00:00Z": 21
  },
  "method": "threshold_on_derivative",
  "derivative_method": "difference",
  "normalize_by_time": true,
  "threshold": "auto",
  "min_distance": "2",
  "direction": "both"
}
```
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hdutils import ComponentInputValidationException, parse_default_value


def parse_bool_like(value: Any, input_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ComponentInputValidationException(
        f"{input_name} must be boolean-like (true/false)",
        error_code="422",
        invalid_component_inputs=[input_name],
    )


def parse_optional_window(value: Any, input_name: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
        if stripped.isdigit():
            parsed = int(stripped)
            if parsed < 1:
                raise ComponentInputValidationException(
                    f"{input_name} must be >= 1",
                    error_code="422",
                    invalid_component_inputs=[input_name],
                )
            return parsed
        return stripped
    if isinstance(value, int):
        if value < 1:
            raise ComponentInputValidationException(
                f"{input_name} must be >= 1",
                error_code="422",
                invalid_component_inputs=[input_name],
            )
        return value
    raise ComponentInputValidationException(
        f"{input_name} must be int, string, or null",
        error_code="422",
        invalid_component_inputs=[input_name],
    )


def parse_threshold(value: Any) -> float | str:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "auto":
            return "auto"
        try:
            parsed = float(stripped)
            if parsed < 0:
                raise ComponentInputValidationException(
                    "threshold must be >= 0",
                    error_code="422",
                    invalid_component_inputs=["threshold"],
                )
            return parsed
        except ValueError as exc:
            raise ComponentInputValidationException(
                "threshold must be 'auto' or numeric",
                error_code="422",
                invalid_component_inputs=["threshold"],
            ) from exc
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed < 0:
            raise ComponentInputValidationException(
                "threshold must be >= 0",
                error_code="422",
                invalid_component_inputs=["threshold"],
            )
        return parsed
    raise ComponentInputValidationException(
        "threshold must be 'auto' or numeric",
        error_code="422",
        invalid_component_inputs=["threshold"],
    )


def validate_base_inputs(
    timeseries: pd.Series,
    method: str,
    derivative_method: str,
    time_unit: str,
    fill_edge: str,
    smoothing_before: str,
    direction: str,
) -> None:
    if not isinstance(timeseries, pd.Series):
        raise ComponentInputValidationException(
            "timeseries must be a pandas Series",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if timeseries.empty:
        raise ComponentInputValidationException(
            "timeseries must not be empty",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if not pd.api.types.is_numeric_dtype(timeseries):
        raise ComponentInputValidationException(
            "timeseries values must be numeric",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )

    valid_methods = {
        "threshold_on_derivative",
        "window_mean_shift",
        "robust_zscore_on_diff",
    }
    if method not in valid_methods:
        raise ComponentInputValidationException(
            f"method must be one of {sorted(valid_methods)}",
            error_code="422",
            invalid_component_inputs=["method"],
        )

    valid_derivative_methods = {"difference", "quotient", "gradient"}
    if derivative_method not in valid_derivative_methods:
        raise ComponentInputValidationException(
            f"derivative_method must be one of {sorted(valid_derivative_methods)}",
            error_code="422",
            invalid_component_inputs=["derivative_method"],
        )

    if time_unit not in {"s", "min", "h", "d"}:
        raise ComponentInputValidationException(
            "time_unit must be one of 's', 'min', 'h', 'd'",
            error_code="422",
            invalid_component_inputs=["time_unit"],
        )

    if fill_edge not in {"nan", "zero", "ffill"}:
        raise ComponentInputValidationException(
            "fill_edge must be one of 'nan', 'zero', 'ffill'",
            error_code="422",
            invalid_component_inputs=["fill_edge"],
        )

    if smoothing_before not in {"none", "moving_average"}:
        raise ComponentInputValidationException(
            "smoothing_before must be one of 'none', 'moving_average'",
            error_code="422",
            invalid_component_inputs=["smoothing_before"],
        )

    if direction not in {"both", "up", "down"}:
        raise ComponentInputValidationException(
            "direction must be one of 'both', 'up', 'down'",
            error_code="422",
            invalid_component_inputs=["direction"],
        )


def prepare_series(timeseries: pd.Series) -> pd.Series:
    prepared = timeseries.sort_index()
    if not prepared.index.is_unique:
        prepared = prepared.groupby(level=0).mean()
    return prepared


def to_time_factor(time_unit: str) -> float:
    return {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}[time_unit]


def calculate_dt(index: pd.Index, time_unit: str) -> pd.Series:
    if not isinstance(index, pd.DatetimeIndex):
        return pd.Series(1.0, index=index)
    factor = to_time_factor(time_unit)
    diffs = index.to_series().diff().dt.total_seconds() / factor
    diffs = diffs.replace(0.0, np.nan)
    return diffs


def apply_smoothing(
    series: pd.Series,
    smoothing_before: str,
    smoothing_window: str | int | None,
) -> pd.Series:
    if smoothing_before == "none":
        return series
    if smoothing_window is None:
        raise ComponentInputValidationException(
            "smoothing_window is required when smoothing_before='moving_average'",
            error_code="422",
            invalid_component_inputs=["smoothing_window"],
        )
    if isinstance(smoothing_window, int):
        return series.rolling(window=smoothing_window, min_periods=1).mean()
    return series.rolling(window=smoothing_window, min_periods=1).mean()


def calculate_derivative(
    series: pd.Series,
    derivative_method: str,
    normalize_by_time: bool,
    time_unit: str,
) -> pd.Series:
    if derivative_method == "difference":
        derivative = series.diff()
        if normalize_by_time:
            dt = calculate_dt(series.index, time_unit)
            derivative = derivative / dt
        return derivative

    if derivative_method == "quotient":
        prev = series.shift(1)
        derivative = (series / prev) - 1.0
        derivative = derivative.replace([np.inf, -np.inf], np.nan)
        if normalize_by_time:
            dt = calculate_dt(series.index, time_unit)
            derivative = derivative / dt
        return derivative

    values = series.to_numpy(dtype=float)
    if isinstance(series.index, pd.DatetimeIndex) and normalize_by_time:
        unit_factor = to_time_factor(time_unit)
        coords = series.index.view("int64") / 1_000_000_000.0 / unit_factor
        grad = np.gradient(values, coords)
    else:
        grad = np.gradient(values)
    return pd.Series(grad, index=series.index)


def apply_fill_edge(derivative: pd.Series, fill_edge: str) -> pd.Series:
    if fill_edge == "zero":
        return derivative.fillna(0.0)
    if fill_edge == "ffill":
        return derivative.ffill().bfill()
    return derivative


def robust_auto_threshold(score: pd.Series) -> float:
    valid = score.dropna()
    if valid.empty:
        return 0.0
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    sigma = 1.4826 * mad
    if sigma == 0.0:
        sigma = float(valid.std())
    if sigma == 0.0 or np.isnan(sigma):
        return 0.0
    return 3.5 * sigma


def passes_direction(magnitude: float, direction: str) -> bool:
    if direction == "both":
        return True
    if direction == "up":
        return magnitude > 0
    return magnitude < 0


def distance_ok(
    current_idx: pd.Timestamp,
    last_idx: pd.Timestamp | None,
    min_distance: str | int,
) -> bool:
    if last_idx is None:
        return True
    if isinstance(min_distance, int):
        return True
    return (current_idx - last_idx) >= pd.to_timedelta(min_distance)


def enforce_min_distance(
    candidates: pd.Index,
    magnitudes: pd.Series,
    min_distance: str | int,
    index_positions: pd.Series,
) -> pd.Index:
    if len(candidates) == 0:
        return candidates

    kept: list[pd.Timestamp] = []
    last_kept_ts: pd.Timestamp | None = None
    last_kept_pos: int | None = None

    for ts in candidates:
        if isinstance(min_distance, int):
            pos = int(index_positions.loc[ts])
            if last_kept_pos is None or pos - last_kept_pos >= min_distance:
                kept.append(ts)
                last_kept_pos = pos
                last_kept_ts = ts
            elif abs(float(magnitudes.loc[ts])) > abs(
                float(magnitudes.loc[last_kept_ts])
            ):
                kept[-1] = ts
                last_kept_pos = pos
                last_kept_ts = ts
        else:
            if distance_ok(ts, last_kept_ts, min_distance):
                kept.append(ts)
                last_kept_ts = ts
            elif abs(float(magnitudes.loc[ts])) > abs(
                float(magnitudes.loc[last_kept_ts])
            ):
                kept[-1] = ts
                last_kept_ts = ts

    return pd.Index(kept)


def merge_close(
    events_idx: pd.Index,
    magnitudes: pd.Series,
    merge_distance: str | int,
    index_positions: pd.Series,
) -> pd.Index:
    if len(events_idx) <= 1:
        return events_idx
    return enforce_min_distance(events_idx, magnitudes, merge_distance, index_positions)


def detect_threshold_on_derivative(
    derivative: pd.Series,
    threshold: float | str,
) -> tuple[pd.Series, float]:
    score = derivative.abs()
    used_threshold = (
        robust_auto_threshold(derivative) if threshold == "auto" else float(threshold)
    )
    return score, used_threshold


def detect_robust_zscore_on_diff(
    series: pd.Series,
    threshold: float | str,
) -> tuple[pd.Series, pd.Series, float]:
    diff_signal = series.diff()
    med = diff_signal.median()
    mad = (diff_signal - med).abs().median()
    scale = 1.4826 * mad
    if scale == 0 or np.isnan(scale):
        scale = diff_signal.std()
    if scale == 0 or np.isnan(scale):
        z = pd.Series(0.0, index=series.index)
    else:
        z = (diff_signal - med) / scale
    score = z.abs()
    used_threshold = 3.5 if threshold == "auto" else float(threshold)
    return score, diff_signal, used_threshold


def rolling_forward_mean(series: pd.Series, window: str | int) -> pd.Series:
    if isinstance(window, int):
        return series.iloc[::-1].rolling(window=window, min_periods=1).mean().iloc[::-1]
    return series.iloc[::-1].rolling(window=window, min_periods=1).mean().iloc[::-1]


def detect_window_mean_shift(
    series: pd.Series,
    window: str | int | None,
    threshold: float | str,
) -> tuple[pd.Series, pd.Series, float]:
    if window is None:
        raise ComponentInputValidationException(
            "window is required when method='window_mean_shift'",
            error_code="422",
            invalid_component_inputs=["window"],
        )
    left_mean = series.rolling(window=window, min_periods=1).mean()
    right_mean = rolling_forward_mean(series, window)
    shift = right_mean - left_mean
    score = shift.abs()
    used_threshold = (
        robust_auto_threshold(shift) if threshold == "auto" else float(threshold)
    )
    return score, shift, used_threshold


def build_segment_means(series: pd.Series, event_index: pd.Index) -> pd.Series:
    if series.empty:
        return series
    if len(event_index) == 0:
        return pd.Series(series.mean(), index=series.index)

    boundaries = [series.index[0], *event_index.tolist(), series.index[-1]]
    output = pd.Series(index=series.index, dtype=float)

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if left == right:
            segment_slice = series.loc[[left]]
        else:
            segment_slice = series.loc[left:right]
        segment_mean = float(segment_slice.mean())
        output.loc[segment_slice.index] = segment_mean
    return output


# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "timeseries": {"data_type": "SERIES"},
        "method": {
            "data_type": "STRING",
            "default_value": "threshold_on_derivative",
        },
        "derivative": {"data_type": "SERIES", "default_value": None},
        "derivative_method": {"data_type": "STRING", "default_value": "difference"},
        "normalize_by_time": {"data_type": "BOOLEAN", "default_value": True},
        "time_unit": {"data_type": "STRING", "default_value": "s"},
        "fill_edge": {"data_type": "STRING", "default_value": "nan"},
        "smoothing_before": {"data_type": "STRING", "default_value": "none"},
        "smoothing_window": {"data_type": "STRING", "default_value": None},
        "threshold": {"data_type": "STRING", "default_value": "auto"},
        "window": {"data_type": "STRING", "default_value": None},
        "min_distance": {"data_type": "STRING", "default_value": "1"},
        "direction": {"data_type": "STRING", "default_value": "both"},
        "merge_close_events": {"data_type": "BOOLEAN", "default_value": True},
        "merge_distance": {"data_type": "STRING", "default_value": None},
        "output_segments": {"data_type": "BOOLEAN", "default_value": True},
    },
    "outputs": {
        "jump_mask": {"data_type": "SERIES"},
        "jump_events": {"data_type": "DATAFRAME"},
    },
    "name": "Jump Detection",
    "category": "Base Components",
    "description": "Detect jumps and return mask, event table, and segment means.",
    "version_tag": "1.0.0",
    "id": "3f6f8f64-5b12-4a9a-a8e2-12d9d5206d13",
    "revision_group_id": "3f6f8f64-5b12-4a9a-a8e2-12d9d5206d13",
    "state": "DRAFT",
}


def main(
    *,
    timeseries,
    method=parse_default_value(COMPONENT_INFO, "method"),
    derivative=parse_default_value(COMPONENT_INFO, "derivative"),
    derivative_method=parse_default_value(COMPONENT_INFO, "derivative_method"),
    normalize_by_time=parse_default_value(COMPONENT_INFO, "normalize_by_time"),
    time_unit=parse_default_value(COMPONENT_INFO, "time_unit"),
    fill_edge=parse_default_value(COMPONENT_INFO, "fill_edge"),
    smoothing_before=parse_default_value(COMPONENT_INFO, "smoothing_before"),
    smoothing_window=parse_default_value(COMPONENT_INFO, "smoothing_window"),
    threshold=parse_default_value(COMPONENT_INFO, "threshold"),
    window=parse_default_value(COMPONENT_INFO, "window"),
    min_distance=parse_default_value(COMPONENT_INFO, "min_distance"),
    direction=parse_default_value(COMPONENT_INFO, "direction"),
    merge_close_events=parse_default_value(COMPONENT_INFO, "merge_close_events"),
    merge_distance=parse_default_value(COMPONENT_INFO, "merge_distance"),
    output_segments=parse_default_value(COMPONENT_INFO, "output_segments"),
):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    prepared = prepare_series(timeseries)

    normalize_by_time = parse_bool_like(normalize_by_time, "normalize_by_time")
    merge_close_events = parse_bool_like(merge_close_events, "merge_close_events")
    output_segments = parse_bool_like(output_segments, "output_segments")

    threshold = parse_threshold(threshold)
    smoothing_window = parse_optional_window(smoothing_window, "smoothing_window")
    window = parse_optional_window(window, "window")
    min_distance = parse_optional_window(min_distance, "min_distance")
    merge_distance = parse_optional_window(merge_distance, "merge_distance")

    validate_base_inputs(
        prepared,
        method,
        derivative_method,
        time_unit,
        fill_edge,
        smoothing_before,
        direction,
    )

    if isinstance(prepared.index, pd.DatetimeIndex):
        prepared = (
            prepared.tz_convert("UTC") if prepared.index.tz is not None else prepared
        )
    elif isinstance(min_distance, str) or isinstance(merge_distance, str):
        raise ComponentInputValidationException(
            "min_distance/merge_distance as time string require a DatetimeIndex",
            error_code="422",
            invalid_component_inputs=["min_distance", "merge_distance"],
        )

    smoothed = apply_smoothing(prepared, smoothing_before, smoothing_window)

    if isinstance(derivative, pd.Series):
        if not pd.api.types.is_numeric_dtype(derivative):
            raise ComponentInputValidationException(
                "derivative values must be numeric",
                error_code="422",
                invalid_component_inputs=["derivative"],
            )
        derivative_series = derivative.sort_index().reindex(smoothed.index)
    else:
        derivative_series = calculate_derivative(
            smoothed,
            derivative_method=derivative_method,
            normalize_by_time=normalize_by_time,
            time_unit=time_unit,
        )
    derivative_series = apply_fill_edge(derivative_series, fill_edge)

    if method == "threshold_on_derivative":
        score, used_threshold = detect_threshold_on_derivative(
            derivative_series, threshold
        )
        magnitudes = derivative_series
    elif method == "robust_zscore_on_diff":
        score, diff_signal, used_threshold = detect_robust_zscore_on_diff(
            smoothed, threshold
        )
        magnitudes = diff_signal
    else:
        score, shift_signal, used_threshold = detect_window_mean_shift(
            smoothed, window, threshold
        )
        magnitudes = shift_signal

    candidate_mask = score >= used_threshold
    candidate_mask = candidate_mask.fillna(False)

    if direction != "both":
        direction_mask = magnitudes.apply(
            lambda x: passes_direction(float(x) if pd.notna(x) else 0.0, direction)
        )
        candidate_mask = candidate_mask & direction_mask

    candidate_index = candidate_mask[candidate_mask].index
    positions = pd.Series(np.arange(len(smoothed.index)), index=smoothed.index)

    min_distance_value = min_distance if min_distance is not None else 1
    filtered_index = enforce_min_distance(
        candidate_index, magnitudes, min_distance_value, positions
    )

    if merge_close_events:
        merge_distance_value = (
            merge_distance if merge_distance is not None else min_distance_value
        )
        filtered_index = merge_close(
            filtered_index, magnitudes, merge_distance_value, positions
        )

    jump_mask = pd.Series(False, index=smoothed.index)
    if len(filtered_index) > 0:
        jump_mask.loc[filtered_index] = True

    jump_events = pd.DataFrame(
        {
            "timestamp": filtered_index.astype(str),
            "magnitude": magnitudes.reindex(filtered_index).astype(float).to_numpy(),
        }
    )
    if not jump_events.empty:
        jump_events["direction"] = jump_events["magnitude"].apply(
            lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")
        )
        jump_events["confidence"] = jump_events["magnitude"].abs() / max(
            used_threshold, 1e-12
        )
    else:
        jump_events["direction"] = pd.Series(dtype=str)
        jump_events["confidence"] = pd.Series(dtype=float)
    jump_events["method"] = method

    return {
        "jump_mask": jump_mask,
        "jump_events": jump_events,
    }


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{"2025-01-12T00:00:00Z": 10, "2025-01-12T00:05:00Z": 10, "2025-01-12T00:10:00Z": 11, "2025-01-12T00:15:00Z": 12, "2025-01-12T00:20:00Z": 26, "2025-01-12T00:25:00Z": 27, "2025-01-12T00:30:00Z": 27, "2025-01-12T00:35:00Z": 13, "2025-01-12T00:40:00Z": 13, "2025-01-12T00:45:00Z": 12, "2025-01-12T00:50:00Z": 12, "2025-01-12T00:55:00Z": 20, "2025-01-12T01:00:00Z": 21}'
            },
        },
        {
            "workflow_input_name": "method",
            "filters": {"value": "threshold_on_derivative"},
        },
        {
            "workflow_input_name": "derivative_method",
            "filters": {"value": "difference"},
        },
        {"workflow_input_name": "normalize_by_time", "filters": {"value": "True"}},
        {"workflow_input_name": "time_unit", "filters": {"value": "s"}},
        {"workflow_input_name": "fill_edge", "filters": {"value": "nan"}},
        {"workflow_input_name": "smoothing_before", "filters": {"value": "none"}},
        {"workflow_input_name": "threshold", "filters": {"value": "auto"}},
        {"workflow_input_name": "window", "filters": {"value": "None"}},
        {"workflow_input_name": "min_distance", "filters": {"value": "2"}},
        {"workflow_input_name": "direction", "filters": {"value": "both"}},
        {"workflow_input_name": "merge_close_events", "filters": {"value": "True"}},
        {"workflow_input_name": "merge_distance", "filters": {"value": "None"}},
        {"workflow_input_name": "output_segments", "filters": {"value": "True"}},
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{"2025-01-12T00:00:00Z": 10, "2025-01-12T00:05:00Z": 10, "2025-01-12T00:10:00Z": 11, "2025-01-12T00:15:00Z": 12, "2025-01-12T00:20:00Z": 26, "2025-01-12T00:25:00Z": 27, "2025-01-12T00:30:00Z": 27, "2025-01-12T00:35:00Z": 13, "2025-01-12T00:40:00Z": 13, "2025-01-12T00:45:00Z": 12, "2025-01-12T00:50:00Z": 12, "2025-01-12T00:55:00Z": 20, "2025-01-12T01:00:00Z": 21}'
            },
        },
        {
            "workflow_input_name": "method",
            "filters": {"value": "threshold_on_derivative"},
        },
        {
            "workflow_input_name": "derivative_method",
            "filters": {"value": "difference"},
        },
        {"workflow_input_name": "normalize_by_time", "filters": {"value": "True"}},
        {"workflow_input_name": "time_unit", "filters": {"value": "s"}},
        {"workflow_input_name": "fill_edge", "filters": {"value": "nan"}},
        {"workflow_input_name": "smoothing_before", "filters": {"value": "none"}},
        {"workflow_input_name": "threshold", "filters": {"value": "auto"}},
        {"workflow_input_name": "window", "filters": {"value": "None"}},
        {"workflow_input_name": "min_distance", "filters": {"value": "2"}},
        {"workflow_input_name": "direction", "filters": {"value": "both"}},
        {"workflow_input_name": "merge_close_events", "filters": {"value": "True"}},
        {"workflow_input_name": "merge_distance", "filters": {"value": "None"}},
        {"workflow_input_name": "output_segments", "filters": {"value": "True"}},
    ]
}
