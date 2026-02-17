"""Documentation for Handle Gaps and Missing Data

# Handle Gaps and Missing Data

## Description
Single-point component to detect gaps (missing values and missing timestamps),
optionally fill them, and return gap masks plus a gap event list.

## Inputs
- **timeseries** (Pandas Series):
    The input time series. Index must be datetime, values numeric.
- **drop_na** (Boolean, default value: False):
    If True, drop all NaN values after optional filling.
- **mode** (String, default value: "fill"):
    One of "fill", "flag", "drop".
- **method** (String, default value: "time"):
    Filling method. One of "time", "linear", "ffill", "bfill", "constant".
- **limit_direction** (String, default value: "both"):
    Interpolation direction for "time"/"linear" ("forward", "backward", "both").
- **min_gap_length** (Integer, default value: 1):
    Minimum number of consecutive missing points to consider a gap fillable.
- **max_gap_length** (Integer, default value: null):
    Maximum length of fillable gaps. If null, no upper limit.
- **constant_value** (Float, default value: 0):
    Constant value used when method="constant".
- **resample_to** (String, default value: null):
    Optional target frequency (e.g. "10min") to create a regular grid.
    Missing timestamps on the grid are treated as gaps.
- **auto_frequency_determination** (Boolean, default value: False):
    If True and resample_to is not set, infer a regular frequency from the
    median time difference and resample to that grid.

## Outputs
- **corrected_timeseries** (Pandas Series):
    The resulting series (filled/flagged/dropped depending on mode).
- **gap_mask** (Pandas Series):
    Boolean series; True where values are still missing after processing.
- **filled_mask** (Pandas Series):
    Boolean series; True where values were filled.
- **gap_events** (List[Dict]):
    List of gap events with start/end/duration/num_points.

## Details
1. Sorts the input by time and removes duplicate timestamps (keeps the mean).
2. Optionally resamples to a regular grid to make missing timestamps visible.
3. Detects gaps as NaN values and as missing timestamps on the grid.
4. If resample_to is set, it takes precedence over auto-frequency detection.
5. Optionally infers a regular grid from the median time difference.
6. Fills only gaps within the configured length limits.
7. Returns the processed series, masks, and a list of gap events.

## Example
```json
{
  "timeseries": {
    "2026-01-12T00:00:00Z": 10,
    "2026-01-12T00:07:00Z": 12,
    "2026-01-12T00:13:00Z": 13,
    "2026-01-12T00:18:00Z": 14,
    "2026-01-12T00:22:00Z": 15,
    "2026-01-12T00:25:00Z": 15,
    "2026-01-12T00:32:00Z": 15,
    "2026-01-12T00:38:00Z": 14,
    "2026-01-12T00:43:00Z": 13,
    "2026-01-12T00:47:00Z": 12,
    "2026-01-12T00:50:00Z": 11,
    "2026-01-12T00:57:00Z": 10,
    "2026-01-12T01:03:00Z": null,
    "2026-01-12T01:08:00Z": 9,
    "2026-01-12T01:12:00Z": 9,
    "2026-01-12T01:15:00Z": 10,
    "2026-01-12T01:22:00Z": 11,
    "2026-01-12T01:28:00Z": 12,
    "2026-01-12T01:33:00Z": 14,
    "2026-01-12T01:37:00Z": 15,
    "2026-01-12T01:40:00Z": 17,
    "2026-01-12T01:47:00Z": 18,
    "2026-01-12T01:53:00Z": 18,
    "2026-01-12T01:58:00Z": 19,
    "2026-01-12T02:02:00Z": 18,
    "2026-01-12T02:05:00Z": 18,
    "2026-01-12T02:12:00Z": 16,
    "2026-01-12T02:18:00Z": 15,
    "2026-01-12T02:23:00Z": 14,
    "2026-01-12T02:27:00Z": 13,
    "2026-01-12T02:30:00Z": null,
    "2026-01-12T02:37:00Z": null,
    "2026-01-12T02:43:00Z": null,
    "2026-01-12T02:48:00Z": 13,
    "2026-01-12T02:52:00Z": 14,
    "2026-01-12T02:55:00Z": 15,
    "2026-01-12T03:02:00Z": 16,
    "2026-01-12T03:08:00Z": 18,
    "2026-01-12T03:13:00Z": 19,
    "2026-01-12T03:17:00Z": 20,
    "2026-01-12T03:20:00Z": 21,
    "2026-01-12T03:27:00Z": 21,
    "2026-01-12T03:33:00Z": 20,
    "2026-01-12T03:38:00Z": 20,
    "2026-01-12T03:42:00Z": 19,
    "2026-01-12T03:45:00Z": 17,
    "2026-01-12T03:52:00Z": 16,
    "2026-01-12T03:58:00Z": 15,
    "2026-01-12T04:03:00Z": 14,
    "2026-01-12T04:07:00Z": 13,
    "2026-01-12T04:10:00Z": 13,
    "2026-01-12T04:17:00Z": 14,
    "2026-01-12T04:23:00Z": 15,
    "2026-01-12T04:28:00Z": 16,
    "2026-01-12T04:32:00Z": null,
    "2026-01-12T04:35:00Z": null,
    "2026-01-12T04:42:00Z": null,
    "2026-01-12T04:48:00Z": null,
    "2026-01-12T04:53:00Z": null,
    "2026-01-12T04:57:00Z": null,
    "2026-01-12T05:00:00Z": 21,
    "2026-01-12T05:07:00Z": 20,
    "2026-01-12T05:13:00Z": 19
},
  "max_gap_length": 4
}
```
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hdutils import ComponentInputValidationException, parse_default_value


def validate_inputs(
    series: pd.Series,
    mode: str,
    method: str,
    limit_direction: str,
    min_gap_length: int,
    max_gap_length: int | None,
) -> None:
    if not isinstance(series, pd.Series):
        raise ComponentInputValidationException(
            "timeseries must be a pandas Series",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ComponentInputValidationException(
            "timeseries index must be datetime",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if mode not in {"fill", "flag", "drop"}:
        raise ComponentInputValidationException(
            f"mode must be one of 'fill', 'flag', 'drop', got '{mode}'",
            error_code="422",
            invalid_component_inputs=["mode"],
        )
    if method not in {"time", "linear", "ffill", "bfill", "constant"}:
        raise ComponentInputValidationException(
            f"method must be one of 'time', 'linear', 'ffill', 'bfill', 'constant', got '{method}'",
            error_code="422",
            invalid_component_inputs=["method"],
        )
    if limit_direction not in {"forward", "backward", "both"}:
        raise ComponentInputValidationException(
            "limit_direction must be one of 'forward', 'backward', 'both'",
            error_code="422",
            invalid_component_inputs=["limit_direction"],
        )
    if not isinstance(min_gap_length, int):
        raise ComponentInputValidationException(
            "min_gap_length must be an integer >= 1",
            error_code="422",
            invalid_component_inputs=["min_gap_length"],
        )
    if min_gap_length < 1:
        raise ComponentInputValidationException(
            "min_gap_length must be >= 1",
            error_code="422",
            invalid_component_inputs=["min_gap_length"],
        )
    if max_gap_length is not None and not isinstance(max_gap_length, int):
        raise ComponentInputValidationException(
            "max_gap_length must be an integer >= min_gap_length or null",
            error_code="422",
            invalid_component_inputs=["max_gap_length"],
        )
    if max_gap_length is not None and max_gap_length < min_gap_length:
        raise ComponentInputValidationException(
            "max_gap_length must be >= min_gap_length",
            error_code="422",
            invalid_component_inputs=["max_gap_length"],
        )


def gap_lengths(mask: pd.Series) -> pd.Series:
    """Return gap length for each position in a boolean gap mask."""
    group = (mask != mask.shift()).cumsum()
    return mask.groupby(group).transform("sum")


def gap_events(mask: pd.Series) -> list[dict[str, Any]]:
    if mask.empty or not mask.any():
        return []

    events: list[dict[str, Any]] = []
    idx = mask.index
    mask_values = mask.to_numpy()
    starts = np.flatnonzero((~mask_values[:-1]) & mask_values[1:]) + 1
    if mask_values[0]:
        starts = np.r_[0, starts]
    ends = np.flatnonzero(mask_values[:-1] & (~mask_values[1:]))
    if mask_values[-1]:
        ends = np.r_[ends, len(mask_values) - 1]

    for start_pos, end_pos in zip(starts, ends):
        start_ts = idx[start_pos]
        end_ts = idx[end_pos]
        duration = end_ts - start_ts if start_ts is not None else None
        events.append(
            {
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "num_points": int(end_pos - start_pos + 1),
                "duration": str(duration),
            }
        )
    return events


def prepare_series(
    series: pd.Series,
    resample_to: str | None,
    auto_frequency_determination: bool,
) -> pd.Series:
    ordered = series.sort_index()
    if not ordered.index.is_unique:
        ordered = ordered.groupby(level=0).mean()
    resample_value = resample_to
    if resample_value is False or resample_value == "":
        resample_value = None
    if resample_value is not None and not isinstance(resample_value, str):
        raise ComponentInputValidationException(
            "resample_to must be a string or null",
            error_code="422",
            invalid_component_inputs=["resample_to"],
        )
    if resample_value:
        try:
            full_index = pd.date_range(
                start=ordered.index.min(), end=ordered.index.max(), freq=resample_value
            )
        except (ValueError, TypeError) as exc:
            raise ComponentInputValidationException(
                f"resample_to could not be parsed as frequency: {resample_value}",
                error_code="422",
                invalid_component_inputs=["resample_to"],
            ) from exc
        ordered = ordered.reindex(full_index)
    elif auto_frequency_determination and len(ordered.index) > 1:
        diffs = ordered.index.to_series().diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        if positive_diffs.empty:
            raise ComponentInputValidationException(
                "Cannot infer frequency from timestamps after sorting",
                error_code="422",
                invalid_component_inputs=["auto_frequency_determination"],
            )
        inferred = positive_diffs.median()
        if inferred <= pd.Timedelta(0):
            raise ComponentInputValidationException(
                "Inferred frequency must be positive",
                error_code="422",
                invalid_component_inputs=["auto_frequency_determination"],
            )
        full_index = pd.date_range(
            start=ordered.index.min(), end=ordered.index.max(), freq=inferred
        )
        ordered = ordered.reindex(full_index)
    return ordered


def fill_series(
    series: pd.Series,
    fillable_mask: pd.Series,
    method: str,
    limit_direction: str,
    constant_value: float,
) -> pd.Series:
    if method == "constant":
        filled = series.copy()
        filled.loc[fillable_mask] = constant_value
        return filled

    if method in {"ffill", "bfill"}:
        filled = series.ffill() if method == "ffill" else series.bfill()
    else:
        filled = series.interpolate(method=method, limit_direction=limit_direction)

    # Restore non-fillable missing points
    filled.loc[~fillable_mask & series.isna()] = np.nan
    return filled


# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "timeseries": {"data_type": "SERIES"},
        "drop_na": {"data_type": "BOOLEAN", "default_value": False},
        "mode": {"data_type": "STRING", "default_value": "fill"},
        "method": {"data_type": "STRING", "default_value": "time"},
        "limit_direction": {"data_type": "STRING", "default_value": "both"},
        "auto_frequency_determination": {
            "data_type": "BOOLEAN",
            "default_value": False,
        },
        "min_gap_length": {"data_type": "INT", "default_value": 1},
        "max_gap_length": {"data_type": "INT", "default_value": None},
        "constant_value": {"data_type": "FLOAT", "default_value": 0},
        "resample_to": {"data_type": "STRING", "default_value": None},
    },
    "outputs": {
        "corrected_timeseries": {"data_type": "SERIES"},
        "gap_mask": {"data_type": "SERIES"},
        "filled_mask": {"data_type": "SERIES"},
        "gap_events": {"data_type": "ANY"},
    },
    "name": "Handle Gaps and Missing Data",
    "category": "Base Components",
    "description": "Detect and optionally fill gaps in time series, returning masks and events.",
    "version_tag": "1.0.0",
    "id": "1791fa22-f749-4145-ab19-ab88576335b2",
    "revision_group_id": "cf8a9dda-ff6e-4540-bd0a-246e3f746e0a",
    "state": "DRAFT",
}


def main(
    *,
    timeseries,
    drop_na=parse_default_value(COMPONENT_INFO, "drop_na"),
    mode=parse_default_value(COMPONENT_INFO, "mode"),
    method=parse_default_value(COMPONENT_INFO, "method"),
    limit_direction=parse_default_value(COMPONENT_INFO, "limit_direction"),
    auto_frequency_determination=parse_default_value(
        COMPONENT_INFO, "auto_frequency_determination"
    ),
    min_gap_length=parse_default_value(COMPONENT_INFO, "min_gap_length"),
    max_gap_length=None,
    constant_value=parse_default_value(COMPONENT_INFO, "constant_value"),
    resample_to=None,
):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    validate_inputs(
        timeseries,
        mode,
        method,
        limit_direction,
        min_gap_length,
        max_gap_length,
    )
    series = prepare_series(timeseries, resample_to, auto_frequency_determination)

    missing_mask = series.isna()
    gap_lengths_values = gap_lengths(missing_mask)
    fillable_mask = missing_mask & (gap_lengths_values >= min_gap_length)
    if max_gap_length is not None:
        fillable_mask &= gap_lengths_values <= max_gap_length

    if mode == "fill":
        processed = fill_series(
            series,
            fillable_mask,
            method,
            limit_direction,
            constant_value,
        )
    elif mode == "drop":
        processed = series.copy()
        processed = processed.dropna()
    else:
        processed = series.copy()

    filled_mask = (
        (fillable_mask & processed.notna())
        if mode == "fill"
        else pd.Series(False, index=series.index)
    )
    gap_mask = (
        processed.isna() if mode != "drop" else pd.Series(False, index=processed.index)
    )

    if drop_na and mode != "drop":
        processed = processed.dropna()
        filled_mask = filled_mask.reindex(processed.index, fill_value=False)
        gap_mask = gap_mask.reindex(processed.index, fill_value=False)

    events = gap_events(missing_mask)

    return {
        "corrected_timeseries": processed,
        "gap_mask": gap_mask,
        "filled_mask": filled_mask,
        "gap_events": events,
    }


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{\n    "2026-01-12T00:00:00Z": 10,\n    "2026-01-12T00:07:00Z": 12,\n    "2026-01-12T00:13:00Z": 13,\n    "2026-01-12T00:18:00Z": 14,\n    "2026-01-12T00:22:00Z": 15,\n    "2026-01-12T00:25:00Z": 15,\n    "2026-01-12T00:32:00Z": 15,\n    "2026-01-12T00:38:00Z": 14,\n    "2026-01-12T00:43:00Z": 13,\n    "2026-01-12T00:47:00Z": 12,\n    "2026-01-12T00:50:00Z": 11,\n    "2026-01-12T00:57:00Z": 10,\n    "2026-01-12T01:03:00Z": null,\n    "2026-01-12T01:08:00Z": 9,\n    "2026-01-12T01:12:00Z": 9,\n    "2026-01-12T01:15:00Z": 10,\n    "2026-01-12T01:22:00Z": 11,\n    "2026-01-12T01:28:00Z": 12,\n    "2026-01-12T01:33:00Z": 14,\n    "2026-01-12T01:37:00Z": 15,\n    "2026-01-12T01:40:00Z": 17,\n    "2026-01-12T01:47:00Z": 18,\n    "2026-01-12T01:53:00Z": 18,\n    "2026-01-12T01:58:00Z": 19,\n    "2026-01-12T02:02:00Z": 18,\n    "2026-01-12T02:05:00Z": 18,\n    "2026-01-12T02:12:00Z": 16,\n    "2026-01-12T02:18:00Z": 15,\n    "2026-01-12T02:23:00Z": 14,\n    "2026-01-12T02:27:00Z": 13,\n    "2026-01-12T02:30:00Z": null,\n    "2026-01-12T02:37:00Z": null,\n    "2026-01-12T02:43:00Z": null,\n    "2026-01-12T02:48:00Z": 13,\n    "2026-01-12T02:52:00Z": 14,\n    "2026-01-12T02:55:00Z": 15,\n    "2026-01-12T03:02:00Z": 16,\n    "2026-01-12T03:08:00Z": 18,\n    "2026-01-12T03:13:00Z": 19,\n    "2026-01-12T03:17:00Z": 20,\n    "2026-01-12T03:20:00Z": 21,\n    "2026-01-12T03:27:00Z": 21,\n    "2026-01-12T03:33:00Z": 20,\n    "2026-01-12T03:38:00Z": 20,\n    "2026-01-12T03:42:00Z": 19,\n    "2026-01-12T03:45:00Z": 17,\n    "2026-01-12T03:52:00Z": 16,\n    "2026-01-12T03:58:00Z": 15,\n    "2026-01-12T04:03:00Z": 14,\n    "2026-01-12T04:07:00Z": 13,\n    "2026-01-12T04:10:00Z": 13,\n    "2026-01-12T04:17:00Z": 14,\n    "2026-01-12T04:23:00Z": 15,\n    "2026-01-12T04:28:00Z": 16,\n    "2026-01-12T04:32:00Z": null,\n    "2026-01-12T04:35:00Z": null,\n    "2026-01-12T04:42:00Z": null,\n    "2026-01-12T04:48:00Z": null,\n    "2026-01-12T04:53:00Z": null,\n    "2026-01-12T04:57:00Z": null,\n    "2026-01-12T05:00:00Z": 21,\n    "2026-01-12T05:07:00Z": 20,\n    "2026-01-12T05:13:00Z": 19,\n    "2026-01-12T05:18:00Z": 17,\n    "2026-01-12T05:22:00Z": 16,\n    "2026-01-12T05:25:00Z": 15,\n    "2026-01-12T05:32:00Z": 14,\n    "2026-01-12T05:38:00Z": 13,\n    "2026-01-12T05:43:00Z": 13,\n    "2026-01-12T05:47:00Z": 14,\n    "2026-01-12T05:50:00Z": 15,\n    "2026-01-12T05:57:00Z": 16,\n    "2026-01-12T06:03:00Z": 17,\n    "2026-01-12T06:08:00Z": 18,\n    "2026-01-12T06:12:00Z": 19,\n    "2026-01-12T06:15:00Z": 20,\n    "2026-01-12T06:22:00Z": 21,\n    "2026-01-12T06:28:00Z": 21,\n    "2026-01-12T06:33:00Z": 20,\n    "2026-01-12T06:37:00Z": 19,\n    "2026-01-12T06:40:00Z": 18,\n    "2026-01-12T06:47:00Z": 16,\n    "2026-01-12T06:53:00Z": 15,\n    "2026-01-12T06:58:00Z": 14,\n    "2026-01-12T07:02:00Z": 13,\n    "2026-01-12T07:05:00Z": 12,\n    "2026-01-12T07:12:00Z": 12,\n    "2026-01-12T07:18:00Z": 12,\n    "2026-01-12T07:23:00Z": 13,\n    "2026-01-12T07:27:00Z": 14,\n    "2026-01-12T07:30:00Z": 15,\n    "2026-01-12T07:37:00Z": 17,\n    "2026-01-12T07:43:00Z": 18,\n    "2026-01-12T07:48:00Z": 19,\n    "2026-01-12T07:52:00Z": 19,\n    "2026-01-12T07:55:00Z": 19,\n    "2026-01-12T08:02:00Z": null,\n    "2026-01-12T08:08:00Z": null,\n    "2026-01-12T08:13:00Z": null,\n    "2026-01-12T08:17:00Z": null,\n    "2026-01-12T08:20:00Z": null,\n    "2026-01-12T08:27:00Z": null,\n    "2026-01-12T08:33:00Z": null,\n    "2026-01-12T08:38:00Z": null,\n    "2026-01-12T08:42:00Z": null,\n    "2026-01-12T08:45:00Z": null,\n    "2026-01-12T08:52:00Z": null,\n    "2026-01-12T08:58:00Z": null,\n    "2026-01-12T09:03:00Z": null,\n    "2026-01-12T09:07:00Z": null,\n    "2026-01-12T09:10:00Z": null,\n    "2026-01-12T09:17:00Z": null,\n    "2026-01-12T09:23:00Z": null,\n    "2026-01-12T09:28:00Z": null,\n    "2026-01-12T09:32:00Z": null,\n    "2026-01-12T09:35:00Z": null,\n    "2026-01-12T09:42:00Z": null,\n    "2026-01-12T09:48:00Z": null,\n    "2026-01-12T09:53:00Z": null,\n    "2026-01-12T09:57:00Z": null,\n    "2026-01-12T10:00:00Z": null,\n    "2026-01-12T10:07:00Z": null,\n    "2026-01-12T10:13:00Z": null,\n    "2026-01-12T10:18:00Z": null,\n    "2026-01-12T10:22:00Z": null,\n    "2026-01-12T10:25:00Z": null,\n    "2026-01-12T10:32:00Z": 12,\n    "2026-01-12T10:38:00Z": 13,\n    "2026-01-12T10:43:00Z": 14,\n    "2026-01-12T10:47:00Z": 15,\n    "2026-01-12T10:50:00Z": 15,\n    "2026-01-12T10:57:00Z": 15,\n    "2026-01-12T11:03:00Z": 15,\n    "2026-01-12T11:08:00Z": 14,\n    "2026-01-12T11:12:00Z": 12,\n    "2026-01-12T11:15:00Z": 11,\n    "2026-01-12T11:22:00Z": 9,\n    "2026-01-12T11:28:00Z": 8,\n    "2026-01-12T11:33:00Z": 7,\n    "2026-01-12T11:37:00Z": 6,\n    "2026-01-12T11:40:00Z": null,\n    "2026-01-12T11:47:00Z": null,\n    "2026-01-12T11:53:00Z": 8,\n    "2026-01-12T11:58:00Z": 9,\n    "2026-01-12T12:02:00Z": 10,\n    "2026-01-12T12:05:00Z": 11,\n    "2026-01-12T12:12:00Z": 12,\n    "2026-01-12T12:18:00Z": 13,\n    "2026-01-12T12:23:00Z": 14,\n    "2026-01-12T12:27:00Z": 14,\n    "2026-01-12T12:30:00Z": 13,\n    "2026-01-12T12:37:00Z": 12,\n    "2026-01-12T12:43:00Z": 11,\n    "2026-01-12T12:48:00Z": 9,\n    "2026-01-12T12:52:00Z": 8,\n    "2026-01-12T12:55:00Z": 6,\n    "2026-01-12T13:02:00Z": 5,\n    "2026-01-12T13:08:00Z": 5,\n    "2026-01-12T13:13:00Z": 5,\n    "2026-01-12T13:17:00Z": 5,\n    "2026-01-12T13:20:00Z": 6,\n    "2026-01-12T13:27:00Z": 7,\n    "2026-01-12T13:33:00Z": 8,\n    "2026-01-12T13:38:00Z": 10,\n    "2026-01-12T13:42:00Z": 11,\n    "2026-01-12T13:45:00Z": null,\n    "2026-01-12T13:52:00Z": null,\n    "2026-01-12T13:58:00Z": null,\n    "2026-01-12T14:03:00Z": null,\n    "2026-01-12T14:07:00Z": null,\n    "2026-01-12T14:10:00Z": null,\n    "2026-01-12T14:17:00Z": null,\n    "2026-01-12T14:23:00Z": null,\n    "2026-01-12T14:28:00Z": 5,\n    "2026-01-12T14:32:00Z": 4,\n    "2026-01-12T14:35:00Z": 3,\n    "2026-01-12T14:42:00Z": 3,\n    "2026-01-12T14:48:00Z": 3,\n    "2026-01-12T14:53:00Z": 4,\n    "2026-01-12T14:57:00Z": 5,\n    "2026-01-12T15:00:00Z": 7,\n    "2026-01-12T15:07:00Z": 8,\n    "2026-01-12T15:13:00Z": 9,\n    "2026-01-12T15:18:00Z": 10,\n    "2026-01-12T15:22:00Z": 10,\n    "2026-01-12T15:25:00Z": 10,\n    "2026-01-12T15:32:00Z": 9,\n    "2026-01-12T15:38:00Z": 8,\n    "2026-01-12T15:43:00Z": 7,\n    "2026-01-12T15:47:00Z": 6,\n    "2026-01-12T15:50:00Z": 4,\n    "2026-01-12T15:57:00Z": 3,\n    "2026-01-12T16:03:00Z": 2,\n    "2026-01-12T16:08:00Z": 1,\n    "2026-01-12T16:12:00Z": 1,\n    "2026-01-12T16:15:00Z": 1,\n    "2026-01-12T16:22:00Z": 2,\n    "2026-01-12T16:28:00Z": 3,\n    "2026-01-12T16:33:00Z": 5,\n    "2026-01-12T16:37:00Z": 6,\n    "2026-01-12T16:40:00Z": null,\n    "2026-01-12T16:47:00Z": null,\n    "2026-01-12T16:53:00Z": null,\n    "2026-01-12T16:58:00Z": null,\n    "2026-01-12T17:02:00Z": 7,\n    "2026-01-12T17:05:00Z": 6,\n    "2026-01-12T17:12:00Z": 5,\n    "2026-01-12T17:18:00Z": 4,\n    "2026-01-12T17:23:00Z": 2,\n    "2026-01-12T17:27:00Z": 1,\n    "2026-01-12T17:30:00Z": 0,\n    "2026-01-12T17:37:00Z": -1,\n    "2026-01-12T17:43:00Z": -1,\n    "2026-01-12T17:48:00Z": 0,\n    "2026-01-12T17:52:00Z": 1,\n    "2026-01-12T17:55:00Z": 2,\n    "2026-01-12T18:02:00Z": 3,\n    "2026-01-12T18:08:00Z": 4,\n    "2026-01-12T18:13:00Z": 5,\n    "2026-01-12T18:17:00Z": 6,\n    "2026-01-12T18:20:00Z": null,\n    "2026-01-12T18:27:00Z": null,\n    "2026-01-12T18:33:00Z": null,\n    "2026-01-12T18:38:00Z": null,\n    "2026-01-12T18:42:00Z": null,\n    "2026-01-12T18:45:00Z": null,\n    "2026-01-12T18:52:00Z": null,\n    "2026-01-12T18:58:00Z": null,\n    "2026-01-12T19:03:00Z": null,\n    "2026-01-12T19:07:00Z": -1,\n    "2026-01-12T19:10:00Z": -1,\n    "2026-01-12T19:17:00Z": -1,\n    "2026-01-12T19:23:00Z": 0,\n    "2026-01-12T19:28:00Z": 1,\n    "2026-01-12T19:32:00Z": 2,\n    "2026-01-12T19:35:00Z": 4,\n    "2026-01-12T19:42:00Z": null,\n    "2026-01-12T19:48:00Z": null,\n    "2026-01-12T19:53:00Z": null,\n    "2026-01-12T19:57:00Z": null,\n    "2026-01-12T20:00:00Z": null,\n    "2026-01-12T20:07:00Z": 5,\n    "2026-01-12T20:13:00Z": 4,\n    "2026-01-12T20:18:00Z": 3,\n    "2026-01-12T20:22:00Z": 1,\n    "2026-01-12T20:25:00Z": 0,\n    "2026-01-12T20:32:00Z": 0,\n    "2026-01-12T20:38:00Z": -1,\n    "2026-01-12T20:43:00Z": -1,\n    "2026-01-12T20:47:00Z": 0,\n    "2026-01-12T20:50:00Z": 1,\n    "2026-01-12T20:57:00Z": 2,\n    "2026-01-12T21:03:00Z": 4,\n    "2026-01-12T21:08:00Z": 5,\n    "2026-01-12T21:12:00Z": 6,\n    "2026-01-12T21:15:00Z": 7,\n    "2026-01-12T21:22:00Z": 8,\n    "2026-01-12T21:28:00Z": 8,\n    "2026-01-12T21:33:00Z": 8,\n    "2026-01-12T21:37:00Z": 7,\n    "2026-01-12T21:40:00Z": null,\n    "2026-01-12T21:47:00Z": null,\n    "2026-01-12T21:53:00Z": null,\n    "2026-01-12T21:58:00Z": null,\n    "2026-01-12T22:02:00Z": null,\n    "2026-01-12T22:05:00Z": null,\n    "2026-01-12T22:12:00Z": null,\n    "2026-01-12T22:18:00Z": 2,\n    "2026-01-12T22:23:00Z": 3,\n    "2026-01-12T22:27:00Z": 5,\n    "2026-01-12T22:30:00Z": 6,\n    "2026-01-12T22:37:00Z": 8,\n    "2026-01-12T22:43:00Z": 9,\n    "2026-01-12T22:48:00Z": 10,\n    "2026-01-12T22:52:00Z": 11,\n    "2026-01-12T22:55:00Z": null,\n    "2026-01-12T23:02:00Z": null,\n    "2026-01-12T23:08:00Z": null,\n    "2026-01-12T23:13:00Z": null,\n    "2026-01-12T23:17:00Z": null,\n    "2026-01-12T23:20:00Z": null,\n    "2026-01-12T23:27:00Z": null,\n    "2026-01-12T23:33:00Z": null,\n    "2026-01-12T23:38:00Z": null,\n    "2026-01-12T23:42:00Z": null,\n    "2026-01-12T23:45:00Z": 6,\n    "2026-01-12T23:52:00Z": 7,\n    "2026-01-12T23:58:00Z": 8\n}'
            },
        },
        {"workflow_input_name": "mode", "filters": {"value": "fill"}},
        {"workflow_input_name": "method", "filters": {"value": "time"}},
        {"workflow_input_name": "limit_direction", "filters": {"value": "both"}},
        {
            "workflow_input_name": "auto_frequency_determination",
            "filters": {"value": "False"},
        },
        {"workflow_input_name": "resample_to", "filters": {"value": "5min"}},
        {"workflow_input_name": "max_gap_length", "filters": {"value": "4"}},
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{\n    "2026-01-12T00:00:00Z": 10,\n    "2026-01-12T00:07:00Z": 12,\n    "2026-01-12T00:13:00Z": 13,\n    "2026-01-12T00:18:00Z": 14,\n    "2026-01-12T00:22:00Z": 15,\n    "2026-01-12T00:25:00Z": 15,\n    "2026-01-12T00:32:00Z": 15,\n    "2026-01-12T00:38:00Z": 14,\n    "2026-01-12T00:43:00Z": 13,\n    "2026-01-12T00:47:00Z": 12,\n    "2026-01-12T00:50:00Z": 11,\n    "2026-01-12T00:57:00Z": 10,\n    "2026-01-12T01:03:00Z": null,\n    "2026-01-12T01:08:00Z": 9,\n    "2026-01-12T01:12:00Z": 9,\n    "2026-01-12T01:15:00Z": 10,\n    "2026-01-12T01:22:00Z": 11,\n    "2026-01-12T01:28:00Z": 12,\n    "2026-01-12T01:33:00Z": 14,\n    "2026-01-12T01:37:00Z": 15,\n    "2026-01-12T01:40:00Z": 17,\n    "2026-01-12T01:47:00Z": 18,\n    "2026-01-12T01:53:00Z": 18,\n    "2026-01-12T01:58:00Z": 19,\n    "2026-01-12T02:02:00Z": 18,\n    "2026-01-12T02:05:00Z": 18,\n    "2026-01-12T02:12:00Z": 16,\n    "2026-01-12T02:18:00Z": 15,\n    "2026-01-12T02:23:00Z": 14,\n    "2026-01-12T02:27:00Z": 13,\n    "2026-01-12T02:30:00Z": null,\n    "2026-01-12T02:37:00Z": null,\n    "2026-01-12T02:43:00Z": null,\n    "2026-01-12T02:48:00Z": 13,\n    "2026-01-12T02:52:00Z": 14,\n    "2026-01-12T02:55:00Z": 15,\n    "2026-01-12T03:02:00Z": 16,\n    "2026-01-12T03:08:00Z": 18,\n    "2026-01-12T03:13:00Z": 19,\n    "2026-01-12T03:17:00Z": 20,\n    "2026-01-12T03:20:00Z": 21,\n    "2026-01-12T03:27:00Z": 21,\n    "2026-01-12T03:33:00Z": 20,\n    "2026-01-12T03:38:00Z": 20,\n    "2026-01-12T03:42:00Z": 19,\n    "2026-01-12T03:45:00Z": 17,\n    "2026-01-12T03:52:00Z": 16,\n    "2026-01-12T03:58:00Z": 15,\n    "2026-01-12T04:03:00Z": 14,\n    "2026-01-12T04:07:00Z": 13,\n    "2026-01-12T04:10:00Z": 13,\n    "2026-01-12T04:17:00Z": 14,\n    "2026-01-12T04:23:00Z": 15,\n    "2026-01-12T04:28:00Z": 16,\n    "2026-01-12T04:32:00Z": null,\n    "2026-01-12T04:35:00Z": null,\n    "2026-01-12T04:42:00Z": null,\n    "2026-01-12T04:48:00Z": null,\n    "2026-01-12T04:53:00Z": null,\n    "2026-01-12T04:57:00Z": null,\n    "2026-01-12T05:00:00Z": 21,\n    "2026-01-12T05:07:00Z": 20,\n    "2026-01-12T05:13:00Z": 19,\n    "2026-01-12T05:18:00Z": 17,\n    "2026-01-12T05:22:00Z": 16,\n    "2026-01-12T05:25:00Z": 15,\n    "2026-01-12T05:32:00Z": 14,\n    "2026-01-12T05:38:00Z": 13,\n    "2026-01-12T05:43:00Z": 13,\n    "2026-01-12T05:47:00Z": 14,\n    "2026-01-12T05:50:00Z": 15,\n    "2026-01-12T05:57:00Z": 16,\n    "2026-01-12T06:03:00Z": 17,\n    "2026-01-12T06:08:00Z": 18,\n    "2026-01-12T06:12:00Z": 19,\n    "2026-01-12T06:15:00Z": 20,\n    "2026-01-12T06:22:00Z": 21,\n    "2026-01-12T06:28:00Z": 21,\n    "2026-01-12T06:33:00Z": 20,\n    "2026-01-12T06:37:00Z": 19,\n    "2026-01-12T06:40:00Z": 18,\n    "2026-01-12T06:47:00Z": 16,\n    "2026-01-12T06:53:00Z": 15,\n    "2026-01-12T06:58:00Z": 14,\n    "2026-01-12T07:02:00Z": 13,\n    "2026-01-12T07:05:00Z": 12,\n    "2026-01-12T07:12:00Z": 12,\n    "2026-01-12T07:18:00Z": 12,\n    "2026-01-12T07:23:00Z": 13,\n    "2026-01-12T07:27:00Z": 14,\n    "2026-01-12T07:30:00Z": 15,\n    "2026-01-12T07:37:00Z": 17,\n    "2026-01-12T07:43:00Z": 18,\n    "2026-01-12T07:48:00Z": 19,\n    "2026-01-12T07:52:00Z": 19,\n    "2026-01-12T07:55:00Z": 19,\n    "2026-01-12T08:02:00Z": null,\n    "2026-01-12T08:08:00Z": null,\n    "2026-01-12T08:13:00Z": null,\n    "2026-01-12T08:17:00Z": null,\n    "2026-01-12T08:20:00Z": null,\n    "2026-01-12T08:27:00Z": null,\n    "2026-01-12T08:33:00Z": null,\n    "2026-01-12T08:38:00Z": null,\n    "2026-01-12T08:42:00Z": null,\n    "2026-01-12T08:45:00Z": null,\n    "2026-01-12T08:52:00Z": null,\n    "2026-01-12T08:58:00Z": null,\n    "2026-01-12T09:03:00Z": null,\n    "2026-01-12T09:07:00Z": null,\n    "2026-01-12T09:10:00Z": null,\n    "2026-01-12T09:17:00Z": null,\n    "2026-01-12T09:23:00Z": null,\n    "2026-01-12T09:28:00Z": null,\n    "2026-01-12T09:32:00Z": null,\n    "2026-01-12T09:35:00Z": null,\n    "2026-01-12T09:42:00Z": null,\n    "2026-01-12T09:48:00Z": null,\n    "2026-01-12T09:53:00Z": null,\n    "2026-01-12T09:57:00Z": null,\n    "2026-01-12T10:00:00Z": null,\n    "2026-01-12T10:07:00Z": null,\n    "2026-01-12T10:13:00Z": null,\n    "2026-01-12T10:18:00Z": null,\n    "2026-01-12T10:22:00Z": null,\n    "2026-01-12T10:25:00Z": null,\n    "2026-01-12T10:32:00Z": 12,\n    "2026-01-12T10:38:00Z": 13,\n    "2026-01-12T10:43:00Z": 14,\n    "2026-01-12T10:47:00Z": 15,\n    "2026-01-12T10:50:00Z": 15,\n    "2026-01-12T10:57:00Z": 15,\n    "2026-01-12T11:03:00Z": 15,\n    "2026-01-12T11:08:00Z": 14,\n    "2026-01-12T11:12:00Z": 12,\n    "2026-01-12T11:15:00Z": 11,\n    "2026-01-12T11:22:00Z": 9,\n    "2026-01-12T11:28:00Z": 8,\n    "2026-01-12T11:33:00Z": 7,\n    "2026-01-12T11:37:00Z": 6,\n    "2026-01-12T11:40:00Z": null,\n    "2026-01-12T11:47:00Z": null,\n    "2026-01-12T11:53:00Z": 8,\n    "2026-01-12T11:58:00Z": 9,\n    "2026-01-12T12:02:00Z": 10,\n    "2026-01-12T12:05:00Z": 11,\n    "2026-01-12T12:12:00Z": 12,\n    "2026-01-12T12:18:00Z": 13,\n    "2026-01-12T12:23:00Z": 14,\n    "2026-01-12T12:27:00Z": 14,\n    "2026-01-12T12:30:00Z": 13,\n    "2026-01-12T12:37:00Z": 12,\n    "2026-01-12T12:43:00Z": 11,\n    "2026-01-12T12:48:00Z": 9,\n    "2026-01-12T12:52:00Z": 8,\n    "2026-01-12T12:55:00Z": 6,\n    "2026-01-12T13:02:00Z": 5,\n    "2026-01-12T13:08:00Z": 5,\n    "2026-01-12T13:13:00Z": 5,\n    "2026-01-12T13:17:00Z": 5,\n    "2026-01-12T13:20:00Z": 6,\n    "2026-01-12T13:27:00Z": 7,\n    "2026-01-12T13:33:00Z": 8,\n    "2026-01-12T13:38:00Z": 10,\n    "2026-01-12T13:42:00Z": 11,\n    "2026-01-12T13:45:00Z": null,\n    "2026-01-12T13:52:00Z": null,\n    "2026-01-12T13:58:00Z": null,\n    "2026-01-12T14:03:00Z": null,\n    "2026-01-12T14:07:00Z": null,\n    "2026-01-12T14:10:00Z": null,\n    "2026-01-12T14:17:00Z": null,\n    "2026-01-12T14:23:00Z": null,\n    "2026-01-12T14:28:00Z": 5,\n    "2026-01-12T14:32:00Z": 4,\n    "2026-01-12T14:35:00Z": 3,\n    "2026-01-12T14:42:00Z": 3,\n    "2026-01-12T14:48:00Z": 3,\n    "2026-01-12T14:53:00Z": 4,\n    "2026-01-12T14:57:00Z": 5,\n    "2026-01-12T15:00:00Z": 7,\n    "2026-01-12T15:07:00Z": 8,\n    "2026-01-12T15:13:00Z": 9,\n    "2026-01-12T15:18:00Z": 10,\n    "2026-01-12T15:22:00Z": 10,\n    "2026-01-12T15:25:00Z": 10,\n    "2026-01-12T15:32:00Z": 9,\n    "2026-01-12T15:38:00Z": 8,\n    "2026-01-12T15:43:00Z": 7,\n    "2026-01-12T15:47:00Z": 6,\n    "2026-01-12T15:50:00Z": 4,\n    "2026-01-12T15:57:00Z": 3,\n    "2026-01-12T16:03:00Z": 2,\n    "2026-01-12T16:08:00Z": 1,\n    "2026-01-12T16:12:00Z": 1,\n    "2026-01-12T16:15:00Z": 1,\n    "2026-01-12T16:22:00Z": 2,\n    "2026-01-12T16:28:00Z": 3,\n    "2026-01-12T16:33:00Z": 5,\n    "2026-01-12T16:37:00Z": 6,\n    "2026-01-12T16:40:00Z": null,\n    "2026-01-12T16:47:00Z": null,\n    "2026-01-12T16:53:00Z": null,\n    "2026-01-12T16:58:00Z": null,\n    "2026-01-12T17:02:00Z": 7,\n    "2026-01-12T17:05:00Z": 6,\n    "2026-01-12T17:12:00Z": 5,\n    "2026-01-12T17:18:00Z": 4,\n    "2026-01-12T17:23:00Z": 2,\n    "2026-01-12T17:27:00Z": 1,\n    "2026-01-12T17:30:00Z": 0,\n    "2026-01-12T17:37:00Z": -1,\n    "2026-01-12T17:43:00Z": -1,\n    "2026-01-12T17:48:00Z": 0,\n    "2026-01-12T17:52:00Z": 1,\n    "2026-01-12T17:55:00Z": 2,\n    "2026-01-12T18:02:00Z": 3,\n    "2026-01-12T18:08:00Z": 4,\n    "2026-01-12T18:13:00Z": 5,\n    "2026-01-12T18:17:00Z": 6,\n    "2026-01-12T18:20:00Z": null,\n    "2026-01-12T18:27:00Z": null,\n    "2026-01-12T18:33:00Z": null,\n    "2026-01-12T18:38:00Z": null,\n    "2026-01-12T18:42:00Z": null,\n    "2026-01-12T18:45:00Z": null,\n    "2026-01-12T18:52:00Z": null,\n    "2026-01-12T18:58:00Z": null,\n    "2026-01-12T19:03:00Z": null,\n    "2026-01-12T19:07:00Z": -1,\n    "2026-01-12T19:10:00Z": -1,\n    "2026-01-12T19:17:00Z": -1,\n    "2026-01-12T19:23:00Z": 0,\n    "2026-01-12T19:28:00Z": 1,\n    "2026-01-12T19:32:00Z": 2,\n    "2026-01-12T19:35:00Z": 4,\n    "2026-01-12T19:42:00Z": null,\n    "2026-01-12T19:48:00Z": null,\n    "2026-01-12T19:53:00Z": null,\n    "2026-01-12T19:57:00Z": null,\n    "2026-01-12T20:00:00Z": null,\n    "2026-01-12T20:07:00Z": 5,\n    "2026-01-12T20:13:00Z": 4,\n    "2026-01-12T20:18:00Z": 3,\n    "2026-01-12T20:22:00Z": 1,\n    "2026-01-12T20:25:00Z": 0,\n    "2026-01-12T20:32:00Z": 0,\n    "2026-01-12T20:38:00Z": -1,\n    "2026-01-12T20:43:00Z": -1,\n    "2026-01-12T20:47:00Z": 0,\n    "2026-01-12T20:50:00Z": 1,\n    "2026-01-12T20:57:00Z": 2,\n    "2026-01-12T21:03:00Z": 4,\n    "2026-01-12T21:08:00Z": 5,\n    "2026-01-12T21:12:00Z": 6,\n    "2026-01-12T21:15:00Z": 7,\n    "2026-01-12T21:22:00Z": 8,\n    "2026-01-12T21:28:00Z": 8,\n    "2026-01-12T21:33:00Z": 8,\n    "2026-01-12T21:37:00Z": 7,\n    "2026-01-12T21:40:00Z": null,\n    "2026-01-12T21:47:00Z": null,\n    "2026-01-12T21:53:00Z": null,\n    "2026-01-12T21:58:00Z": null,\n    "2026-01-12T22:02:00Z": null,\n    "2026-01-12T22:05:00Z": null,\n    "2026-01-12T22:12:00Z": null,\n    "2026-01-12T22:18:00Z": 2,\n    "2026-01-12T22:23:00Z": 3,\n    "2026-01-12T22:27:00Z": 5,\n    "2026-01-12T22:30:00Z": 6,\n    "2026-01-12T22:37:00Z": 8,\n    "2026-01-12T22:43:00Z": 9,\n    "2026-01-12T22:48:00Z": 10,\n    "2026-01-12T22:52:00Z": 11,\n    "2026-01-12T22:55:00Z": null,\n    "2026-01-12T23:02:00Z": null,\n    "2026-01-12T23:08:00Z": null,\n    "2026-01-12T23:13:00Z": null,\n    "2026-01-12T23:17:00Z": null,\n    "2026-01-12T23:20:00Z": null,\n    "2026-01-12T23:27:00Z": null,\n    "2026-01-12T23:33:00Z": null,\n    "2026-01-12T23:38:00Z": null,\n    "2026-01-12T23:42:00Z": null,\n    "2026-01-12T23:45:00Z": 6,\n    "2026-01-12T23:52:00Z": 7,\n    "2026-01-12T23:58:00Z": 8\n}'
            },
        },
        {"workflow_input_name": "mode", "filters": {"value": "fill"}},
        {"workflow_input_name": "method", "filters": {"value": "time"}},
        {"workflow_input_name": "limit_direction", "filters": {"value": "both"}},
        {
            "workflow_input_name": "auto_frequency_determination",
            "filters": {"value": "False"},
        },
        {"workflow_input_name": "resample_to", "filters": {"value": "5min"}},
        {"workflow_input_name": "max_gap_length", "filters": {"value": "4"}},
    ]
}
