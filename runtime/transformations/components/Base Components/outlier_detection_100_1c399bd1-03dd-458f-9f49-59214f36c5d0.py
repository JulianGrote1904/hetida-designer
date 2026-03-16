"""Documentation for Outlier Detection

# Outlier Detection

## Description
Component to detect outliers using a moving-window median absolute deviation
(MAD) band around the local median.

## Inputs
- **timeseries** (Pandas Series):
    The input time series. The index must contain timestamps, and the values
    must be numeric.
- **window_size** (String, default value: "5h"):
    Size of the centered time window around each datapoint. All values inside
    this window are used to estimate the local normal range. Only timedelta
    strings such as `15min`, `1h`, `5h`, or `7D` are supported.
- **min_num_datapoints_in_window** (Integer, default value: 10):
    Minimum number of datapoints required inside a window before the component
    calculates a median and a MAD. If fewer points are available, no outlier
    decision is made for that timestamp.
- **mad_scaling_factor** (Float, default value: 4.4478):
    Multiplier for the calculated MAD. Larger values create a wider accepted
    band and therefore mark fewer points as outliers. The value `1.4836` makes
    the MAD comparable to a standard deviation.
- **min_band_width_factor** (Float, default value: 1.0):
    Safety factor for the minimum band width. It prevents the band from
    becoming unrealistically narrow in very stable phases.
- **direction** (String, default value: "both"):
    Controls which type of outliers should be detected. Use `"both"` for both
    sides, `"high"` for only unusually high values, or `"low"` for only
    unusually low values.

## Outputs
- **band_filter_dataframe** (Pandas DataFrame):
    Data frame containing the original values, the moving band center, the
    rolling deviation, and the outlier mask.
- **outlier_mask** (Pandas Series):
    Boolean series. `True` means the datapoint is treated as an outlier.
    `False` means the datapoint stays inside the accepted band.

## Details
1. The input series is sorted by time and duplicate timestamps are merged by mean.
2. A centered moving median is calculated for each timestamp.
3. A centered moving MAD is calculated for the same windows.
4. The MAD is scaled by `mad_scaling_factor`.
5. A minimum band width is enforced using `min_band_width_factor`.
6. A datapoint is kept if its distance to the local median is smaller than or
   equal to the rolling deviation.
7. The `direction` setting can restrict detection to only unusually high
   values, only unusually low values, or both.
8. If too few datapoints are available in a window, no outlier decision is made
   for that point and the outlier mask is set to `False`.
9. The component returns both the detailed statistics and the boolean mask.

## Example
```json
{
  "timeseries": {
    "2026-03-01T00:00:00Z": 1.058,
    "2026-03-01T00:20:48Z": 0.699,
    "2026-03-01T00:41:37Z": 1.08,
    "2026-03-01T01:02:26Z": 1.054,
    "2026-03-01T01:23:15Z": 0.763,
    "2026-03-01T01:44:04Z": 1.034,
    "2026-03-01T02:04:53Z": 0.965,
    "2026-03-01T02:25:42Z": 0.601,
    "2026-03-01T02:46:31Z": 0.933,
    "2026-03-01T03:07:20Z": 1.081,
    "2026-03-01T03:28:09Z": 2.1,
    "2026-03-01T03:48:58Z": 0.988,
    "2026-03-01T04:09:47Z": 0.741,
    "2026-03-01T04:30:36Z": 0.647,
    "2026-03-01T04:51:25Z": 0.556,
    "2026-03-01T05:12:14Z": 0.453,
    "2026-03-01T05:33:03Z": 1.009,
    "2026-03-01T05:53:52Z": 1.72,
    "2026-03-01T06:14:41Z": 1.002,
    "2026-03-01T06:56:19Z": 0.857,
    "2026-03-01T07:17:08Z": 0.864,
    "2026-03-01T07:37:57Z": 0.606,
    "2026-03-01T07:58:46Z": 0.899,
    "2026-03-01T08:40:24Z": 0.62,
    "2026-03-01T10:03:40Z": 0.721,
    "2026-03-01T10:24:29Z": 1.193,
    "2026-03-01T10:45:18Z": 0.833,
    "2026-03-01T11:06:07Z": 2.06,
    "2026-03-01T11:26:56Z": 0.68,
    "2026-03-01T11:47:45Z": 1.136,
    "2026-03-01T12:08:34Z": 0.62,
    "2026-03-01T12:29:23Z": 0.946,
    "2026-03-01T12:50:12Z": 0.746,
    "2026-03-01T13:11:01Z": 0.833,
    "2026-03-01T13:31:50Z": 0.857,
    "2026-03-01T14:13:28Z": 0.947,
    "2026-03-01T14:34:17Z": 0.841,
    "2026-03-01T14:55:06Z": 0.668,
    "2026-03-01T15:15:55Z": 0.675,
    "2026-03-01T15:36:44Z": 0.84,
    "2026-03-01T15:57:33Z": 0.821,
    "2026-03-01T16:18:22Z": 0.625,
    "2026-03-01T16:39:11Z": 1.155,
    "2026-03-01T17:00:00Z": 0.968
  }
}
```
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

from hdutils import ComponentInputValidationException, parse_default_value


LARGE_SERIES_NUMBA_THRESHOLD = 10000


def parse_fixed_window_size(window_size: str) -> str:
    try:
        offset = pd.tseries.frequencies.to_offset(window_size)
    except ValueError as exc:
        raise ComponentInputValidationException(
            "window_size must be a valid timedelta string like '15min', '1h', or '7D'",
            error_code="422",
            invalid_component_inputs=["window_size"],
        ) from exc

    try:
        _ = offset.nanos
    except ValueError as exc:
        raise ComponentInputValidationException(
            "window_size must be a fixed timedelta string like '15min', '1h', or '7D'",
            error_code="422",
            invalid_component_inputs=["window_size"],
        ) from exc

    return window_size


def validate_inputs(
    timeseries: pd.Series,
    window_size: str,
    min_num_datapoints_in_window: int,
    mad_scaling_factor: float,
    min_band_width_factor: float,
    direction: str,
) -> str:
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
    if not pd.api.types.is_datetime64_any_dtype(timeseries.index):
        raise ComponentInputValidationException(
            "timeseries index must be datetime",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if not pd.api.types.is_numeric_dtype(timeseries):
        raise ComponentInputValidationException(
            "timeseries values must be numeric",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if not np.isfinite(timeseries.dropna().to_numpy(dtype=float)).all():
        raise ComponentInputValidationException(
            "timeseries must not contain inf or -inf values",
            error_code="422",
            invalid_component_inputs=["timeseries"],
        )
    if not isinstance(window_size, str):
        raise ComponentInputValidationException(
            "window_size must be a pandas time window string",
            error_code="422",
            invalid_component_inputs=["window_size"],
        )
    normalized_window_size = parse_fixed_window_size(window_size)
    if (
        not isinstance(min_num_datapoints_in_window, int)
        or min_num_datapoints_in_window < 1
    ):
        raise ComponentInputValidationException(
            "min_num_datapoints_in_window must be an integer >= 1",
            error_code="422",
            invalid_component_inputs=["min_num_datapoints_in_window"],
        )
    for value, input_name in (
        (mad_scaling_factor, "mad_scaling_factor"),
        (min_band_width_factor, "min_band_width_factor"),
    ):
        if not isinstance(value, (int, float)):
            raise ComponentInputValidationException(
                f"{input_name} must be a finite number",
                error_code="422",
                invalid_component_inputs=[input_name],
            )
        if not np.isfinite(float(value)):
            raise ComponentInputValidationException(
                f"{input_name} must be a finite number",
                error_code="422",
                invalid_component_inputs=[input_name],
            )
    if float(mad_scaling_factor) <= 0:
        raise ComponentInputValidationException(
            "mad_scaling_factor must be > 0",
            error_code="422",
            invalid_component_inputs=["mad_scaling_factor"],
        )
    if float(min_band_width_factor) < 0:
        raise ComponentInputValidationException(
            "min_band_width_factor must be >= 0",
            error_code="422",
            invalid_component_inputs=["min_band_width_factor"],
        )
    if direction not in {"both", "high", "low"}:
        raise ComponentInputValidationException(
            "direction must be one of 'both', 'high', 'low'",
            error_code="422",
            invalid_component_inputs=["direction"],
        )
    return normalized_window_size


def prepare_series(timeseries: pd.Series) -> pd.Series:
    prepared = timeseries.sort_index()
    if not prepared.index.is_unique:
        prepared = prepared.groupby(level=0).mean()
    return prepared


def median_absolute_deviation(data: npt.ArrayLike) -> np.float64:
    return np.median(np.abs(data - np.median(data)))


@njit
def median_absolute_deviation_numba(data: npt.ArrayLike) -> np.float64:
    return np.median(np.abs(data - np.median(data)))


def calculate_rolling_median_absolute_deviation(
    series: pd.Series,
    mad_scaling_factor: float,
    window_size: str,
    min_num_datapoints_in_window: int,
) -> pd.Series:
    rolling = series.rolling(
        window=window_size,
        min_periods=min_num_datapoints_in_window,
        center=True,
    )
    if len(series) < LARGE_SERIES_NUMBA_THRESHOLD:
        return mad_scaling_factor * rolling.apply(median_absolute_deviation, raw=True)
    return mad_scaling_factor * rolling.apply(
        median_absolute_deviation_numba,
        raw=True,
        engine="numba",
    )


def calculate_band_filter_statistics(
    series: pd.Series,
    window_size: str,
    min_num_datapoints_in_window: int,
    mad_scaling_factor: float,
    min_band_width_factor: float,
    direction: str,
) -> tuple[pd.DataFrame, pd.Series]:
    band_filter_dataframe = series.to_frame(name="values")

    band_filter_dataframe["band_center"] = series.rolling(
        window=window_size,
        min_periods=min_num_datapoints_in_window,
        center=True,
    ).median()

    band_filter_dataframe["rolling_deviation"] = (
        calculate_rolling_median_absolute_deviation(
            series=series,
            mad_scaling_factor=mad_scaling_factor,
            window_size=window_size,
            min_num_datapoints_in_window=min_num_datapoints_in_window,
        )
    )

    min_width = (
        np.median(band_filter_dataframe["rolling_deviation"].dropna())
        * min_band_width_factor
    )
    band_filter_dataframe.loc[
        band_filter_dataframe["rolling_deviation"] < min_width,
        "rolling_deviation",
    ] = min_width

    deviation_from_center = series - band_filter_dataframe["band_center"]
    if direction == "both":
        band_filter_dataframe["outlier_mask"] = (
            np.abs(deviation_from_center) > band_filter_dataframe["rolling_deviation"]
        )
    elif direction == "high":
        band_filter_dataframe["outlier_mask"] = (
            deviation_from_center > band_filter_dataframe["rolling_deviation"]
        )
    else:
        band_filter_dataframe["outlier_mask"] = (
            deviation_from_center < -band_filter_dataframe["rolling_deviation"]
        )

    band_filter_dataframe.loc[
        band_filter_dataframe["band_center"].isna(),
        "outlier_mask",
    ] = False

    return band_filter_dataframe, band_filter_dataframe["outlier_mask"]


# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "timeseries": {"data_type": "SERIES"},
        "window_size": {"data_type": "STRING", "default_value": "5h"},
        "min_num_datapoints_in_window": {"data_type": "INT", "default_value": 10},
        "mad_scaling_factor": {"data_type": "FLOAT", "default_value": 4.4478},
        "min_band_width_factor": {"data_type": "FLOAT", "default_value": 1.0},
        "direction": {"data_type": "STRING", "default_value": "both"},
    },
    "outputs": {
        "band_filter_dataframe": {"data_type": "DATAFRAME"},
        "outlier_mask": {"data_type": "SERIES"},
    },
    "name": "Outlier Detection",
    "category": "Base Components",
    "description": "Detect outliers using a moving-window MAD band filter.",
    "version_tag": "1.0.0",
    "id": "95351956-8ce8-418c-b899-a655ded13cf6",
    "revision_group_id": "1c399bd1-03dd-458f-9f49-59214f36c5d0",
    "state": "DRAFT",
}


def main(
    *,
    timeseries,
    window_size=parse_default_value(COMPONENT_INFO, "window_size"),
    min_num_datapoints_in_window=parse_default_value(
        COMPONENT_INFO, "min_num_datapoints_in_window"
    ),
    mad_scaling_factor=parse_default_value(COMPONENT_INFO, "mad_scaling_factor"),
    min_band_width_factor=parse_default_value(COMPONENT_INFO, "min_band_width_factor"),
    direction=parse_default_value(COMPONENT_INFO, "direction"),
):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    # Step 1: Validate the user inputs.
    window_size = validate_inputs(
        timeseries,
        window_size,
        min_num_datapoints_in_window,
        mad_scaling_factor,
        min_band_width_factor,
        direction,
    )

    # Step 2: Sort the input series and merge duplicate timestamps.
    prepared = prepare_series(timeseries)

    # Step 3: Calculate the moving window band statistics and outlier mask.
    band_filter_dataframe, outlier_mask = calculate_band_filter_statistics(
        series=prepared,
        window_size=window_size,
        min_num_datapoints_in_window=min_num_datapoints_in_window,
        mad_scaling_factor=mad_scaling_factor,
        min_band_width_factor=min_band_width_factor,
        direction=direction,
    )

    # Step 4: Return both the detailed statistics and the filter mask.
    return {
        "band_filter_dataframe": band_filter_dataframe,
        "outlier_mask": outlier_mask,
    }


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{\n    "2026-03-01T00:00:00Z": 1.058,\n    "2026-03-01T00:20:48Z": 0.699,\n    "2026-03-01T00:41:37Z": 1.08,\n    "2026-03-01T01:02:26Z": 1.054,\n    "2026-03-01T01:23:15Z": 0.763,\n    "2026-03-01T01:44:04Z": 1.034,\n    "2026-03-01T02:04:53Z": 0.965,\n    "2026-03-01T02:25:42Z": 0.601,\n    "2026-03-01T02:46:31Z": 0.933,\n    "2026-03-01T03:07:20Z": 1.081,\n    "2026-03-01T03:28:09Z": 2.1,\n    "2026-03-01T03:48:58Z": 0.988,\n    "2026-03-01T04:09:47Z": 0.741,\n    "2026-03-01T04:30:36Z": 0.647,\n    "2026-03-01T04:51:25Z": 0.556,\n    "2026-03-01T05:12:14Z": 0.453,\n    "2026-03-01T05:33:03Z": 1.009,\n    "2026-03-01T05:53:52Z": 1.72,\n    "2026-03-01T06:14:41Z": 1.002,\n    "2026-03-01T06:56:19Z": 0.857,\n    "2026-03-01T07:17:08Z": 0.864,\n    "2026-03-01T07:37:57Z": 0.606,\n    "2026-03-01T07:58:46Z": 0.899,\n    "2026-03-01T08:40:24Z": 0.62,\n    "2026-03-01T10:03:40Z": 0.721,\n    "2026-03-01T10:24:29Z": 1.193,\n    "2026-03-01T10:45:18Z": 0.833,\n    "2026-03-01T11:06:07Z": 2.06,\n    "2026-03-01T11:26:56Z": 0.68,\n    "2026-03-01T11:47:45Z": 1.136,\n    "2026-03-01T12:08:34Z": 0.62,\n    "2026-03-01T12:29:23Z": 0.946,\n    "2026-03-01T12:50:12Z": 0.746,\n    "2026-03-01T13:11:01Z": 0.833,\n    "2026-03-01T13:31:50Z": 0.857,\n    "2026-03-01T14:13:28Z": 0.947,\n    "2026-03-01T14:34:17Z": 0.841,\n    "2026-03-01T14:55:06Z": 0.668,\n    "2026-03-01T15:15:55Z": 0.675,\n    "2026-03-01T15:36:44Z": 0.84,\n    "2026-03-01T15:57:33Z": 0.821,\n    "2026-03-01T16:18:22Z": 0.625,\n    "2026-03-01T16:39:11Z": 1.155,\n    "2026-03-01T17:00:00Z": 0.968\n}'
            },
        },
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "timeseries",
            "filters": {
                "value": '{\n    "2026-03-01T00:00:00Z": 1.058,\n    "2026-03-01T00:20:48Z": 0.699,\n    "2026-03-01T00:41:37Z": 1.08,\n    "2026-03-01T01:02:26Z": 1.054,\n    "2026-03-01T01:23:15Z": 0.763,\n    "2026-03-01T01:44:04Z": 1.034,\n    "2026-03-01T02:04:53Z": 0.965,\n    "2026-03-01T02:25:42Z": 0.601,\n    "2026-03-01T02:46:31Z": 0.933,\n    "2026-03-01T03:07:20Z": 1.081,\n    "2026-03-01T03:28:09Z": 2.1,\n    "2026-03-01T03:48:58Z": 0.988,\n    "2026-03-01T04:09:47Z": 0.741,\n    "2026-03-01T04:30:36Z": 0.647,\n    "2026-03-01T04:51:25Z": 0.556,\n    "2026-03-01T05:12:14Z": 0.453,\n    "2026-03-01T05:33:03Z": 1.009,\n    "2026-03-01T05:53:52Z": 1.72,\n    "2026-03-01T06:14:41Z": 1.002,\n    "2026-03-01T06:56:19Z": 0.857,\n    "2026-03-01T07:17:08Z": 0.864,\n    "2026-03-01T07:37:57Z": 0.606,\n    "2026-03-01T07:58:46Z": 0.899,\n    "2026-03-01T08:40:24Z": 0.62,\n    "2026-03-01T10:03:40Z": 0.721,\n    "2026-03-01T10:24:29Z": 1.193,\n    "2026-03-01T10:45:18Z": 0.833,\n    "2026-03-01T11:06:07Z": 2.06,\n    "2026-03-01T11:26:56Z": 0.68,\n    "2026-03-01T11:47:45Z": 1.136,\n    "2026-03-01T12:08:34Z": 0.62,\n    "2026-03-01T12:29:23Z": 0.946,\n    "2026-03-01T12:50:12Z": 0.746,\n    "2026-03-01T13:11:01Z": 0.833,\n    "2026-03-01T13:31:50Z": 0.857,\n    "2026-03-01T14:13:28Z": 0.947,\n    "2026-03-01T14:34:17Z": 0.841,\n    "2026-03-01T14:55:06Z": 0.668,\n    "2026-03-01T15:15:55Z": 0.675,\n    "2026-03-01T15:36:44Z": 0.84,\n    "2026-03-01T15:57:33Z": 0.821,\n    "2026-03-01T16:18:22Z": 0.625,\n    "2026-03-01T16:39:11Z": 1.155,\n    "2026-03-01T17:00:00Z": 0.968\n}'
            },
        },
    ]
}
