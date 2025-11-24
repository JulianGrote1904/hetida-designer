"""Documentation for Add (Series)

# Add (Series)

## Description
Adds two Pandas Series or a Pandas Series and a scalar (int/float). The result is always a Series.

## Inputs
* **series** (Pandas Series): First summand.
* **series_or_number** (Integer, Float, or Pandas Series): Second summand.
* **missing_treatment** (String, optional, default `keep_nan`): Handling for mismatching timestamps
  when both inputs are Series. `keep_nan` preserves gaps (Pandas default). `fill_zero` fills missing
  values with zero before adding.

## Outputs
* **sum** (Pandas Series): Result of the addition.

## Details
* Works with Series + Series or Series + scalar. Mixing Series and DataFrame is not supported.
* For Series inputs, `missing_treatment` controls how misaligned indices are handled.
* Unsupported input types raise a concise error message.

## Examples
Adding two Series
```
{
    "series": {
        "2025-11-01T00:00:00.000Z": 1.0,
        "2025-11-02T00:00:00.000Z": 1.1,
        "2025-11-03T00:00:00.000Z": 1.2
    },
    "series_or_number": {
        "2025-11-01T00:00:00.000Z": 0.5,
        "2025-11-03T00:00:00.000Z": 0.3
    }
}
```
The expected output is
```
{
    "sum": {
        "2025-11-01T00:00:00.000Z": 1.5,
        "2025-11-02T00:00:00.000Z": null,
        "2025-11-03T00:00:00.000Z": 1.5
    }
}
```
"""

import pandas as pd

# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "series": {"data_type": "SERIES"},
        "series_or_number": {"data_type": "ANY"},
        "missing_treatment": {"data_type": "STRING", "default_value": "keep_nan"},
    },
    "outputs": {
        "sum": {"data_type": "SERIES"},
    },
    "name": "Add (Series)",
    "category": "Basic Arithmetic",
    "description": "Add a Series with another Series or a scalar",
    "version_tag": "1.0.0",
    "id": "bf6964e6-0726-44e6-8f80-b961d0629b2f",
    "revision_group_id": "bf6964e6-0726-44e6-8f80-b961d0629b2f",
    "state": "DRAFT",
}


def main(*, series, series_or_number, missing_treatment="keep_nan"):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    # Normalize dict inputs to Series for convenience
    if isinstance(series, dict):
        series = pd.Series(series)
        series.index = pd.to_datetime(series.index)
    if isinstance(series_or_number, dict):
        series_or_number = pd.Series(series_or_number)
        series_or_number.index = pd.to_datetime(series_or_number.index)

    # Type checks
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"Input 'series' must be a pandas Series, but got {type(series).__name__}."
        )
    if not (
        isinstance(series_or_number, pd.Series)
        or isinstance(series_or_number, (int, float))
    ):
        raise TypeError(
            f"Input 'series_or_number' must be a pandas Series or numeric scalar, "
            f"but got {type(series_or_number).__name__}."
        )

    if missing_treatment not in ("keep_nan", "fill_zero"):
        raise ValueError(
            "missing_treatment must be one of {'keep_nan', 'fill_zero'}, "
            f"but got '{missing_treatment}'."
        )

    if isinstance(series_or_number, pd.Series):
        if missing_treatment == "fill_zero":
            return {"sum": series.add(series_or_number, fill_value=0)}
        return {"sum": series + series_or_number}

    # b is numeric
    return {"sum": series + series_or_number}


# --- Unit tests -------------------------------------------------------------

try:
    import pytest
except ImportError:
    pass
else:

    def test_series_plus_series_keep_nan():
        left = pd.Series(
            data=[1.0, 2.0, 3.0],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        right = pd.Series(
            data=[0.5, 1.5],
            index=pd.to_datetime(["2025-11-01T00:00:00", "2025-11-03T00:00:00"]),
        )

        expected = pd.Series(
            data=[1.5, None, 4.5],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_series_equal(
            main(series=left, series_or_number=right)["sum"], expected
        )

    def test_series_plus_series_fill_zero():
        left = pd.Series(
            data=[1.0, 2.0, 3.0],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        right = pd.Series(
            data=[0.5, 1.5],
            index=pd.to_datetime(["2025-11-01T00:00:00", "2025-11-03T00:00:00"]),
        )

        expected = pd.Series(
            data=[1.5, 2.0, 4.5],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_series_equal(
            main(series=left, series_or_number=right, missing_treatment="fill_zero")[
                "sum"
            ],
            expected,
        )

    def test_series_plus_scalar():
        left = pd.Series(
            data=[1.0, 2.0, 3.0],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        result = main(series=left, series_or_number=2)["sum"]
        expected = pd.Series(
            data=[3.0, 4.0, 5.0],
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_b_type_raises():
        left = pd.Series([1.0], index=pd.to_datetime(["2025-11-01T00:00:00"]))
        with pytest.raises(TypeError):
            main(series=left, series_or_number=[1, 2, 3])

    def test_invalid_missing_treatment_raises():
        left = pd.Series([1.0], index=pd.to_datetime(["2025-11-01T00:00:00"]))
        with pytest.raises(ValueError):
            main(series=left, series_or_number=1, missing_treatment="unsupported")


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "series",
            "filters": {
                "value": '{\n    "2025-11-01T00:00:00.000Z": 1.0,\n    "2025-11-02T00:00:00.000Z": 1.1,\n    "2025-11-03T00:00:00.000Z": 1.2,\n    "2025-11-04T00:00:00.000Z": 1.3,\n    "2025-11-05T00:00:00.000Z": 1.4,\n    "2025-11-06T00:00:00.000Z": 1.5,\n    "2025-11-07T00:00:00.000Z": 1.6,\n    "2025-11-08T00:00:00.000Z": 1.7,\n    "2025-11-09T00:00:00.000Z": 1.8,\n    "2025-11-10T00:00:00.000Z": 1.9,\n    "2025-11-11T00:00:00.000Z": 2.0,\n    "2025-11-12T00:00:00.000Z": 2.1,\n    "2025-11-13T00:00:00.000Z": 2.2,\n    "2025-11-14T00:00:00.000Z": 2.3,\n    "2025-11-15T00:00:00.000Z": 2.4,\n    "2025-11-16T00:00:00.000Z": 2.5,\n    "2025-11-17T00:00:00.000Z": 2.6,\n    "2025-11-18T00:00:00.000Z": 2.7,\n    "2025-11-19T00:00:00.000Z": 2.8,\n    "2025-11-20T00:00:00.000Z": 2.9\n}'
            },
        },
        {
            "workflow_input_name": "series_or_number",
            "filters": {
                "value": '{\n    "2025-11-01T00:00:00.000Z": 0.5,\n    "2025-11-03T00:00:00.000Z": 0.3,\n    "2025-11-04T00:00:00.000Z": 0.2,\n    "2025-11-05T00:00:00.000Z": 0.1,\n    "2025-11-06T00:00:00.000Z": 0.0,\n    "2025-11-07T00:00:00.000Z": -0.1,\n    "2025-11-08T00:00:00.000Z": -0.2,\n    "2025-11-09T00:00:00.000Z": -0.3,\n    "2025-11-10T00:00:00.000Z": -0.4,\n    "2025-11-11T00:00:00.000Z": -0.5,\n    "2025-11-12T00:00:00.000Z": -0.4,\n    "2025-11-13T00:00:00.000Z": -0.3,\n    "2025-11-14T00:00:00.000Z": -0.2,\n    "2025-11-15T00:00:00.000Z": -0.1,\n    "2025-11-16T00:00:00.000Z": 0.0,\n    "2025-11-17T00:00:00.000Z": 0.1,\n    "2025-11-18T00:00:00.000Z": 0.2,\n    "2025-11-19T00:00:00.000Z": 0.3,\n    "2025-11-20T00:00:00.000Z": 0.4\n}'
            },
        },
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "series",
            "filters": {
                "value": '{\n    "2025-11-01T00:00:00.000Z": 1.0,\n    "2025-11-02T00:00:00.000Z": 1.1,\n    "2025-11-03T00:00:00.000Z": 1.2,\n    "2025-11-04T00:00:00.000Z": 1.3,\n    "2025-11-05T00:00:00.000Z": 1.4,\n    "2025-11-06T00:00:00.000Z": 1.5,\n    "2025-11-07T00:00:00.000Z": 1.6,\n    "2025-11-08T00:00:00.000Z": 1.7,\n    "2025-11-09T00:00:00.000Z": 1.8,\n    "2025-11-10T00:00:00.000Z": 1.9,\n    "2025-11-11T00:00:00.000Z": 2.0,\n    "2025-11-12T00:00:00.000Z": 2.1,\n    "2025-11-13T00:00:00.000Z": 2.2,\n    "2025-11-14T00:00:00.000Z": 2.3,\n    "2025-11-15T00:00:00.000Z": 2.4,\n    "2025-11-16T00:00:00.000Z": 2.5,\n    "2025-11-17T00:00:00.000Z": 2.6,\n    "2025-11-18T00:00:00.000Z": 2.7,\n    "2025-11-19T00:00:00.000Z": 2.8,\n    "2025-11-20T00:00:00.000Z": 2.9\n}'
            },
        },
        {
            "workflow_input_name": "series_or_number",
            "filters": {
                "value": '{\n    "2025-11-01T00:00:00.000Z": 0.5,\n    "2025-11-03T00:00:00.000Z": 0.3,\n    "2025-11-04T00:00:00.000Z": 0.2,\n    "2025-11-05T00:00:00.000Z": 0.1,\n    "2025-11-06T00:00:00.000Z": 0.0,\n    "2025-11-07T00:00:00.000Z": -0.1,\n    "2025-11-08T00:00:00.000Z": -0.2,\n    "2025-11-09T00:00:00.000Z": -0.3,\n    "2025-11-10T00:00:00.000Z": -0.4,\n    "2025-11-11T00:00:00.000Z": -0.5,\n    "2025-11-12T00:00:00.000Z": -0.4,\n    "2025-11-13T00:00:00.000Z": -0.3,\n    "2025-11-14T00:00:00.000Z": -0.2,\n    "2025-11-15T00:00:00.000Z": -0.1,\n    "2025-11-16T00:00:00.000Z": 0.0,\n    "2025-11-17T00:00:00.000Z": 0.1,\n    "2025-11-18T00:00:00.000Z": 0.2,\n    "2025-11-19T00:00:00.000Z": 0.3,\n    "2025-11-20T00:00:00.000Z": 0.4\n}'
            },
        },
    ]
}
