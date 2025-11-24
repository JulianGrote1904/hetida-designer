"""Documentation for Add (DataFrame)

# Add (DataFrame)

## Description
Adds two Pandas DataFrames or a Pandas DataFrame and a scalar number. The result is always a DataFrame.

## Inputs
* **dataframe** (Pandas DataFrame): First summand.
* **dataframe_or_number** (Integer, Float, or Pandas Series): Second summand.
* **missing_treatment** (String, optional, default `keep_nan`): Handling for mismatching timestamps
  when both inputs are DataFrames. `keep_nan` preserves gaps (Pandas default). `fill_zero` fills
  missing values with zero before adding.

## Outputs
* **sum** (Pandas DataFrame): Result of the addition.

## Details
* Works with DataFrame + DataFrame or DataFrame + scalar. Mixing DataFrame and Series is not supported.
* For DataFrame inputs, `missing_treatment` controls how misaligned indices are handled.
* Unsupported input types raise a concise error message.

## Examples
Adding two DataFrames with `missing_treatment="fill_zero"`
```
{
    "dataframe": {
        "first": {
            "2025-11-01T00:00:00.000Z": 1.0,
            "2025-11-02T00:00:00.000Z": 2.0
        },
        "second": {
            "2025-11-01T00:00:00.000Z": 10.0,
            "2025-11-02T00:00:00.000Z": 20.0
        }
    },
    "dataframe_or_number": {
        "first": {
            "2025-11-01T00:00:00.000Z": 0.5,
            "2025-11-03T00:00:00.000Z": 0.7
        },
        "second": {
            "2025-11-01T00:00:00.000Z": 1.0,
            "2025-11-03T00:00:00.000Z": 1.5
        }
    }
    ,
    "missing_treatment": "fill_zero"
}
```
The expected output is
```
{
    "sum": {
        "first": {
            "2025-11-01T00:00:00.000Z": 1.5,
            "2025-11-02T00:00:00.000Z": 2.0,
            "2025-11-03T00:00:00.000Z": 0.7
        },
        "second": {
            "2025-11-01T00:00:00.000Z": 11.0,
            "2025-11-02T00:00:00.000Z": 20.0,
            "2025-11-03T00:00:00.000Z": 1.5
        }
    }
}
```
"""

import pandas as pd

# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "dataframe": {"data_type": "DATAFRAME"},
        "dataframe_or_number": {"data_type": "ANY"},
        "missing_treatment": {"data_type": "STRING", "default_value": "keep_nan"},
    },
    "outputs": {
        "sum": {"data_type": "DATAFRAME"},
    },
    "name": "Add (DataFrame)",
    "category": "Basic Arithmetic",
    "description": "Add a DataFrame with another DataFrame or a scalar",
    "version_tag": "1.0.0",
    "id": "ccc98508-2c95-4ee9-af36-2e1e519736b2",
    "revision_group_id": "ccc98508-2c95-4ee9-af36-2e1e519736b2",
    "state": "DRAFT",
}


def main(*, dataframe, dataframe_or_number, missing_treatment="keep_nan"):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****
    # Normalize dict inputs to DataFrame for convenience
    if isinstance(dataframe, dict):
        dataframe = pd.DataFrame.from_dict(dataframe)
        dataframe.index = pd.to_datetime(dataframe.index)
    if isinstance(dataframe_or_number, dict):
        dataframe_or_number = pd.DataFrame.from_dict(dataframe_or_number)
        dataframe_or_number.index = pd.to_datetime(dataframe_or_number.index)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Input 'dataframe' must be a pandas DataFrame, but got {type(dataframe).__name__}."
        )
    if not (
        isinstance(dataframe_or_number, pd.DataFrame)
        or isinstance(dataframe_or_number, (int, float))
    ):
        raise TypeError(
            "Input 'dataframe_or_number' must be a pandas DataFrame or numeric scalar, "
            f"but got {type(dataframe_or_number).__name__}."
        )

    if missing_treatment not in ("keep_nan", "fill_zero"):
        raise ValueError(
            "missing_treatment must be one of {'keep_nan', 'fill_zero'}, "
            f"but got '{missing_treatment}'."
        )

    if isinstance(dataframe_or_number, pd.DataFrame):
        if missing_treatment == "fill_zero":
            return {"sum": dataframe.add(dataframe_or_number, fill_value=0)}
        return {"sum": dataframe + dataframe_or_number}

    # dataframe_or_number is numeric
    return {"sum": dataframe + dataframe_or_number}


# --- Unit tests -------------------------------------------------------------

try:
    import pytest
except ImportError:
    pass
else:

    def test_dataframe_plus_dataframe_keep_nan():
        left = pd.DataFrame(
            {
                "first": [1.0, 2.0, 3.0],
                "second": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        right = pd.DataFrame(
            {
                "first": [0.5, 0.7],
                "second": [1.0, 1.5],
            },
            index=pd.to_datetime(["2025-11-01T00:00:00", "2025-11-03T00:00:00"]),
        )
        expected = pd.DataFrame(
            {
                "first": [1.5, None, 3.7],
                "second": [11.0, None, 31.5],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_frame_equal(
            main(dataframe=left, dataframe_or_number=right)["sum"], expected
        )

    def test_dataframe_plus_dataframe_fill_zero():
        left = pd.DataFrame(
            {
                "first": [1.0, 2.0, 3.0],
                "second": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        right = pd.DataFrame(
            {
                "first": [0.5, 0.7],
                "second": [1.0, 1.5],
            },
            index=pd.to_datetime(["2025-11-01T00:00:00", "2025-11-03T00:00:00"]),
        )
        expected = pd.DataFrame(
            {
                "first": [1.5, 2.0, 3.7],
                "second": [11.0, 20.0, 31.5],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_frame_equal(
            main(
                dataframe=left, dataframe_or_number=right, missing_treatment="fill_zero"
            )["sum"],
            expected,
        )

    def test_dataframe_plus_scalar():
        left = pd.DataFrame(
            {
                "first": [1.0, 2.0, 3.0],
                "second": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        result = main(dataframe=left, dataframe_or_number=2)["sum"]
        expected = pd.DataFrame(
            {
                "first": [3.0, 4.0, 5.0],
                "second": [12.0, 22.0, 32.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
            dtype="float64",
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_dataframe_or_number_type_raises():
        left = pd.DataFrame(
            {
                "first": [1.0, 2.0, 3.0],
                "second": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        with pytest.raises(TypeError):
            main(dataframe=left, dataframe_or_number=[1, 2, 3])

    def test_invalid_missing_treatment_raises():
        left = pd.DataFrame(
            {
                "first": [1.0, 2.0, 3.0],
                "second": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                ["2025-11-01T00:00:00", "2025-11-02T00:00:00", "2025-11-03T00:00:00"]
            ),
        )
        with pytest.raises(ValueError):
            main(dataframe=left, dataframe_or_number=1, missing_treatment="unsupported")


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {
            "workflow_input_name": "dataframe",
            "filters": {
                "value": '{\n    "first": {\n        "2025-11-01T00:00:00.000Z": 1.0,\n        "2025-11-02T00:00:00.000Z": 2.0,\n        "2025-11-03T00:00:00.000Z": 3.0\n    },\n    "second": {\n        "2025-11-01T00:00:00.000Z": 10.0,\n        "2025-11-02T00:00:00.000Z": 20.0,\n        "2025-11-03T00:00:00.000Z": 30.0\n    }\n}'
            },
        },
        {
            "workflow_input_name": "dataframe_or_number",
            "filters": {
                "value": '{\n    "first": {\n        "2025-11-01T00:00:00.000Z": 0.5,\n        "2025-11-03T00:00:00.000Z": 0.7\n    },\n    "second": {\n        "2025-11-01T00:00:00.000Z": 1.0,\n        "2025-11-03T00:00:00.000Z": 1.5\n    }\n}'
            },
        },
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {
            "workflow_input_name": "dataframe",
            "filters": {
                "value": '{\n    "first": {\n        "2025-11-01T00:00:00.000Z": 1.0,\n        "2025-11-02T00:00:00.000Z": 2.0,\n        "2025-11-03T00:00:00.000Z": 3.0\n    },\n    "second": {\n        "2025-11-01T00:00:00.000Z": 10.0,\n        "2025-11-02T00:00:00.000Z": 20.0,\n        "2025-11-03T00:00:00.000Z": 30.0\n    }\n}'
            },
        },
        {
            "workflow_input_name": "dataframe_or_number",
            "filters": {
                "value": '{\n    "first": {\n        "2025-11-01T00:00:00.000Z": 0.5,\n        "2025-11-03T00:00:00.000Z": 0.7\n    },\n    "second": {\n        "2025-11-01T00:00:00.000Z": 1.0,\n        "2025-11-03T00:00:00.000Z": 1.5\n    }\n}'
            },
        },
    ]
}
