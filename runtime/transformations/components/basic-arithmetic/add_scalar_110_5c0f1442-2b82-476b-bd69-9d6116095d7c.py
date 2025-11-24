"""Documentation for Add (Scalar)

# Add (Scalar)

## Description
Adds two numbers. Inputs and output are numeric scalars.

## Inputs
* **x** (Integer or Float): First summand.
* **y** (Integer or Float): Second summand.

## Outputs
* **sum** (Float): Result of the addition.

## Details
* Accepts integers and floats. Unsupported types raise a TypeError.
"""

# ***** DO NOT EDIT LINES BELOW *****
# These lines may be overwritten if component details or inputs/outputs change.
COMPONENT_INFO = {
    "inputs": {
        "x": {"data_type": "FLOAT"},
        "y": {"data_type": "FLOAT"},
    },
    "outputs": {
        "sum": {"data_type": "FLOAT"},
    },
    "name": "Add (Scalar)",
    "category": "Basic Arithmetic",
    "description": "Add two numeric scalars",
    "version_tag": "1.0.0",
    "id": "5c0f1442-2b82-476b-bd69-9d6116095d7c",
    "revision_group_id": "5c0f1442-2b82-476b-bd69-9d6116095d7c",
    "state": "DRAFT",
}


def main(*, x, y):
    # entrypoint function for this component
    # ***** DO NOT EDIT LINES ABOVE *****

    if not isinstance(x, (int, float)):
        raise TypeError(f"Input 'x' must be an int or float, but got {type(x).__name__}.")
    if not isinstance(y, (int, float)):
        raise TypeError(f"Input 'y' must be an int or float, but got {type(y).__name__}.")

    return {"sum": float(x + y)}


# --- Unit tests -------------------------------------------------------------

try:
    import pytest
except ImportError:
    pass
else:

    def test_add_positive_numbers():
        assert main(x=1, y=2)["sum"] == 3.0

    def test_add_negative_and_positive():
        assert main(x=-1.5, y=0.5)["sum"] == -1.0

    def test_invalid_type_raises():
        with pytest.raises(TypeError):
            main(x="a", y=1)


TEST_WIRING_FROM_PY_FILE_IMPORT = {
    "input_wirings": [
        {"workflow_input_name": "x", "filters": {"value": "1.5"}},
        {"workflow_input_name": "y", "filters": {"value": "2.5"}},
    ]
}

RELEASE_WIRING = {
    "input_wirings": [
        {"workflow_input_name": "x", "filters": {"value": "1.5"}},
        {"workflow_input_name": "y", "filters": {"value": "2.5"}},
    ]
}
