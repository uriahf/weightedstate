import polars as pl
import pytest
from polars.testing import assert_frame_equal
from weightedstate.survival import weighted_aalen_johansen

def test_weighted_aalen_johansen_basic():
    times = pl.Series([1, 2, 3, 4, 5])
    reals = pl.Series([1, 0, 1, 2, 0])
    weights = pl.Series([1.0, 1.0, 1.0, 1.0, 1.0])

    result = weighted_aalen_johansen(times, reals, weights)

    assert isinstance(result, pl.DataFrame)
    assert "cif_1" in result.columns
    assert "cif_2" in result.columns
    assert "overall_survival" in result.columns

def test_weighted_aalen_johansen_km():
    # In a Kaplan-Meier setting, there are no competing events (reals=2)
    times = pl.Series([1, 2, 3, 4, 5])
    reals = pl.Series([1, 0, 1, 1, 0])
    weights = pl.Series([1.0, 1.0, 1.0, 1.0, 1.0])

    result = weighted_aalen_johansen(times, reals, weights)

    # In this case, cif_2 should be all zeros
    assert result["cif_2"].sum() == 0
    # and cif_1 should be the cumulative hazard
    assert result["cif_1"].is_between(0, 1).all()

def test_weighted_aalen_johansen_weights():
    times = pl.Series([1, 2, 3, 4, 5])
    reals = pl.Series([1, 0, 1, 2, 0])
    weights = pl.Series([0.5, 1.5, 0.8, 1.2, 0.9])

    result = weighted_aalen_johansen(times, reals, weights)

    assert isinstance(result, pl.DataFrame)
    assert result["at_risk"].to_list() is not None

def test_weighted_aalen_johansen_empty():
    times = pl.Series([], dtype=pl.Int64)
    reals = pl.Series([], dtype=pl.Int64)
    weights = pl.Series([], dtype=pl.Float64)

    result = weighted_aalen_johansen(times, reals, weights)

    assert result.is_empty()

def test_numerical_correctness_unweighted():
    """
    Test the function's output against a known, manually calculated result.
    """
    times = pl.Series([1, 2, 3])
    reals = pl.Series([1, 2, 0])
    weights = pl.Series([1.0, 1.0, 1.0])

    result = weighted_aalen_johansen(times, reals, weights)

    expected_df = pl.DataFrame({
        "times": [1, 2, 3],
        "overall_survival": [2/3, 1/3, 1/3],
        "cif_1": [1/3, 1/3, 1/3],
        "cif_2": [0.0, 1/3, 1/3],
    })

    # Check that the key columns are close to the expected values
    assert_frame_equal(
        result.select(["times", "overall_survival", "cif_1", "cif_2"]),
        expected_df,
        check_dtype=False,
        atol=1e-6
    )
