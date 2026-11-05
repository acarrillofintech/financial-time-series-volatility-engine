"""Tests for stationarity and correlation diagnostics."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.stationarity import (
    CorrelationDiagnostics,
    StationarityResult,
    calculate_correlation_diagnostics,
    difference_series,
    main,
    run_adf_test,
    validate_lag_count,
    validate_significance_level,
    validate_time_series,
)


def test_validate_time_series_returns_float_array() -> None:
    """Valid input should become a float NumPy array."""
    result = validate_time_series(
        [1, 2, 3],
        minimum_observations=3,
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0, 3.0]),
    )


def test_validate_time_series_rejects_short_sample() -> None:
    """The minimum number of observations should be enforced."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        validate_time_series(
            [1.0, 2.0],
            minimum_observations=3,
        )


def test_validate_time_series_rejects_two_dimensions() -> None:
    """Time-series input must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_time_series(
            np.ones((3, 2)),
            minimum_observations=3,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_validate_time_series_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    """Time series cannot contain NaN or infinite values."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_time_series(
            [1.0, 2.0, invalid_value],
            minimum_observations=3,
        )


@pytest.mark.parametrize(
    "minimum_observations",
    [0, -1],
)
def test_validate_time_series_rejects_invalid_minimum(
    minimum_observations: int,
) -> None:
    """Minimum observations must be positive."""
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        validate_time_series(
            [1.0, 2.0, 3.0],
            minimum_observations=minimum_observations,
        )


@pytest.mark.parametrize(
    "minimum_observations",
    [2.5, True],
)
def test_validate_time_series_rejects_non_integer_minimum(
    minimum_observations: object,
) -> None:
    """Minimum observations must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_time_series(
            [1.0, 2.0, 3.0],
            minimum_observations=minimum_observations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "significance_level",
    [0.01, 0.05, 0.10],
)
def test_validate_significance_level_accepts_valid_values(
    significance_level: float,
) -> None:
    """Common significance levels should be accepted."""
    result = validate_significance_level(
        significance_level
    )

    assert result == significance_level


@pytest.mark.parametrize(
    "significance_level",
    [0.0, 1.0, -0.01, 1.01],
)
def test_validate_significance_level_rejects_invalid_values(
    significance_level: float,
) -> None:
    """Significance must lie strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        validate_significance_level(
            significance_level
        )


@pytest.mark.parametrize(
    "significance_level",
    ["0.05", True],
)
def test_validate_significance_level_rejects_non_numeric_values(
    significance_level: object,
) -> None:
    """Significance must be a real number."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_significance_level(
            significance_level  # type: ignore[arg-type]
        )


def test_validate_lag_count_accepts_valid_lags() -> None:
    """A valid lag count should be returned."""
    result = validate_lag_count(
        10,
        observations=100,
    )

    assert result == 10


@pytest.mark.parametrize(
    "lags",
    [0, -1],
)
def test_validate_lag_count_rejects_non_positive_values(
    lags: int,
) -> None:
    """Lag count must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_lag_count(
            lags,
            observations=100,
        )


@pytest.mark.parametrize(
    "lags",
    [2.5, True],
)
def test_validate_lag_count_rejects_non_integer_values(
    lags: object,
) -> None:
    """Lag count must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_lag_count(
            lags,  # type: ignore[arg-type]
            observations=100,
        )


def test_validate_lag_count_rejects_excessive_lags() -> None:
    """PACF lag count must respect the sample-size limit."""
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        validate_lag_count(
            50,
            observations=100,
        )


def test_validate_lag_count_rejects_very_short_series() -> None:
    """Very short samples cannot support lag diagnostics."""
    with pytest.raises(
        ValueError,
        match="too short",
    ):
        validate_lag_count(
            1,
            observations=3,
        )


def test_first_difference_known_values() -> None:
    """First differences should match consecutive changes."""
    result = difference_series(
        [10.0, 12.0, 11.0, 15.0]
    )

    np.testing.assert_array_equal(
        result,
        np.array([2.0, -1.0, 4.0]),
    )


def test_second_period_difference_known_values() -> None:
    """A two-period difference should compare t with t-2."""
    result = difference_series(
        [10.0, 12.0, 11.0, 15.0],
        periods=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array([1.0, 3.0]),
    )


@pytest.mark.parametrize(
    "periods",
    [0, -1],
)
def test_difference_rejects_non_positive_periods(
    periods: int,
) -> None:
    """Differencing period must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        difference_series(
            [1.0, 2.0, 3.0],
            periods=periods,
        )


@pytest.mark.parametrize(
    "periods",
    [1.5, True],
)
def test_difference_rejects_non_integer_periods(
    periods: object,
) -> None:
    """Differencing period must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        difference_series(
            [1.0, 2.0, 3.0],
            periods=periods,  # type: ignore[arg-type]
        )


def test_difference_rejects_period_equal_to_length() -> None:
    """Differencing period must be shorter than the series."""
    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        difference_series(
            [1.0, 2.0, 3.0],
            periods=3,
        )


def test_adf_identifies_white_noise_as_stationary() -> None:
    """White noise should normally reject the unit-root null."""
    generator = np.random.default_rng(123)
    white_noise = generator.normal(size=1000)

    result = run_adf_test(white_noise)

    assert isinstance(result, StationarityResult)
    assert result.p_value < 0.05
    assert result.is_stationary is True
    assert result.observations > 0
    assert result.used_lags >= 0
    assert set(result.critical_values) == {
        "1%",
        "5%",
        "10%",
    }


def test_adf_identifies_random_walk_as_non_stationary() -> None:
    """A deterministic seeded random walk should retain a unit root."""
    generator = np.random.default_rng(321)
    random_walk = np.cumsum(
        generator.normal(size=1000)
    )

    result = run_adf_test(random_walk)

    assert result.p_value > 0.05
    assert result.is_stationary is False


def test_differencing_random_walk_produces_stationarity() -> None:
    """The first difference of a random walk is white noise."""
    generator = np.random.default_rng(321)
    random_walk = np.cumsum(
        generator.normal(size=1000)
    )

    differenced = difference_series(random_walk)
    result = run_adf_test(differenced)

    assert result.p_value < 0.05
    assert result.is_stationary is True


def test_adf_custom_significance_controls_decision() -> None:
    """The stationarity decision should use the requested level."""
    generator = np.random.default_rng(77)
    series = generator.normal(size=500)

    result = run_adf_test(
        series,
        significance_level=0.01,
    )

    assert result.significance_level == 0.01
    assert result.is_stationary == (
        result.p_value < 0.01
    )


def test_adf_rejects_constant_series() -> None:
    """ADF is undefined for a constant series."""
    with pytest.raises(
        ValueError,
        match="non-constant",
    ):
        run_adf_test(np.ones(100))


def test_adf_rejects_short_series() -> None:
    """ADF requires a sufficient sample."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        run_adf_test(np.arange(9.0))


def test_adf_rejects_invalid_regression() -> None:
    """Only supported deterministic terms should be accepted."""
    generator = np.random.default_rng(10)

    with pytest.raises(
        ValueError,
        match="regression",
    ):
        run_adf_test(
            generator.normal(size=100),
            regression="invalid",
        )


@pytest.mark.parametrize(
    "regression",
    ["c", "ct", "ctt", "n"],
)
def test_adf_accepts_supported_regressions(
    regression: str,
) -> None:
    """All documented ADF regression specifications should run."""
    generator = np.random.default_rng(44)
    series = generator.normal(size=200)

    result = run_adf_test(
        series,
        regression=regression,
    )

    assert isinstance(result, StationarityResult)
    assert np.isfinite(result.adf_statistic)
    assert 0.0 <= result.p_value <= 1.0


def test_correlation_diagnostics_structure() -> None:
    """ACF and PACF should return one value per lag."""
    generator = np.random.default_rng(55)
    series = generator.normal(size=200)

    result = calculate_correlation_diagnostics(
        series,
        lags=12,
    )

    assert isinstance(result, CorrelationDiagnostics)
    assert result.lags.shape == (13,)
    assert result.autocorrelation.shape == (13,)
    assert result.partial_autocorrelation.shape == (13,)
    np.testing.assert_array_equal(
        result.lags,
        np.arange(13),
    )


def test_correlation_at_lag_zero_equals_one() -> None:
    """ACF and PACF at lag zero must equal one."""
    generator = np.random.default_rng(66)
    series = generator.normal(size=200)

    result = calculate_correlation_diagnostics(
        series,
        lags=10,
    )

    assert result.autocorrelation[0] == pytest.approx(1.0)
    assert result.partial_autocorrelation[0] == pytest.approx(1.0)


def test_ar_one_series_has_positive_first_autocorrelation() -> None:
    """A persistent AR(1) process should have positive lag-one ACF."""
    generator = np.random.default_rng(88)
    innovations = generator.normal(size=1000)
    series = np.zeros(1000)

    for index in range(1, series.size):
        series[index] = (
            0.80 * series[index - 1]
            + innovations[index]
        )

    result = calculate_correlation_diagnostics(
        series,
        lags=10,
    )

    assert result.autocorrelation[1] > 0.60
    assert result.partial_autocorrelation[1] > 0.60


def test_correlation_diagnostics_rejects_excessive_lags() -> None:
    """Lag diagnostics should reject unsupported lag counts."""
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        calculate_correlation_diagnostics(
            np.arange(20.0),
            lags=10,
        )


def test_stationarity_result_is_immutable() -> None:
    """ADF result should be an immutable data class."""
    generator = np.random.default_rng(99)
    result = run_adf_test(
        generator.normal(size=100)
    )

    with pytest.raises(FrozenInstanceError):
        result.p_value = 0.50  # type: ignore[misc]


def test_correlation_result_is_immutable() -> None:
    """Correlation result attributes should be immutable."""
    generator = np.random.default_rng(100)
    result = calculate_correlation_diagnostics(
        generator.normal(size=100),
        lags=10,
    )

    with pytest.raises(FrozenInstanceError):
        result.lags = np.array([0])  # type: ignore[misc]


def test_main_prints_stationarity_analysis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The demonstration should print all principal diagnostics."""
    main()

    output = capsys.readouterr().out

    assert "Financial time-series stationarity" in output
    assert "White noise" in output
    assert "Random walk" in output
    assert "Differenced random walk" in output
    assert "Correlation diagnostics" in output