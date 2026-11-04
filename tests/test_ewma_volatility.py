"""Tests for EWMA volatility estimation and forecasting."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.ewma_volatility import (
    EwmaForecast,
    EwmaVolatilityResult,
    calculate_ewma_volatility,
    calculate_ewma_weights,
    forecast_ewma_volatility,
    main,
    validate_decay_factor,
    validate_forecast_horizon,
    validate_initial_variance,
)


@pytest.mark.parametrize(
    "decay_factor",
    [0.50, 0.94, 0.97],
)
def test_validate_decay_factor_accepts_valid_values(
    decay_factor: float,
) -> None:
    """Valid decay factors should be accepted."""
    assert (
        validate_decay_factor(decay_factor)
        == decay_factor
    )


@pytest.mark.parametrize(
    "decay_factor",
    [0.0, 1.0, -0.10, 1.10],
)
def test_validate_decay_factor_rejects_invalid_values(
    decay_factor: float,
) -> None:
    """Decay factor must lie strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        validate_decay_factor(decay_factor)


@pytest.mark.parametrize(
    "decay_factor",
    ["0.94", True],
)
def test_validate_decay_factor_rejects_non_numeric_values(
    decay_factor: object,
) -> None:
    """Decay factor must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_decay_factor(
            decay_factor  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "initial_variance",
    [0.0001, 0.04, 1.0],
)
def test_validate_initial_variance_accepts_positive_values(
    initial_variance: float,
) -> None:
    """Positive finite variances should be accepted."""
    assert (
        validate_initial_variance(initial_variance)
        == initial_variance
    )


@pytest.mark.parametrize(
    "initial_variance",
    [0.0, -0.01],
)
def test_validate_initial_variance_rejects_non_positive_values(
    initial_variance: float,
) -> None:
    """Initial variance must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_initial_variance(
            initial_variance
        )


@pytest.mark.parametrize(
    "initial_variance",
    [np.nan, np.inf, -np.inf],
)
def test_validate_initial_variance_rejects_non_finite_values(
    initial_variance: float,
) -> None:
    """Initial variance must be finite."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_initial_variance(
            initial_variance
        )


@pytest.mark.parametrize(
    "initial_variance",
    ["0.01", True],
)
def test_validate_initial_variance_rejects_non_numeric_values(
    initial_variance: object,
) -> None:
    """Initial variance must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_initial_variance(
            initial_variance  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "steps",
    [1, 5, 20],
)
def test_validate_forecast_horizon_accepts_positive_values(
    steps: int,
) -> None:
    """Positive forecast horizons should be accepted."""
    assert validate_forecast_horizon(steps) == steps


@pytest.mark.parametrize(
    "steps",
    [0, -1],
)
def test_validate_forecast_horizon_rejects_non_positive_values(
    steps: int,
) -> None:
    """Forecast horizon must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_forecast_horizon(steps)


@pytest.mark.parametrize(
    "steps",
    [5.0, "5", True],
)
def test_validate_forecast_horizon_rejects_non_integer_values(
    steps: object,
) -> None:
    """Forecast horizon must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_forecast_horizon(
            steps  # type: ignore[arg-type]
        )


def test_ewma_weights_known_values() -> None:
    """EWMA weights should follow geometric decay."""
    result = calculate_ewma_weights(
        decay_factor=0.50,
        observations=4,
    )

    np.testing.assert_allclose(
        result,
        np.array(
            [0.50, 0.25, 0.125, 0.0625]
        ),
    )


def test_ewma_weights_are_positive_and_decreasing() -> None:
    """More recent observations should receive more weight."""
    result = calculate_ewma_weights(
        decay_factor=0.94,
        observations=100,
    )

    assert np.all(result > 0.0)
    assert np.all(np.diff(result) < 0.0)
    assert result[0] == pytest.approx(0.06)


def test_ewma_weights_sum_toward_one() -> None:
    """A long sequence of EWMA weights should nearly sum to one."""
    result = calculate_ewma_weights(
        decay_factor=0.94,
        observations=1000,
    )

    assert np.sum(result) == pytest.approx(
        1.0,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    "observations",
    [0, -1],
)
def test_weights_reject_non_positive_observation_count(
    observations: int,
) -> None:
    """Weight count must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_ewma_weights(
            observations=observations
        )


@pytest.mark.parametrize(
    "observations",
    [10.0, True],
)
def test_weights_reject_non_integer_observation_count(
    observations: object,
) -> None:
    """Weight count must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        calculate_ewma_weights(
            observations=observations  # type: ignore[arg-type]
        )


def test_ewma_variance_known_recursion() -> None:
    """Conditional variance should follow the EWMA equation."""
    returns = np.array(
        [0.01, -0.02, 0.03]
    )

    result = calculate_ewma_volatility(
        returns,
        decay_factor=0.50,
        initial_variance=0.0004,
        periods_per_year=12,
    )

    expected_variances = np.array(
        [
            0.0004,
            0.00025,
            0.000325,
        ]
    )

    np.testing.assert_allclose(
        result.conditional_variances,
        expected_variances,
    )


def test_ewma_volatility_is_square_root_of_variance() -> None:
    """Conditional volatility should equal square-root variance."""
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    result = calculate_ewma_volatility(
        returns,
        decay_factor=0.80,
        initial_variance=0.0002,
    )

    np.testing.assert_allclose(
        result.conditional_volatilities,
        np.sqrt(result.conditional_variances),
    )


def test_annualized_ewma_volatility() -> None:
    """Annualized volatility should use square-root-of-time."""
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    result = calculate_ewma_volatility(
        returns,
        decay_factor=0.80,
        initial_variance=0.0002,
        periods_per_year=12,
    )

    np.testing.assert_allclose(
        result.annualized_volatilities,
        result.conditional_volatilities
        * np.sqrt(12),
    )


def test_default_initial_variance_is_sample_variance() -> None:
    """Default initialization should use sample variance."""
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    result = calculate_ewma_volatility(
        returns
    )

    assert result.initial_variance == pytest.approx(
        np.var(returns, ddof=1)
    )


def test_ewma_result_structure() -> None:
    """EWMA should return complete aligned arrays."""
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01, 0.02]
    )

    result = calculate_ewma_volatility(
        returns
    )

    assert isinstance(result, EwmaVolatilityResult)
    assert result.observations == 5
    assert result.decay_factor == 0.94
    assert result.conditional_variances.shape == (5,)
    assert result.conditional_volatilities.shape == (5,)
    assert result.annualized_volatilities.shape == (5,)
    assert np.all(result.conditional_variances > 0.0)


def test_ewma_requires_two_returns() -> None:
    """EWMA estimation needs at least two returns."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        calculate_ewma_volatility([0.01])


def test_ewma_rejects_zero_sample_variance() -> None:
    """Default initialization requires return variation."""
    with pytest.raises(
        ValueError,
        match="positive sample variance",
    ):
        calculate_ewma_volatility(
            np.full(20, 0.01)
        )


def test_custom_initial_variance_allows_constant_returns() -> None:
    """An explicit positive variance can initialize constant data."""
    result = calculate_ewma_volatility(
        np.full(20, 0.01),
        initial_variance=0.0001,
    )

    assert np.all(
        result.conditional_variances > 0.0
    )


def test_ewma_responds_to_large_recent_return() -> None:
    """A large shock should increase next conditional variance."""
    calm_returns = np.full(20, 0.01)
    shocked_returns = calm_returns.copy()
    shocked_returns[-2] = 0.10

    calm_result = calculate_ewma_volatility(
        calm_returns,
        initial_variance=0.0001,
    )
    shocked_result = calculate_ewma_volatility(
        shocked_returns,
        initial_variance=0.0001,
    )

    assert (
        shocked_result.conditional_variances[-1]
        > calm_result.conditional_variances[-1]
    )


def test_lower_decay_reacts_more_to_recent_shock() -> None:
    """Lower lambda should respond more strongly to new data."""
    returns = np.concatenate(
        [
            np.full(20, 0.01),
            np.array([0.10, 0.01]),
        ]
    )

    fast_result = calculate_ewma_volatility(
        returns,
        decay_factor=0.70,
        initial_variance=0.0001,
    )
    slow_result = calculate_ewma_volatility(
        returns,
        decay_factor=0.97,
        initial_variance=0.0001,
    )

    assert (
        fast_result.conditional_variances[-1]
        > slow_result.conditional_variances[-1]
    )


def test_forecast_known_next_variance() -> None:
    """EWMA forecast should include the latest squared return."""
    returns = np.array(
        [0.01, -0.02, 0.03]
    )

    forecast = forecast_ewma_volatility(
        returns,
        steps=3,
        decay_factor=0.50,
        initial_variance=0.0004,
        periods_per_year=12,
    )

    expected_next_variance = (
        0.50 * 0.000325
        + 0.50 * 0.03**2
    )

    np.testing.assert_allclose(
        forecast.variance_forecasts,
        np.full(3, expected_next_variance),
    )


def test_ewma_multistep_forecast_is_constant() -> None:
    """Integrated EWMA variance forecasts should remain constant."""
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    forecast = forecast_ewma_volatility(
        returns,
        steps=20,
    )

    assert isinstance(forecast, EwmaForecast)
    assert forecast.steps == 20
    assert np.allclose(
        forecast.variance_forecasts,
        forecast.variance_forecasts[0],
    )
    assert np.allclose(
        forecast.volatility_forecasts,
        forecast.volatility_forecasts[0],
    )


def test_forecast_volatility_matches_variance() -> None:
    """Forecast volatility should equal square-root variance."""
    forecast = forecast_ewma_volatility(
        [0.01, -0.02, 0.03, -0.01],
        steps=5,
    )

    np.testing.assert_allclose(
        forecast.volatility_forecasts,
        np.sqrt(forecast.variance_forecasts),
    )


def test_forecast_annualization() -> None:
    """Forecast annualization should use the selected frequency."""
    forecast = forecast_ewma_volatility(
        [0.01, -0.02, 0.03, -0.01],
        steps=5,
        periods_per_year=12,
    )

    np.testing.assert_allclose(
        forecast.annualized_volatility_forecasts,
        forecast.volatility_forecasts
        * np.sqrt(12),
    )


def test_ewma_result_is_immutable() -> None:
    """EWMA result attributes should be immutable."""
    result = calculate_ewma_volatility(
        [0.01, -0.02, 0.03]
    )

    with pytest.raises(FrozenInstanceError):
        result.observations = 99  # type: ignore[misc]


def test_ewma_forecast_is_immutable() -> None:
    """EWMA forecast attributes should be immutable."""
    result = forecast_ewma_volatility(
        [0.01, -0.02, 0.03],
        steps=5,
    )

    with pytest.raises(FrozenInstanceError):
        result.steps = 99  # type: ignore[misc]


def test_main_prints_ewma_analysis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The demonstration should print principal EWMA results."""
    main()

    output = capsys.readouterr().out

    assert "EWMA volatility model" in output
    assert "Decay factor" in output
    assert "Latest daily volatility" in output
    assert "annualized volatility" in output.lower()
    assert "EWMA weights" in output
    assert "Volatility forecast" in output