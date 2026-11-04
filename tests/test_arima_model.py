"""Tests for ARIMA modeling and forecasting."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.arima_model import (
    ArimaForecast,
    FittedArimaModel,
    ForecastMetrics,
    calculate_forecast_metrics,
    fit_arima_model,
    forecast_arima_model,
    main,
    simulate_ar_one_process,
    validate_arima_order,
    validate_confidence_level,
    validate_forecast_steps,
)


@pytest.fixture(scope="module")
def ar_one_series() -> np.ndarray:
    """Create a reproducible stationary AR(1) process."""
    return simulate_ar_one_process(
        observations=600,
        coefficient=0.70,
        innovation_scale=1.0,
        seed=123,
    )


@pytest.fixture(scope="module")
def fitted_ar_one_model(
    ar_one_series: np.ndarray,
) -> FittedArimaModel:
    """Fit one reusable ARIMA model for the tests."""
    return fit_arima_model(
        ar_one_series,
        order=(1, 0, 0),
    )


@pytest.mark.parametrize(
    "order",
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 3),
    ],
)
def test_validate_arima_order_accepts_valid_orders(
    order: tuple[int, int, int],
) -> None:
    """Valid ARIMA orders should be accepted."""
    assert validate_arima_order(order) == order


def test_validate_arima_order_requires_tuple() -> None:
    """ARIMA order must be represented by a tuple."""
    with pytest.raises(
        TypeError,
        match="must be a tuple",
    ):
        validate_arima_order(
            [1, 0, 0]  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "order",
    [
        (),
        (1,),
        (1, 0),
        (1, 0, 0, 1),
    ],
)
def test_validate_arima_order_requires_three_values(
    order: tuple[int, ...],
) -> None:
    """An ARIMA order must contain p, d, and q."""
    with pytest.raises(
        ValueError,
        match="exactly three",
    ):
        validate_arima_order(
            order  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "order",
    [
        (-1, 0, 0),
        (1, -1, 0),
        (1, 0, -1),
    ],
)
def test_validate_arima_order_rejects_negative_values(
    order: tuple[int, int, int],
) -> None:
    """ARIMA orders cannot contain negative values."""
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        validate_arima_order(order)


@pytest.mark.parametrize(
    "order",
    [
        (1.0, 0, 0),
        (1, "0", 0),
        (1, 0, True),
    ],
)
def test_validate_arima_order_rejects_non_integer_values(
    order: tuple[object, object, object],
) -> None:
    """Every ARIMA order component must be an integer."""
    with pytest.raises(
        TypeError,
        match="must be integers",
    ):
        validate_arima_order(
            order  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "steps",
    [1, 5, 20],
)
def test_validate_forecast_steps_accepts_positive_values(
    steps: int,
) -> None:
    """Positive forecast horizons should be accepted."""
    assert validate_forecast_steps(steps) == steps


@pytest.mark.parametrize(
    "steps",
    [0, -1],
)
def test_validate_forecast_steps_rejects_non_positive_values(
    steps: int,
) -> None:
    """Forecast horizon must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_forecast_steps(steps)


@pytest.mark.parametrize(
    "steps",
    [5.0, "5", True],
)
def test_validate_forecast_steps_rejects_non_integer_values(
    steps: object,
) -> None:
    """Forecast horizon must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_forecast_steps(
            steps  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.80, 0.95, 0.99],
)
def test_validate_confidence_level_accepts_valid_values(
    confidence_level: float,
) -> None:
    """Common confidence levels should be accepted."""
    result = validate_confidence_level(
        confidence_level
    )

    assert result == confidence_level


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.10, 1.10],
)
def test_validate_confidence_level_rejects_invalid_values(
    confidence_level: float,
) -> None:
    """Confidence level must lie between zero and one."""
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        validate_confidence_level(
            confidence_level
        )


@pytest.mark.parametrize(
    "confidence_level",
    ["0.95", True],
)
def test_validate_confidence_level_rejects_non_numeric_values(
    confidence_level: object,
) -> None:
    """Confidence level must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        validate_confidence_level(
            confidence_level  # type: ignore[arg-type]
        )


def test_simulation_returns_requested_observations() -> None:
    """AR simulation should have the requested length."""
    result = simulate_ar_one_process(
        observations=250
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (250,)
    assert result.dtype == float
    assert np.all(np.isfinite(result))


def test_simulation_is_reproducible() -> None:
    """The same seed should produce the same AR process."""
    first = simulate_ar_one_process(seed=99)
    second = simulate_ar_one_process(seed=99)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_series() -> None:
    """Different seeds should change simulated innovations."""
    first = simulate_ar_one_process(seed=1)
    second = simulate_ar_one_process(seed=2)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "observations",
    [0, 1, -1],
)
def test_simulation_requires_two_observations(
    observations: int,
) -> None:
    """An AR process requires at least two values."""
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        simulate_ar_one_process(
            observations=observations
        )


@pytest.mark.parametrize(
    "observations",
    [10.0, True],
)
def test_simulation_rejects_non_integer_observations(
    observations: object,
) -> None:
    """Observation count must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        simulate_ar_one_process(
            observations=observations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "coefficient",
    [-1.0, 1.0, -1.10, 1.10],
)
def test_simulation_requires_stationary_coefficient(
    coefficient: float,
) -> None:
    """The AR(1) coefficient must have magnitude below one."""
    with pytest.raises(
        ValueError,
        match="strictly between",
    ):
        simulate_ar_one_process(
            coefficient=coefficient
        )


@pytest.mark.parametrize(
    "coefficient",
    ["0.5", True],
)
def test_simulation_rejects_non_numeric_coefficient(
    coefficient: object,
) -> None:
    """The AR coefficient must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        simulate_ar_one_process(
            coefficient=coefficient,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "innovation_scale",
    [0.0, -1.0],
)
def test_simulation_requires_positive_innovation_scale(
    innovation_scale: float,
) -> None:
    """Innovation volatility must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        simulate_ar_one_process(
            innovation_scale=innovation_scale
        )


@pytest.mark.parametrize(
    "innovation_scale",
    ["1.0", True],
)
def test_simulation_rejects_non_numeric_scale(
    innovation_scale: object,
) -> None:
    """Innovation scale must be numeric."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        simulate_ar_one_process(
            innovation_scale=innovation_scale,  # type: ignore[arg-type]
        )


def test_fit_arima_returns_expected_structure(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """ARIMA fitting should return model diagnostics."""
    result = fitted_ar_one_model

    assert isinstance(result, FittedArimaModel)
    assert result.order == (1, 0, 0)
    assert result.observations == 600
    assert result.fitted_values.shape == (600,)
    assert result.residuals.shape == (600,)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert np.isfinite(result.hqic)


def test_fitted_ar_coefficient_is_close_to_simulation_value(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """The fitted AR coefficient should recover the true value."""
    estimated_coefficient = float(
        fitted_ar_one_model.result.arparams[0]
    )

    assert estimated_coefficient == pytest.approx(
        0.70,
        abs=0.10,
    )


def test_fit_arima_rejects_short_series() -> None:
    """ARIMA fitting requires at least twenty observations."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        fit_arima_model(
            np.arange(19.0)
        )


def test_fit_arima_rejects_constant_series() -> None:
    """ARIMA should reject a series without variation."""
    with pytest.raises(
        ValueError,
        match="non-constant",
    ):
        fit_arima_model(
            np.ones(100)
        )


def test_fit_arima_validates_order(
    ar_one_series: np.ndarray,
) -> None:
    """Model fitting should validate the selected order."""
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        fit_arima_model(
            ar_one_series,
            order=(-1, 0, 0),
        )


def test_forecast_returns_requested_horizon(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """Forecast arrays should match the requested horizon."""
    result = forecast_arima_model(
        fitted_ar_one_model,
        steps=12,
    )

    assert isinstance(result, ArimaForecast)
    assert result.steps == 12
    assert result.confidence_level == 0.95
    assert result.predicted_mean.shape == (12,)
    assert result.lower_bound.shape == (12,)
    assert result.upper_bound.shape == (12,)
    assert np.all(np.isfinite(result.predicted_mean))


def test_forecast_intervals_contain_point_forecasts(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """Each point forecast should lie inside its interval."""
    result = forecast_arima_model(
        fitted_ar_one_model,
        steps=20,
    )

    assert np.all(
        result.lower_bound
        <= result.predicted_mean
    )
    assert np.all(
        result.predicted_mean
        <= result.upper_bound
    )
    assert np.all(
        result.upper_bound
        > result.lower_bound
    )


def test_higher_confidence_produces_wider_intervals(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """A 99% interval should be wider than a 90% interval."""
    forecast_90 = forecast_arima_model(
        fitted_ar_one_model,
        steps=10,
        confidence_level=0.90,
    )
    forecast_99 = forecast_arima_model(
        fitted_ar_one_model,
        steps=10,
        confidence_level=0.99,
    )

    width_90 = (
        forecast_90.upper_bound
        - forecast_90.lower_bound
    )
    width_99 = (
        forecast_99.upper_bound
        - forecast_99.lower_bound
    )

    assert np.all(width_99 > width_90)


def test_forecast_rejects_wrong_model_type() -> None:
    """Forecasting requires a fitted model wrapper."""
    with pytest.raises(
        TypeError,
        match="FittedArimaModel",
    ):
        forecast_arima_model(
            "model",  # type: ignore[arg-type]
            steps=5,
        )


def test_calculate_forecast_metrics_known_values() -> None:
    """Forecast metrics should match known calculations."""
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.0, 2.5])

    result = calculate_forecast_metrics(
        actual,
        predicted,
    )

    expected_mae = 1.0 / 3.0
    expected_rmse = np.sqrt(0.5 / 3.0)
    expected_mape = (
        0.50
        + 0.00
        + (0.50 / 3.0)
    ) / 3.0

    assert isinstance(result, ForecastMetrics)
    assert result.observations == 3
    assert result.mean_absolute_error == pytest.approx(
        expected_mae
    )
    assert result.root_mean_squared_error == pytest.approx(
        expected_rmse
    )
    assert (
        result.mean_absolute_percentage_error
        == pytest.approx(expected_mape)
    )


def test_mape_excludes_zero_actual_values() -> None:
    """MAPE should avoid division by zero."""
    result = calculate_forecast_metrics(
        [0.0, 2.0],
        [1.0, 1.0],
    )

    assert (
        result.mean_absolute_percentage_error
        == pytest.approx(0.50)
    )


def test_mape_is_nan_when_all_actual_values_are_zero() -> None:
    """MAPE is undefined when every actual value is zero."""
    result = calculate_forecast_metrics(
        [0.0, 0.0],
        [1.0, -1.0],
    )

    assert np.isnan(
        result.mean_absolute_percentage_error
    )


def test_forecast_metrics_require_matching_lengths() -> None:
    """Actual and predicted arrays must align."""
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_forecast_metrics(
            [1.0, 2.0],
            [1.0],
        )


def test_fitted_model_is_immutable(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """The fitted-model wrapper should be immutable."""
    with pytest.raises(FrozenInstanceError):
        fitted_ar_one_model.aic = 0.0  # type: ignore[misc]


def test_forecast_is_immutable(
    fitted_ar_one_model: FittedArimaModel,
) -> None:
    """The forecast wrapper should be immutable."""
    forecast = forecast_arima_model(
        fitted_ar_one_model,
        steps=5,
    )

    with pytest.raises(FrozenInstanceError):
        forecast.steps = 99  # type: ignore[misc]


def test_main_prints_arima_analysis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The demonstration should print model and forecast results."""
    main()

    output = capsys.readouterr().out

    assert "ARIMA time-series model" in output
    assert "Information criteria" in output
    assert "Forecast evaluation" in output
    assert "First five forecasts" in output
    assert "AIC" in output
    assert "RMSE" in output