"""ARIMA modeling, forecasting, and forecast evaluation."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterable

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from src.stationarity import validate_time_series


@dataclass(frozen=True)
class ForecastMetrics:
    """Point-forecast error metrics."""

    observations: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float


@dataclass(frozen=True)
class ArimaForecast:
    """ARIMA point forecasts and confidence intervals."""

    steps: int
    confidence_level: float
    predicted_mean: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray


@dataclass(frozen=True)
class FittedArimaModel:
    """Summary and underlying result of a fitted ARIMA model."""

    order: tuple[int, int, int]
    observations: int
    aic: float
    bic: float
    hqic: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    result: Any = field(
        repr=False,
        compare=False,
    )


def validate_arima_order(
    order: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Validate an ARIMA order expressed as (p, d, q)."""
    if not isinstance(order, tuple):
        raise TypeError(
            "order must be a tuple containing (p, d, q)."
        )

    if len(order) != 3:
        raise ValueError(
            "order must contain exactly three values."
        )

    normalized_order: list[int] = []

    for value in order:
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
        ):
            raise TypeError(
                "ARIMA order values must be integers."
            )

        if value < 0:
            raise ValueError(
                "ARIMA order values cannot be negative."
            )

        normalized_order.append(int(value))

    return (
        normalized_order[0],
        normalized_order[1],
        normalized_order[2],
    )


def validate_forecast_steps(steps: int) -> int:
    """Validate the number of forecast periods."""
    if (
        not isinstance(steps, Integral)
        or isinstance(steps, bool)
    ):
        raise TypeError("steps must be an integer.")

    normalized_steps = int(steps)

    if normalized_steps <= 0:
        raise ValueError(
            "steps must be greater than zero."
        )

    return normalized_steps


def validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate a forecast confidence level."""
    if (
        not isinstance(confidence_level, Real)
        or isinstance(confidence_level, bool)
    ):
        raise TypeError(
            "confidence_level must be a real number."
        )

    normalized_level = float(confidence_level)

    if not 0.0 < normalized_level < 1.0:
        raise ValueError(
            "confidence_level must be between zero and one."
        )

    return normalized_level


def fit_arima_model(
    values: Iterable[float] | np.ndarray,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    trend: str | None = None,
) -> FittedArimaModel:
    """Fit an ARIMA(p, d, q) model to a univariate series."""
    series = validate_time_series(
        values,
        minimum_observations=20,
    )
    normalized_order = validate_arima_order(order)

    if np.isclose(np.std(series), 0.0):
        raise ValueError(
            "ARIMA requires a non-constant time series."
        )

    model = ARIMA(
        series,
        order=normalized_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_result = model.fit()

    fitted_values = np.asarray(
        fitted_result.fittedvalues,
        dtype=float,
    )
    residuals = np.asarray(
        fitted_result.resid,
        dtype=float,
    )

    if not np.all(np.isfinite(fitted_values)):
        raise RuntimeError(
            "ARIMA produced non-finite fitted values."
        )

    if not np.all(np.isfinite(residuals)):
        raise RuntimeError(
            "ARIMA produced non-finite residuals."
        )

    return FittedArimaModel(
        order=normalized_order,
        observations=int(series.size),
        aic=float(fitted_result.aic),
        bic=float(fitted_result.bic),
        hqic=float(fitted_result.hqic),
        fitted_values=fitted_values,
        residuals=residuals,
        result=fitted_result,
    )


def forecast_arima_model(
    fitted_model: FittedArimaModel,
    *,
    steps: int,
    confidence_level: float = 0.95,
) -> ArimaForecast:
    """Produce ARIMA forecasts and confidence intervals."""
    if not isinstance(
        fitted_model,
        FittedArimaModel,
    ):
        raise TypeError(
            "fitted_model must be a FittedArimaModel."
        )

    normalized_steps = validate_forecast_steps(steps)
    normalized_level = validate_confidence_level(
        confidence_level
    )

    forecast_result = fitted_model.result.get_forecast(
        steps=normalized_steps
    )

    predicted_mean = np.asarray(
        forecast_result.predicted_mean,
        dtype=float,
    )

    confidence_intervals = np.asarray(
        forecast_result.conf_int(
            alpha=1.0 - normalized_level
        ),
        dtype=float,
    )

    lower_bound = confidence_intervals[:, 0]
    upper_bound = confidence_intervals[:, 1]

    if not np.all(np.isfinite(predicted_mean)):
        raise RuntimeError(
            "ARIMA produced non-finite forecasts."
        )

    if not np.all(np.isfinite(lower_bound)):
        raise RuntimeError(
            "ARIMA produced non-finite lower bounds."
        )

    if not np.all(np.isfinite(upper_bound)):
        raise RuntimeError(
            "ARIMA produced non-finite upper bounds."
        )

    return ArimaForecast(
        steps=normalized_steps,
        confidence_level=normalized_level,
        predicted_mean=predicted_mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def calculate_forecast_metrics(
    actual_values: Iterable[float] | np.ndarray,
    predicted_values: Iterable[float] | np.ndarray,
) -> ForecastMetrics:
    """Calculate MAE, RMSE, and MAPE for point forecasts."""
    actual = validate_time_series(
        actual_values,
        minimum_observations=1,
    )
    predicted = validate_time_series(
        predicted_values,
        minimum_observations=1,
    )

    if actual.size != predicted.size:
        raise ValueError(
            "actual and predicted values must have "
            "the same length."
        )

    errors = actual - predicted

    mean_absolute_error = float(
        np.mean(np.abs(errors))
    )
    root_mean_squared_error = float(
        np.sqrt(np.mean(errors**2))
    )

    nonzero_mask = ~np.isclose(actual, 0.0)

    if not np.any(nonzero_mask):
        mean_absolute_percentage_error = float("nan")
    else:
        percentage_errors = np.abs(
            errors[nonzero_mask]
            / actual[nonzero_mask]
        )
        mean_absolute_percentage_error = float(
            np.mean(percentage_errors)
        )

    return ForecastMetrics(
        observations=int(actual.size),
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=root_mean_squared_error,
        mean_absolute_percentage_error=(
            mean_absolute_percentage_error
        ),
    )


def simulate_ar_one_process(
    *,
    observations: int = 500,
    coefficient: float = 0.75,
    innovation_scale: float = 1.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Simulate a stationary AR(1) process."""
    if (
        not isinstance(observations, Integral)
        or isinstance(observations, bool)
    ):
        raise TypeError(
            "observations must be an integer."
        )

    if observations < 2:
        raise ValueError(
            "observations must be at least two."
        )

    if (
        not isinstance(coefficient, Real)
        or isinstance(coefficient, bool)
    ):
        raise TypeError(
            "coefficient must be a real number."
        )

    normalized_coefficient = float(coefficient)

    if abs(normalized_coefficient) >= 1.0:
        raise ValueError(
            "coefficient must be strictly between -1 and 1."
        )

    if (
        not isinstance(innovation_scale, Real)
        or isinstance(innovation_scale, bool)
    ):
        raise TypeError(
            "innovation_scale must be a real number."
        )

    normalized_scale = float(innovation_scale)

    if normalized_scale <= 0.0:
        raise ValueError(
            "innovation_scale must be greater than zero."
        )

    random_generator = np.random.default_rng(seed)
    innovations = random_generator.normal(
        loc=0.0,
        scale=normalized_scale,
        size=int(observations),
    )

    process = np.zeros(
        int(observations),
        dtype=float,
    )

    for index in range(1, process.size):
        process[index] = (
            normalized_coefficient
            * process[index - 1]
            + innovations[index]
        )

    return process


def main() -> None:
    """Demonstrate ARIMA fitting and out-of-sample forecasting."""
    series = simulate_ar_one_process(
        observations=500,
        coefficient=0.75,
        innovation_scale=1.0,
        seed=42,
    )

    forecast_horizon = 20
    training_values = series[:-forecast_horizon]
    testing_values = series[-forecast_horizon:]

    fitted_model = fit_arima_model(
        training_values,
        order=(1, 0, 0),
    )

    forecast = forecast_arima_model(
        fitted_model,
        steps=forecast_horizon,
        confidence_level=0.95,
    )

    metrics = calculate_forecast_metrics(
        testing_values,
        forecast.predicted_mean,
    )

    print("ARIMA time-series model")
    print(f"Model order: {fitted_model.order}")
    print(
        f"Training observations: "
        f"{fitted_model.observations}"
    )
    print(
        f"Forecast horizon: "
        f"{forecast.steps}"
    )

    print("\nInformation criteria")
    print(f"AIC: {fitted_model.aic:.6f}")
    print(f"BIC: {fitted_model.bic:.6f}")
    print(f"HQIC: {fitted_model.hqic:.6f}")

    print("\nForecast evaluation")
    print(
        f"MAE: "
        f"{metrics.mean_absolute_error:.6f}"
    )
    print(
        f"RMSE: "
        f"{metrics.root_mean_squared_error:.6f}"
    )
    print(
        f"MAPE: "
        f"{metrics.mean_absolute_percentage_error:.2%}"
    )

    print("\nFirst five forecasts")
    for index in range(5):
        print(
            f"Step {index + 1:2d}: "
            f"forecast={forecast.predicted_mean[index]: .6f}, "
            f"interval=["
            f"{forecast.lower_bound[index]: .6f}, "
            f"{forecast.upper_bound[index]: .6f}]"
        )


if __name__ == "__main__":
    main()