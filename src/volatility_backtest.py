"""Out-of-sample backtesting for volatility forecasts."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np

from src.data_loader import load_market_data
from src.ewma_volatility import (
    forecast_ewma_volatility,
    validate_decay_factor,
)
from src.garch_model import (
    fit_volatility_model,
    forecast_volatility_model,
)
from src.returns import validate_returns


@dataclass(frozen=True)
class VolatilityBacktestMetrics:
    """Evaluation metrics for variance forecasts."""

    observations: int
    mean_absolute_error: float
    root_mean_squared_error: float
    qlike_loss: float
    mean_realized_variance: float
    mean_forecast_variance: float
    forecast_bias: float
    variance_correlation: float


@dataclass(frozen=True)
class VolatilityBacktestResult:
    """Forecasts, realizations, and metrics for one model."""

    model_name: str
    estimation_window: int
    realized_variances: np.ndarray
    variance_forecasts: np.ndarray
    volatility_forecasts: np.ndarray
    metrics: VolatilityBacktestMetrics


@dataclass(frozen=True)
class VolatilityModelComparison:
    """EWMA and GARCH backtesting results."""

    ewma: VolatilityBacktestResult
    garch: VolatilityBacktestResult
    best_model_by_qlike: str


def validate_estimation_window(
    estimation_window: int,
    *,
    observations: int,
    minimum_window: int = 20,
) -> int:
    """Validate an out-of-sample estimation window."""
    if (
        not isinstance(estimation_window, Integral)
        or isinstance(estimation_window, bool)
    ):
        raise TypeError(
            "estimation_window must be an integer."
        )

    if (
        not isinstance(minimum_window, Integral)
        or isinstance(minimum_window, bool)
    ):
        raise TypeError(
            "minimum_window must be an integer."
        )

    normalized_window = int(estimation_window)
    normalized_minimum = int(minimum_window)

    if normalized_minimum < 2:
        raise ValueError(
            "minimum_window must be at least two."
        )

    if normalized_window < normalized_minimum:
        raise ValueError(
            f"estimation_window must be at least "
            f"{normalized_minimum}."
        )

    if normalized_window >= observations:
        raise ValueError(
            "estimation_window must be smaller "
            "than the number of observations."
        )

    return normalized_window


def validate_refit_frequency(
    refit_frequency: int,
) -> int:
    """Validate how often a GARCH model is re-estimated."""
    if (
        not isinstance(refit_frequency, Integral)
        or isinstance(refit_frequency, bool)
    ):
        raise TypeError(
            "refit_frequency must be an integer."
        )

    normalized_frequency = int(
        refit_frequency
    )

    if normalized_frequency <= 0:
        raise ValueError(
            "refit_frequency must be greater than zero."
        )

    return normalized_frequency


def calculate_volatility_backtest_metrics(
    realized_variances: (
        Iterable[float] | np.ndarray
    ),
    variance_forecasts: (
        Iterable[float] | np.ndarray
    ),
) -> VolatilityBacktestMetrics:
    """Calculate MAE, RMSE, QLIKE, bias, and correlation."""
    realized = validate_returns(
        realized_variances,
        minimum_observations=2,
    )
    forecasts = validate_returns(
        variance_forecasts,
        minimum_observations=2,
    )

    if realized.size != forecasts.size:
        raise ValueError(
            "realized variances and forecasts "
            "must have the same length."
        )

    if np.any(realized < 0.0):
        raise ValueError(
            "realized variances cannot be negative."
        )

    if np.any(forecasts <= 0.0):
        raise ValueError(
            "variance forecasts must be strictly positive."
        )

    errors = forecasts - realized

    mean_absolute_error = float(
        np.mean(np.abs(errors))
    )
    root_mean_squared_error = float(
        np.sqrt(np.mean(errors**2))
    )

    qlike_values = (
        np.log(forecasts)
        + realized / forecasts
    )
    qlike_loss = float(
        np.mean(qlike_values)
    )

    mean_realized_variance = float(
        np.mean(realized)
    )
    mean_forecast_variance = float(
        np.mean(forecasts)
    )
    forecast_bias = float(
        mean_forecast_variance
        - mean_realized_variance
    )

    if (
        np.isclose(np.std(realized), 0.0)
        or np.isclose(np.std(forecasts), 0.0)
    ):
        variance_correlation = float("nan")
    else:
        variance_correlation = float(
            np.corrcoef(
                realized,
                forecasts,
            )[0, 1]
        )

    return VolatilityBacktestMetrics(
        observations=int(realized.size),
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=(
            root_mean_squared_error
        ),
        qlike_loss=qlike_loss,
        mean_realized_variance=(
            mean_realized_variance
        ),
        mean_forecast_variance=(
            mean_forecast_variance
        ),
        forecast_bias=forecast_bias,
        variance_correlation=(
            variance_correlation
        ),
    )


def backtest_ewma_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    estimation_window: int = 250,
    decay_factor: float = 0.94,
) -> VolatilityBacktestResult:
    """Generate rolling one-step-ahead EWMA forecasts."""
    validated_returns = validate_returns(
        returns,
        minimum_observations=21,
    )
    window = validate_estimation_window(
        estimation_window,
        observations=validated_returns.size,
        minimum_window=20,
    )
    normalized_decay = validate_decay_factor(
        decay_factor
    )

    out_of_sample_observations = (
        validated_returns.size - window
    )
    variance_forecasts = np.empty(
        out_of_sample_observations,
        dtype=float,
    )

    for forecast_index, time_index in enumerate(
        range(window, validated_returns.size)
    ):
        estimation_sample = validated_returns[
            time_index - window : time_index
        ]

        forecast = forecast_ewma_volatility(
            estimation_sample,
            steps=1,
            decay_factor=normalized_decay,
        )

        variance_forecasts[forecast_index] = (
            forecast.variance_forecasts[0]
        )

    realized_returns = validated_returns[window:]
    realized_variances = realized_returns**2
    volatility_forecasts = np.sqrt(
        variance_forecasts
    )

    metrics = (
        calculate_volatility_backtest_metrics(
            realized_variances,
            variance_forecasts,
        )
    )

    return VolatilityBacktestResult(
        model_name="EWMA",
        estimation_window=window,
        realized_variances=realized_variances,
        variance_forecasts=variance_forecasts,
        volatility_forecasts=(
            volatility_forecasts
        ),
        metrics=metrics,
    )


def backtest_garch_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    estimation_window: int = 250,
    refit_frequency: int = 20,
    distribution: str = "normal",
) -> VolatilityBacktestResult:
    """Backtest block forecasts from rolling GARCH(1,1) models.

    The model is re-estimated at each block origin using only
    observations available at that time. Forecasts inside the
    block are generated without using future realized returns.
    """
    validated_returns = validate_returns(
        returns,
        minimum_observations=101,
    )
    window = validate_estimation_window(
        estimation_window,
        observations=validated_returns.size,
        minimum_window=100,
    )
    frequency = validate_refit_frequency(
        refit_frequency
    )

    forecast_blocks: list[np.ndarray] = []
    realized_blocks: list[np.ndarray] = []

    for origin in range(
        window,
        validated_returns.size,
        frequency,
    ):
        estimation_sample = validated_returns[
            origin - window : origin
        ]

        block_size = min(
            frequency,
            validated_returns.size - origin,
        )

        fitted_model = fit_volatility_model(
            estimation_sample,
            model_type="GARCH",
            p=1,
            q=1,
            distribution=distribution,
            mean_model="zero",
        )

        forecast = forecast_volatility_model(
            fitted_model,
            steps=block_size,
        )

        forecast_blocks.append(
            forecast.variance_forecasts
        )
        realized_blocks.append(
            validated_returns[
                origin : origin + block_size
            ]
            ** 2
        )

    variance_forecasts = np.concatenate(
        forecast_blocks
    )
    realized_variances = np.concatenate(
        realized_blocks
    )
    volatility_forecasts = np.sqrt(
        variance_forecasts
    )

    metrics = (
        calculate_volatility_backtest_metrics(
            realized_variances,
            variance_forecasts,
        )
    )

    return VolatilityBacktestResult(
        model_name="GARCH",
        estimation_window=window,
        realized_variances=realized_variances,
        variance_forecasts=variance_forecasts,
        volatility_forecasts=(
            volatility_forecasts
        ),
        metrics=metrics,
    )


def compare_volatility_models(
    returns: Iterable[float] | np.ndarray,
    *,
    estimation_window: int = 250,
    decay_factor: float = 0.94,
    refit_frequency: int = 20,
    distribution: str = "normal",
) -> VolatilityModelComparison:
    """Backtest and compare EWMA with GARCH."""
    validated_returns = validate_returns(
        returns,
        minimum_observations=101,
    )

    ewma_result = backtest_ewma_volatility(
        validated_returns,
        estimation_window=estimation_window,
        decay_factor=decay_factor,
    )
    garch_result = backtest_garch_volatility(
        validated_returns,
        estimation_window=estimation_window,
        refit_frequency=refit_frequency,
        distribution=distribution,
    )

    if (
        ewma_result.metrics.qlike_loss
        <= garch_result.metrics.qlike_loss
    ):
        best_model = "EWMA"
    else:
        best_model = "GARCH"

    return VolatilityModelComparison(
        ewma=ewma_result,
        garch=garch_result,
        best_model_by_qlike=best_model,
    )


def print_backtest_result(
    result: VolatilityBacktestResult,
) -> None:
    """Print one volatility backtesting summary."""
    print(f"\n{result.model_name}")
    print(
        f"Out-of-sample observations: "
        f"{result.metrics.observations}"
    )
    print(
        f"Variance MAE: "
        f"{result.metrics.mean_absolute_error:.10f}"
    )
    print(
        f"Variance RMSE: "
        f"{result.metrics.root_mean_squared_error:.10f}"
    )
    print(
        f"QLIKE loss: "
        f"{result.metrics.qlike_loss:.6f}"
    )
    print(
        f"Forecast bias: "
        f"{result.metrics.forecast_bias:.10f}"
    )
    print(
        f"Variance correlation: "
        f"{result.metrics.variance_correlation:.6f}"
    )


def main() -> None:
    """Backtest EWMA and GARCH using SPY returns."""
    market_data = load_market_data(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    comparison = compare_volatility_models(
        market_data.simple_returns.to_numpy(),
        estimation_window=500,
        decay_factor=0.94,
        refit_frequency=50,
        distribution="normal",
    )

    print("Volatility model backtesting")
    print(f"Asset: {market_data.ticker}")
    print(
        f"Return observations: "
        f"{market_data.simple_returns.size:,}"
    )
    print(
        f"Estimation window: "
        f"{comparison.ewma.estimation_window}"
    )

    print_backtest_result(
        comparison.ewma
    )
    print_backtest_result(
        comparison.garch
    )

    print("\nModel comparison")
    print(
        f"Best model by QLIKE: "
        f"{comparison.best_model_by_qlike}"
    )


if __name__ == "__main__":
    main()