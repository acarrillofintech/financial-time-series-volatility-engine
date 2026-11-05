"""Exponentially weighted volatility estimation and forecasting."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from src.returns import (
    TRADING_DAYS_PER_YEAR,
    validate_periods_per_year,
    validate_returns,
)


@dataclass(frozen=True)
class EwmaVolatilityResult:
    """Estimated EWMA conditional variances and volatilities."""

    decay_factor: float
    observations: int
    initial_variance: float
    conditional_variances: np.ndarray
    conditional_volatilities: np.ndarray
    annualized_volatilities: np.ndarray


@dataclass(frozen=True)
class EwmaForecast:
    """Multi-period EWMA volatility forecast."""

    steps: int
    variance_forecasts: np.ndarray
    volatility_forecasts: np.ndarray
    annualized_volatility_forecasts: np.ndarray


def validate_decay_factor(
    decay_factor: float,
) -> float:
    """Validate an EWMA decay factor."""
    if (
        not isinstance(decay_factor, Real)
        or isinstance(decay_factor, bool)
    ):
        raise TypeError(
            "decay_factor must be a real number."
        )

    normalized_factor = float(decay_factor)

    if not 0.0 < normalized_factor < 1.0:
        raise ValueError(
            "decay_factor must be between zero and one."
        )

    return normalized_factor


def validate_initial_variance(
    initial_variance: float,
) -> float:
    """Validate a strictly positive initial variance."""
    if (
        not isinstance(initial_variance, Real)
        or isinstance(initial_variance, bool)
    ):
        raise TypeError(
            "initial_variance must be a real number."
        )

    normalized_variance = float(initial_variance)

    if not np.isfinite(normalized_variance):
        raise ValueError(
            "initial_variance must be finite."
        )

    if normalized_variance <= 0.0:
        raise ValueError(
            "initial_variance must be greater than zero."
        )

    return normalized_variance


def validate_forecast_horizon(steps: int) -> int:
    """Validate an EWMA forecast horizon."""
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


def calculate_ewma_weights(
    *,
    decay_factor: float = 0.94,
    observations: int,
) -> np.ndarray:
    """Calculate EWMA weights from newest to oldest observation."""
    normalized_factor = validate_decay_factor(
        decay_factor
    )

    if (
        not isinstance(observations, Integral)
        or isinstance(observations, bool)
    ):
        raise TypeError(
            "observations must be an integer."
        )

    normalized_observations = int(observations)

    if normalized_observations <= 0:
        raise ValueError(
            "observations must be greater than zero."
        )

    lags = np.arange(
        normalized_observations,
        dtype=float,
    )

    return (
        1.0 - normalized_factor
    ) * normalized_factor**lags


def calculate_ewma_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    decay_factor: float = 0.94,
    initial_variance: float | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> EwmaVolatilityResult:
    """Calculate EWMA conditional variance and volatility.

    The recursion is:

        sigma_t^2 =
            lambda * sigma_(t-1)^2
            + (1-lambda) * r_(t-1)^2

    The variance at index t is the one-step-ahead conditional
    variance based on information available through index t-1.
    """
    validated_returns = validate_returns(
        returns,
        minimum_observations=2,
    )
    normalized_factor = validate_decay_factor(
        decay_factor
    )
    frequency = validate_periods_per_year(
        periods_per_year
    )

    if initial_variance is None:
        selected_initial_variance = float(
            np.var(
                validated_returns,
                ddof=1,
            )
        )

        if np.isclose(
            selected_initial_variance,
            0.0,
        ):
            raise ValueError(
                "returns must have positive sample variance."
            )
    else:
        selected_initial_variance = (
            validate_initial_variance(
                initial_variance
            )
        )

    conditional_variances = np.empty(
        validated_returns.size,
        dtype=float,
    )
    conditional_variances[0] = (
        selected_initial_variance
    )

    for index in range(
        1,
        validated_returns.size,
    ):
        previous_squared_return = (
            validated_returns[index - 1] ** 2
        )

        conditional_variances[index] = (
            normalized_factor
            * conditional_variances[index - 1]
            + (1.0 - normalized_factor)
            * previous_squared_return
        )

    conditional_volatilities = np.sqrt(
        conditional_variances
    )
    annualized_volatilities = (
        conditional_volatilities
        * np.sqrt(frequency)
    )

    return EwmaVolatilityResult(
        decay_factor=normalized_factor,
        observations=int(
            validated_returns.size
        ),
        initial_variance=(
            selected_initial_variance
        ),
        conditional_variances=(
            conditional_variances
        ),
        conditional_volatilities=(
            conditional_volatilities
        ),
        annualized_volatilities=(
            annualized_volatilities
        ),
    )


def forecast_ewma_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    steps: int,
    decay_factor: float = 0.94,
    initial_variance: float | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> EwmaForecast:
    """Forecast EWMA variance and volatility.

    EWMA corresponds to an integrated GARCH structure. Its
    expected multi-step variance forecast remains equal to the
    next one-step forecast because it has no separate long-run
    variance target.
    """
    validated_returns = validate_returns(
        returns,
        minimum_observations=2,
    )
    normalized_steps = validate_forecast_horizon(
        steps
    )
    normalized_factor = validate_decay_factor(
        decay_factor
    )
    frequency = validate_periods_per_year(
        periods_per_year
    )

    estimated = calculate_ewma_volatility(
        validated_returns,
        decay_factor=normalized_factor,
        initial_variance=initial_variance,
        periods_per_year=frequency,
    )

    next_variance = (
        normalized_factor
        * estimated.conditional_variances[-1]
        + (1.0 - normalized_factor)
        * validated_returns[-1] ** 2
    )

    variance_forecasts = np.full(
        normalized_steps,
        next_variance,
        dtype=float,
    )
    volatility_forecasts = np.sqrt(
        variance_forecasts
    )
    annualized_volatility_forecasts = (
        volatility_forecasts
        * np.sqrt(frequency)
    )

    return EwmaForecast(
        steps=normalized_steps,
        variance_forecasts=variance_forecasts,
        volatility_forecasts=volatility_forecasts,
        annualized_volatility_forecasts=(
            annualized_volatility_forecasts
        ),
    )


def main() -> None:
    """Demonstrate EWMA response to a volatility regime change."""
    random_generator = np.random.default_rng(42)

    low_volatility_returns = random_generator.normal(
        loc=0.0,
        scale=0.008,
        size=250,
    )
    high_volatility_returns = random_generator.normal(
        loc=0.0,
        scale=0.025,
        size=250,
    )

    returns = np.concatenate(
        [
            low_volatility_returns,
            high_volatility_returns,
        ]
    )

    result = calculate_ewma_volatility(
        returns,
        decay_factor=0.94,
    )
    forecast = forecast_ewma_volatility(
        returns,
        steps=10,
        decay_factor=0.94,
    )
    weights = calculate_ewma_weights(
        decay_factor=0.94,
        observations=10,
    )

    print("EWMA volatility model")
    print(f"Observations: {result.observations}")
    print(
        f"Decay factor: "
        f"{result.decay_factor:.2f}"
    )
    print(
        f"Initial daily volatility: "
        f"{np.sqrt(result.initial_variance):.4%}"
    )
    print(
        f"Latest daily volatility: "
        f"{result.conditional_volatilities[-1]:.4%}"
    )
    print(
        f"Latest annualized volatility: "
        f"{result.annualized_volatilities[-1]:.2%}"
    )

    print("\nTen most recent EWMA weights")
    for lag, weight in enumerate(weights):
        print(
            f"Lag {lag:2d}: "
            f"{weight:.6f}"
        )

    print("\nVolatility forecast")
    print(
        f"Forecast horizon: "
        f"{forecast.steps}"
    )
    print(
        f"Daily volatility: "
        f"{forecast.volatility_forecasts[0]:.4%}"
    )
    print(
        f"Annualized volatility: "
        f"{forecast.annualized_volatility_forecasts[0]:.2%}"
    )


if __name__ == "__main__":
    main()