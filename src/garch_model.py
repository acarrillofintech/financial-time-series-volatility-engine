"""ARCH and GARCH volatility modeling and forecasting."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterable

import numpy as np
from arch import arch_model

from src.returns import (
    TRADING_DAYS_PER_YEAR,
    validate_periods_per_year,
    validate_returns,
)


PERCENT_SCALE = 100.0
PERCENT_VARIANCE_SCALE = PERCENT_SCALE**2


@dataclass(frozen=True)
class GarchSimulation:
    """Simulated returns and conditional variances."""

    returns: np.ndarray
    conditional_variances: np.ndarray
    conditional_volatilities: np.ndarray


@dataclass(frozen=True)
class FittedVolatilityModel:
    """Summary and underlying ARCH/GARCH fitted model."""

    model_type: str
    p: int
    q: int
    distribution: str
    observations: int
    log_likelihood: float
    aic: float
    bic: float
    parameters: dict[str, float]
    conditional_volatilities: np.ndarray
    standardized_residuals: np.ndarray
    result: Any = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class VolatilityForecast:
    """Multi-period conditional volatility forecast."""

    steps: int
    variance_forecasts: np.ndarray
    volatility_forecasts: np.ndarray
    annualized_volatility_forecasts: np.ndarray


def validate_model_type(model_type: str) -> str:
    """Validate and normalize the volatility-model name."""
    if not isinstance(model_type, str):
        raise TypeError(
            "model_type must be a string."
        )

    normalized_type = model_type.strip().upper()

    if normalized_type not in {"ARCH", "GARCH"}:
        raise ValueError(
            "model_type must be 'ARCH' or 'GARCH'."
        )

    return normalized_type


def validate_model_orders(
    *,
    model_type: str,
    p: int,
    q: int,
) -> tuple[int, int]:
    """Validate ARCH/GARCH lag orders."""
    normalized_type = validate_model_type(
        model_type
    )

    for name, value in (("p", p), ("q", q)):
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

    normalized_p = int(p)
    normalized_q = int(q)

    if normalized_p < 1:
        raise ValueError(
            "p must be at least one."
        )

    if normalized_q < 0:
        raise ValueError(
            "q cannot be negative."
        )

    if (
        normalized_type == "ARCH"
        and normalized_q != 0
    ):
        raise ValueError(
            "ARCH models require q equal to zero."
        )

    if (
        normalized_type == "GARCH"
        and normalized_q < 1
    ):
        raise ValueError(
            "GARCH models require q at least one."
        )

    return normalized_p, normalized_q


def validate_distribution(
    distribution: str,
) -> str:
    """Validate the innovation distribution."""
    if not isinstance(distribution, str):
        raise TypeError(
            "distribution must be a string."
        )

    normalized_distribution = (
        distribution.strip().lower()
    )

    valid_distributions = {
        "normal",
        "t",
        "skewt",
        "ged",
    }

    if (
        normalized_distribution
        not in valid_distributions
    ):
        raise ValueError(
            "distribution must be one of: "
            "'normal', 't', 'skewt', or 'ged'."
        )

    return normalized_distribution


def validate_mean_model(mean_model: str) -> str:
    """Validate the conditional-mean specification."""
    if not isinstance(mean_model, str):
        raise TypeError(
            "mean_model must be a string."
        )

    normalized_mean = mean_model.strip().lower()

    mapping = {
        "zero": "Zero",
        "constant": "Constant",
    }

    if normalized_mean not in mapping:
        raise ValueError(
            "mean_model must be 'zero' or 'constant'."
        )

    return mapping[normalized_mean]


def validate_forecast_steps(steps: int) -> int:
    """Validate a volatility forecast horizon."""
    if (
        not isinstance(steps, Integral)
        or isinstance(steps, bool)
    ):
        raise TypeError(
            "steps must be an integer."
        )

    normalized_steps = int(steps)

    if normalized_steps <= 0:
        raise ValueError(
            "steps must be greater than zero."
        )

    return normalized_steps


def fit_volatility_model(
    returns: Iterable[float] | np.ndarray,
    *,
    model_type: str = "GARCH",
    p: int = 1,
    q: int = 1,
    distribution: str = "normal",
    mean_model: str = "zero",
) -> FittedVolatilityModel:
    """Fit an ARCH or GARCH conditional-volatility model.

    Decimal returns are multiplied by 100 before estimation to
    improve numerical conditioning. Volatilities and variances
    are converted back to decimal units before being returned.
    """
    validated_returns = validate_returns(
        returns,
        minimum_observations=100,
    )
    normalized_type = validate_model_type(
        model_type
    )
    normalized_p, normalized_q = (
        validate_model_orders(
            model_type=normalized_type,
            p=p,
            q=q,
        )
    )
    normalized_distribution = (
        validate_distribution(distribution)
    )
    normalized_mean = validate_mean_model(
        mean_model
    )

    if np.isclose(
        np.var(validated_returns, ddof=1),
        0.0,
    ):
        raise ValueError(
            "volatility modeling requires "
            "non-constant returns."
        )

    scaled_returns = (
        validated_returns * PERCENT_SCALE
    )

    model = arch_model(
        scaled_returns,
        mean=normalized_mean,
        vol=normalized_type,
        p=normalized_p,
        q=normalized_q,
        dist=normalized_distribution,
        rescale=False,
    )

    fitted_result = model.fit(
        update_freq=0,
        disp="off",
        show_warning=False,
    )

    parameters = {
        str(name): float(value)
        for name, value
        in fitted_result.params.items()
    }

    conditional_volatilities = (
        np.asarray(
            fitted_result.conditional_volatility,
            dtype=float,
        )
        / PERCENT_SCALE
    )

    standardized_residuals = np.asarray(
        fitted_result.std_resid,
        dtype=float,
    )

    if not np.all(
        np.isfinite(conditional_volatilities)
    ):
        raise RuntimeError(
            "model produced non-finite "
            "conditional volatilities."
        )

    if not np.all(
        np.isfinite(standardized_residuals)
    ):
        raise RuntimeError(
            "model produced non-finite "
            "standardized residuals."
        )

    return FittedVolatilityModel(
        model_type=normalized_type,
        p=normalized_p,
        q=normalized_q,
        distribution=normalized_distribution,
        observations=int(
            validated_returns.size
        ),
        log_likelihood=float(
            fitted_result.loglikelihood
        ),
        aic=float(fitted_result.aic),
        bic=float(fitted_result.bic),
        parameters=parameters,
        conditional_volatilities=(
            conditional_volatilities
        ),
        standardized_residuals=(
            standardized_residuals
        ),
        result=fitted_result,
    )


def forecast_volatility_model(
    fitted_model: FittedVolatilityModel,
    *,
    steps: int,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> VolatilityForecast:
    """Forecast conditional variance and volatility."""
    if not isinstance(
        fitted_model,
        FittedVolatilityModel,
    ):
        raise TypeError(
            "fitted_model must be a "
            "FittedVolatilityModel."
        )

    normalized_steps = validate_forecast_steps(
        steps
    )
    frequency = validate_periods_per_year(
        periods_per_year
    )

    raw_forecast = fitted_model.result.forecast(
        horizon=normalized_steps,
        reindex=False,
    )

    scaled_variances = np.asarray(
        raw_forecast.variance.values[-1],
        dtype=float,
    )

    variance_forecasts = (
        scaled_variances
        / PERCENT_VARIANCE_SCALE
    )
    volatility_forecasts = np.sqrt(
        variance_forecasts
    )
    annualized_volatility_forecasts = (
        volatility_forecasts
        * np.sqrt(frequency)
    )

    if not np.all(
        np.isfinite(variance_forecasts)
    ):
        raise RuntimeError(
            "model produced non-finite "
            "variance forecasts."
        )

    if np.any(variance_forecasts <= 0.0):
        raise RuntimeError(
            "model produced non-positive "
            "variance forecasts."
        )

    return VolatilityForecast(
        steps=normalized_steps,
        variance_forecasts=variance_forecasts,
        volatility_forecasts=(
            volatility_forecasts
        ),
        annualized_volatility_forecasts=(
            annualized_volatility_forecasts
        ),
    )


def calculate_persistence(
    fitted_model: FittedVolatilityModel,
) -> float:
    """Calculate total ARCH/GARCH volatility persistence."""
    if not isinstance(
        fitted_model,
        FittedVolatilityModel,
    ):
        raise TypeError(
            "fitted_model must be a "
            "FittedVolatilityModel."
        )

    alpha_sum = sum(
        value
        for name, value
        in fitted_model.parameters.items()
        if name.startswith("alpha[")
    )
    beta_sum = sum(
        value
        for name, value
        in fitted_model.parameters.items()
        if name.startswith("beta[")
    )

    return float(alpha_sum + beta_sum)


def calculate_unconditional_variance(
    fitted_model: FittedVolatilityModel,
) -> float:
    """Calculate long-run variance for a stationary model."""
    if not isinstance(
        fitted_model,
        FittedVolatilityModel,
    ):
        raise TypeError(
            "fitted_model must be a "
            "FittedVolatilityModel."
        )

    persistence = calculate_persistence(
        fitted_model
    )

    if persistence >= 1.0:
        raise ValueError(
            "unconditional variance is undefined "
            "when persistence is at least one."
        )

    omega = fitted_model.parameters.get(
        "omega"
    )

    if omega is None:
        raise ValueError(
            "fitted model does not contain omega."
        )

    scaled_unconditional_variance = (
        omega / (1.0 - persistence)
    )

    return float(
        scaled_unconditional_variance
        / PERCENT_VARIANCE_SCALE
    )


def simulate_garch_process(
    *,
    observations: int = 1000,
    omega: float = 0.000001,
    alpha: float = 0.08,
    beta: float = 0.90,
    seed: int | None = 42,
) -> GarchSimulation:
    """Simulate a stationary Gaussian GARCH(1,1) process."""
    if (
        not isinstance(observations, Integral)
        or isinstance(observations, bool)
    ):
        raise TypeError(
            "observations must be an integer."
        )

    normalized_observations = int(
        observations
    )

    if normalized_observations < 2:
        raise ValueError(
            "observations must be at least two."
        )

    for name, value in (
        ("omega", omega),
        ("alpha", alpha),
        ("beta", beta),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
        ):
            raise TypeError(
                f"{name} must be a real number."
            )

        if not np.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

    normalized_omega = float(omega)
    normalized_alpha = float(alpha)
    normalized_beta = float(beta)

    if normalized_omega <= 0.0:
        raise ValueError(
            "omega must be greater than zero."
        )

    if normalized_alpha < 0.0:
        raise ValueError(
            "alpha cannot be negative."
        )

    if normalized_beta < 0.0:
        raise ValueError(
            "beta cannot be negative."
        )

    persistence = (
        normalized_alpha
        + normalized_beta
    )

    if persistence >= 1.0:
        raise ValueError(
            "alpha plus beta must be less than one."
        )

    unconditional_variance = (
        normalized_omega
        / (1.0 - persistence)
    )

    random_generator = np.random.default_rng(
        seed
    )
    innovations = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=normalized_observations,
    )

    returns = np.empty(
        normalized_observations,
        dtype=float,
    )
    conditional_variances = np.empty(
        normalized_observations,
        dtype=float,
    )

    conditional_variances[0] = (
        unconditional_variance
    )
    returns[0] = (
        np.sqrt(conditional_variances[0])
        * innovations[0]
    )

    for index in range(
        1,
        normalized_observations,
    ):
        conditional_variances[index] = (
            normalized_omega
            + normalized_alpha
            * returns[index - 1] ** 2
            + normalized_beta
            * conditional_variances[index - 1]
        )

        returns[index] = (
            np.sqrt(
                conditional_variances[index]
            )
            * innovations[index]
        )

    return GarchSimulation(
        returns=returns,
        conditional_variances=(
            conditional_variances
        ),
        conditional_volatilities=np.sqrt(
            conditional_variances
        ),
    )


def main() -> None:
    """Demonstrate GARCH estimation and forecasting."""
    simulation = simulate_garch_process(
        observations=1500,
        omega=0.000001,
        alpha=0.08,
        beta=0.90,
        seed=42,
    )

    fitted_model = fit_volatility_model(
        simulation.returns,
        model_type="GARCH",
        p=1,
        q=1,
        distribution="normal",
        mean_model="zero",
    )

    forecast = forecast_volatility_model(
        fitted_model,
        steps=10,
    )

    persistence = calculate_persistence(
        fitted_model
    )
    unconditional_variance = (
        calculate_unconditional_variance(
            fitted_model
        )
    )

    print("GARCH volatility model")
    print(
        f"Observations: "
        f"{fitted_model.observations}"
    )
    print(
        f"Model: "
        f"{fitted_model.model_type}"
        f"({fitted_model.p}, {fitted_model.q})"
    )
    print(
        f"Distribution: "
        f"{fitted_model.distribution}"
    )

    print("\nEstimated parameters")
    for name, value in (
        fitted_model.parameters.items()
    ):
        print(
            f"{name}: "
            f"{value:.6f}"
        )

    print("\nModel diagnostics")
    print(
        f"Log-likelihood: "
        f"{fitted_model.log_likelihood:.6f}"
    )
    print(f"AIC: {fitted_model.aic:.6f}")
    print(f"BIC: {fitted_model.bic:.6f}")
    print(
        f"Persistence: "
        f"{persistence:.6f}"
    )
    print(
        f"Long-run daily volatility: "
        f"{np.sqrt(unconditional_variance):.4%}"
    )

    print("\nTen-day volatility forecast")
    for index in range(forecast.steps):
        print(
            f"Day {index + 1:2d}: "
            f"daily="
            f"{forecast.volatility_forecasts[index]:.4%}, "
            f"annualized="
            f"{forecast.annualized_volatility_forecasts[index]:.2%}"
        )


if __name__ == "__main__":
    main()