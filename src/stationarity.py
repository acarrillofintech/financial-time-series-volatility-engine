"""Stationarity tests and time-series diagnostic tools."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import numpy as np
from statsmodels.tsa.stattools import (
    acf,
    adfuller,
    pacf,
)


@dataclass(frozen=True)
class StationarityResult:
    """Result of an Augmented Dickey-Fuller stationarity test."""

    adf_statistic: float
    p_value: float
    used_lags: int
    observations: int
    critical_values: dict[str, float]
    significance_level: float
    is_stationary: bool


@dataclass(frozen=True)
class CorrelationDiagnostics:
    """Autocorrelation and partial-autocorrelation results."""

    lags: np.ndarray
    autocorrelation: np.ndarray
    partial_autocorrelation: np.ndarray


def validate_time_series(
    values: Iterable[float] | np.ndarray,
    *,
    minimum_observations: int = 3,
) -> np.ndarray:
    """Validate and return a finite one-dimensional time series."""
    if (
        not isinstance(minimum_observations, Integral)
        or isinstance(minimum_observations, bool)
    ):
        raise TypeError(
            "minimum_observations must be an integer."
        )

    if minimum_observations < 1:
        raise ValueError(
            "minimum_observations must be at least one."
        )

    series = np.asarray(values, dtype=float)

    if series.ndim != 1:
        raise ValueError(
            "time series must be one-dimensional."
        )

    if series.size < minimum_observations:
        raise ValueError(
            "time series does not contain enough observations."
        )

    if not np.all(np.isfinite(series)):
        raise ValueError(
            "time series must contain only finite values."
        )

    return series


def validate_significance_level(
    significance_level: float,
) -> float:
    """Validate a statistical significance level."""
    if (
        not isinstance(significance_level, Real)
        or isinstance(significance_level, bool)
    ):
        raise TypeError(
            "significance_level must be a real number."
        )

    normalized_level = float(significance_level)

    if not 0.0 < normalized_level < 1.0:
        raise ValueError(
            "significance_level must be between zero and one."
        )

    return normalized_level


def validate_lag_count(
    lags: int,
    *,
    observations: int,
) -> int:
    """Validate a lag count for correlation diagnostics."""
    if (
        not isinstance(lags, Integral)
        or isinstance(lags, bool)
    ):
        raise TypeError("lags must be an integer.")

    normalized_lags = int(lags)

    if normalized_lags < 1:
        raise ValueError(
            "lags must be greater than zero."
        )

    maximum_lags = observations // 2 - 1

    if maximum_lags < 1:
        raise ValueError(
            "time series is too short for lag diagnostics."
        )

    if normalized_lags > maximum_lags:
        raise ValueError(
            f"lags cannot exceed {maximum_lags} "
            "for this time series."
        )

    return normalized_lags


def difference_series(
    values: Iterable[float] | np.ndarray,
    *,
    periods: int = 1,
) -> np.ndarray:
    """Difference a time series by a selected number of periods.

    The first difference is:

        Delta X_t = X_t - X_(t-1)
    """
    series = validate_time_series(
        values,
        minimum_observations=2,
    )

    if (
        not isinstance(periods, Integral)
        or isinstance(periods, bool)
    ):
        raise TypeError("periods must be an integer.")

    normalized_periods = int(periods)

    if normalized_periods < 1:
        raise ValueError(
            "periods must be greater than zero."
        )

    if normalized_periods >= series.size:
        raise ValueError(
            "periods must be smaller than the series length."
        )

    return (
        series[normalized_periods:]
        - series[:-normalized_periods]
    )


def run_adf_test(
    values: Iterable[float] | np.ndarray,
    *,
    significance_level: float = 0.05,
    regression: str = "c",
    autolag: str | None = "aic", 
) -> StationarityResult:
    """Run the Augmented Dickey-Fuller unit-root test.

    Null hypothesis:
        The time series contains a unit root and is non-stationary.

    Alternative hypothesis:
        The time series is stationary.
    """
    series = validate_time_series(
        values,
        minimum_observations=10,
    )
    level = validate_significance_level(
        significance_level
    )

    valid_regressions = {"c", "ct", "ctt", "n"}

    if regression not in valid_regressions:
        raise ValueError(
            "regression must be one of: "
            "'c', 'ct', 'ctt', or 'n'."
        )

    if np.isclose(np.std(series), 0.0):
        raise ValueError(
            "ADF test requires a non-constant time series."
        )

    adf_output = adfuller(
    series,
    regression=regression,
    autolag=autolag,
    result_object=False,
)

    statistic = float(adf_output[0])
    p_value = float(adf_output[1])
    used_lags = int(adf_output[2])
    observations = int(adf_output[3])
    critical_values = {
        key: float(value)
        for key, value in adf_output[4].items()
    }

    return StationarityResult(
        adf_statistic=statistic,
        p_value=p_value,
        used_lags=used_lags,
        observations=observations,
        critical_values=critical_values,
        significance_level=level,
        is_stationary=p_value < level,
    )


def calculate_correlation_diagnostics(
    values: Iterable[float] | np.ndarray,
    *,
    lags: int = 20,
) -> CorrelationDiagnostics:
    """Calculate ACF and PACF values for a time series."""
    series = validate_time_series(
        values,
        minimum_observations=5,
    )
    normalized_lags = validate_lag_count(
        lags,
        observations=series.size,
    )

    autocorrelation_values = acf(
        series,
        nlags=normalized_lags,
        fft=True,
    )
    partial_autocorrelation_values = pacf(
        series,
        nlags=normalized_lags,
        method="ywm",
    )

    lag_values = np.arange(
        normalized_lags + 1,
        dtype=int,
    )

    return CorrelationDiagnostics(
        lags=lag_values,
        autocorrelation=np.asarray(
            autocorrelation_values,
            dtype=float,
        ),
        partial_autocorrelation=np.asarray(
            partial_autocorrelation_values,
            dtype=float,
        ),
    )


def main() -> None:
    """Demonstrate stationarity analysis."""
    random_generator = np.random.default_rng(42)

    white_noise = random_generator.normal(
        loc=0.0,
        scale=1.0,
        size=500,
    )

    random_walk = np.cumsum(
        random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=500,
        )
    )

    differenced_random_walk = difference_series(
        random_walk
    )

    white_noise_result = run_adf_test(
        white_noise
    )
    random_walk_result = run_adf_test(
        random_walk
    )
    differenced_result = run_adf_test(
        differenced_random_walk
    )

    print("Financial time-series stationarity")
    print("\nWhite noise")
    print(
        f"ADF statistic: "
        f"{white_noise_result.adf_statistic:.6f}"
    )
    print(
        f"P-value: "
        f"{white_noise_result.p_value:.6f}"
    )
    print(
        f"Stationary: "
        f"{white_noise_result.is_stationary}"
    )

    print("\nRandom walk")
    print(
        f"ADF statistic: "
        f"{random_walk_result.adf_statistic:.6f}"
    )
    print(
        f"P-value: "
        f"{random_walk_result.p_value:.6f}"
    )
    print(
        f"Stationary: "
        f"{random_walk_result.is_stationary}"
    )

    print("\nDifferenced random walk")
    print(
        f"ADF statistic: "
        f"{differenced_result.adf_statistic:.6f}"
    )
    print(
        f"P-value: "
        f"{differenced_result.p_value:.6f}"
    )
    print(
        f"Stationary: "
        f"{differenced_result.is_stationary}"
    )

    diagnostics = calculate_correlation_diagnostics(
        white_noise,
        lags=10,
    )

    print("\nCorrelation diagnostics")
    print(
        f"Calculated lags: "
        f"{diagnostics.lags[-1]}"
    )
    print(
        f"ACF at lag zero: "
        f"{diagnostics.autocorrelation[0]:.6f}"
    )
    print(
        f"PACF at lag zero: "
        f"{diagnostics.partial_autocorrelation[0]:.6f}"
    )


if __name__ == "__main__":
    main()