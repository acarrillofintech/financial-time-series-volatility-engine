"""Financial return calculations and descriptive statistics."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from scipy.stats import kurtosis, skew


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ReturnStatistics:
    """Summary statistics for a series of simple financial returns."""

    observations: int
    mean_daily_return: float
    daily_volatility: float
    annualized_return: float
    annualized_volatility: float
    cumulative_return: float
    minimum_return: float
    maximum_return: float
    skewness: float
    excess_kurtosis: float


def _as_one_dimensional_array(
    values: Iterable[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Convert numerical input into a finite one-dimensional array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def validate_prices(
    prices: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Validate and return a one-dimensional price array."""
    validated_prices = _as_one_dimensional_array(
        prices,
        name="prices",
    )

    if validated_prices.size < 2:
        raise ValueError(
            "prices must contain at least two observations."
        )

    if np.any(validated_prices <= 0.0):
        raise ValueError("prices must be strictly positive.")

    return validated_prices


def validate_returns(
    returns: Iterable[float] | np.ndarray,
    *,
    minimum_observations: int = 1,
) -> np.ndarray:
    """Validate and return a one-dimensional return array."""
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

    validated_returns = _as_one_dimensional_array(
        returns,
        name="returns",
    )

    if validated_returns.size < minimum_observations:
        raise ValueError(
            "returns does not contain enough observations."
        )

    return validated_returns


def validate_periods_per_year(
    periods_per_year: int,
) -> int:
    """Validate the annualization frequency."""
    if (
        not isinstance(periods_per_year, Integral)
        or isinstance(periods_per_year, bool)
    ):
        raise TypeError(
            "periods_per_year must be an integer."
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be greater than zero."
        )

    return int(periods_per_year)


def calculate_simple_returns(
    prices: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Calculate simple returns from consecutive prices.

    The simple return at time t is:

        R_t = P_t / P_(t-1) - 1
    """
    validated_prices = validate_prices(prices)

    return (
        validated_prices[1:]
        / validated_prices[:-1]
        - 1.0
    )


def calculate_log_returns(
    prices: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Calculate continuously compounded logarithmic returns.

    The log return at time t is:

        r_t = log(P_t / P_(t-1))
    """
    validated_prices = validate_prices(prices)

    return np.diff(np.log(validated_prices))


def convert_log_to_simple_returns(
    log_returns: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Convert logarithmic returns into simple returns."""
    validated_returns = validate_returns(log_returns)

    return np.expm1(validated_returns)


def convert_simple_to_log_returns(
    simple_returns: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Convert simple returns into logarithmic returns."""
    validated_returns = validate_returns(simple_returns)

    if np.any(validated_returns <= -1.0):
        raise ValueError(
            "simple returns must be greater than -1."
        )

    return np.log1p(validated_returns)


def calculate_cumulative_return(
    simple_returns: Iterable[float] | np.ndarray,
) -> float:
    """Calculate total compounded return over the sample."""
    validated_returns = validate_returns(simple_returns)

    if np.any(validated_returns < -1.0):
        raise ValueError(
            "simple returns cannot be less than -1."
        )

    growth_factor = np.prod(1.0 + validated_returns)

    return float(growth_factor - 1.0)


def calculate_annualized_return(
    simple_returns: Iterable[float] | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualize the arithmetic mean of periodic returns."""
    validated_returns = validate_returns(simple_returns)
    frequency = validate_periods_per_year(periods_per_year)

    return float(np.mean(validated_returns) * frequency)


def calculate_annualized_volatility(
    returns: Iterable[float] | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualize sample volatility using the square-root-of-time rule."""
    validated_returns = validate_returns(
        returns,
        minimum_observations=2,
    )
    frequency = validate_periods_per_year(periods_per_year)

    sample_volatility = np.std(
        validated_returns,
        ddof=1,
    )

    return float(sample_volatility * np.sqrt(frequency))


def calculate_return_statistics(
    simple_returns: Iterable[float] | np.ndarray,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ReturnStatistics:
    """Calculate a complete statistical summary of simple returns."""
    validated_returns = validate_returns(
        simple_returns,
        minimum_observations=2,
    )
    frequency = validate_periods_per_year(periods_per_year)

    mean_daily_return = float(
        np.mean(validated_returns)
    )
    daily_volatility = float(
        np.std(validated_returns, ddof=1)
    )

    if validated_returns.size >= 3:
        return_skewness = float(
            skew(validated_returns, bias=False)
        )
    else:
        return_skewness = float("nan")

    if validated_returns.size >= 4:
        return_kurtosis = float(
            kurtosis(
                validated_returns,
                fisher=True,
                bias=False,
            )
        )
    else:
        return_kurtosis = float("nan")

    return ReturnStatistics(
        observations=int(validated_returns.size),
        mean_daily_return=mean_daily_return,
        daily_volatility=daily_volatility,
        annualized_return=float(
            mean_daily_return * frequency
        ),
        annualized_volatility=float(
            daily_volatility * np.sqrt(frequency)
        ),
        cumulative_return=calculate_cumulative_return(
            validated_returns
        ),
        minimum_return=float(
            np.min(validated_returns)
        ),
        maximum_return=float(
            np.max(validated_returns)
        ),
        skewness=return_skewness,
        excess_kurtosis=return_kurtosis,
    )


def main() -> None:
    """Run a small financial-return demonstration."""
    prices = np.array(
        [
            100.00,
            101.20,
            100.70,
            102.30,
            101.80,
            103.60,
            104.10,
            103.40,
            105.20,
            106.00,
            105.50,
        ],
        dtype=float,
    )

    simple_returns = calculate_simple_returns(prices)
    log_returns = calculate_log_returns(prices)
    statistics = calculate_return_statistics(
        simple_returns
    )

    print("Financial return analysis")
    print(f"Price observations: {prices.size}")
    print(
        f"Return observations: "
        f"{statistics.observations}"
    )

    print("\nLatest simple returns")
    for value in simple_returns[-5:]:
        print(f"{value: .4%}")

    print("\nReturn statistics")
    print(
        f"Average daily return: "
        f"{statistics.mean_daily_return:.4%}"
    )
    print(
        f"Daily volatility: "
        f"{statistics.daily_volatility:.4%}"
    )
    print(
        f"Annualized return: "
        f"{statistics.annualized_return:.2%}"
    )
    print(
        f"Annualized volatility: "
        f"{statistics.annualized_volatility:.2%}"
    )
    print(
        f"Cumulative return: "
        f"{statistics.cumulative_return:.2%}"
    )
    print(
        f"Minimum return: "
        f"{statistics.minimum_return:.4%}"
    )
    print(
        f"Maximum return: "
        f"{statistics.maximum_return:.4%}"
    )
    print(
        f"Skewness: "
        f"{statistics.skewness:.6f}"
    )
    print(
        f"Excess kurtosis: "
        f"{statistics.excess_kurtosis:.6f}"
    )

    print("\nLatest logarithmic returns")
    for value in log_returns[-5:]:
        print(f"{value: .6f}")


if __name__ == "__main__":
    main()