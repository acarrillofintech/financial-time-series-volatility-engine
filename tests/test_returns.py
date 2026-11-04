"""Tests for financial return calculations."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.returns import (
    ReturnStatistics,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_cumulative_return,
    calculate_log_returns,
    calculate_return_statistics,
    calculate_simple_returns,
    convert_log_to_simple_returns,
    convert_simple_to_log_returns,
    main,
    validate_periods_per_year,
    validate_prices,
    validate_returns,
)


def test_validate_prices_returns_float_array() -> None:
    """Valid prices should become a NumPy float array."""
    result = validate_prices([100, 101, 102])

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_array_equal(
        result,
        np.array([100.0, 101.0, 102.0]),
    )


def test_validate_prices_requires_two_observations() -> None:
    """At least two prices are required to calculate a return."""
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        validate_prices([100.0])


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, 0.0],
        [100.0, -1.0],
    ],
)
def test_validate_prices_requires_positive_values(
    prices: list[float],
) -> None:
    """Prices must be strictly positive."""
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        validate_prices(prices)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0, np.nan],
        [100.0, np.inf],
        [100.0, -np.inf],
    ],
)
def test_validate_prices_rejects_non_finite_values(
    prices: list[float],
) -> None:
    """Prices cannot contain NaN or infinite values."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_prices(prices)


def test_validate_prices_rejects_two_dimensional_data() -> None:
    """Price input must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_prices(
            np.array(
                [
                    [100.0, 101.0],
                    [102.0, 103.0],
                ]
            )
        )


def test_validate_returns_returns_float_array() -> None:
    """Valid returns should become a NumPy float array."""
    result = validate_returns([0.01, -0.02])

    assert result.dtype == float
    np.testing.assert_array_equal(
        result,
        np.array([0.01, -0.02]),
    )


def test_validate_returns_rejects_empty_input() -> None:
    """An empty return series should be rejected."""
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        validate_returns([])


def test_validate_returns_rejects_non_finite_values() -> None:
    """Returns cannot contain non-finite values."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_returns([0.01, np.nan])


def test_validate_returns_rejects_insufficient_sample() -> None:
    """The requested minimum sample size should be enforced."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        validate_returns(
            [0.01],
            minimum_observations=2,
        )


@pytest.mark.parametrize(
    "minimum_observations",
    [0, -1],
)
def test_validate_returns_rejects_invalid_minimum(
    minimum_observations: int,
) -> None:
    """The minimum number of observations must be positive."""
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        validate_returns(
            [0.01],
            minimum_observations=minimum_observations,
        )


@pytest.mark.parametrize(
    "minimum_observations",
    [1.5, True],
)
def test_validate_returns_rejects_non_integer_minimum(
    minimum_observations: object,
) -> None:
    """The minimum sample size must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_returns(
            [0.01],
            minimum_observations=minimum_observations,  # type: ignore[arg-type]
        )


def test_calculate_simple_returns_known_values() -> None:
    """Simple returns should match known price changes."""
    prices = np.array([100.0, 110.0, 99.0])

    result = calculate_simple_returns(prices)

    np.testing.assert_allclose(
        result,
        np.array([0.10, -0.10]),
    )


def test_calculate_log_returns_known_values() -> None:
    """Log returns should equal differences in log prices."""
    prices = np.array([100.0, 110.0, 99.0])

    result = calculate_log_returns(prices)

    expected = np.log(
        np.array([110.0 / 100.0, 99.0 / 110.0])
    )
    np.testing.assert_allclose(result, expected)


def test_simple_and_log_return_conversion_round_trip() -> None:
    """Simple returns should survive conversion to log returns."""
    simple_returns = np.array(
        [0.05, -0.02, 0.01, 0.08]
    )

    log_returns = convert_simple_to_log_returns(
        simple_returns
    )
    reconstructed = convert_log_to_simple_returns(
        log_returns
    )

    np.testing.assert_allclose(
        reconstructed,
        simple_returns,
    )


@pytest.mark.parametrize(
    "invalid_return",
    [-1.0, -1.20],
)
def test_simple_to_log_requires_return_above_minus_one(
    invalid_return: float,
) -> None:
    """Log conversion is undefined for a total loss or worse."""
    with pytest.raises(
        ValueError,
        match="greater than -1",
    ):
        convert_simple_to_log_returns(
            [0.01, invalid_return]
        )


def test_calculate_cumulative_return_known_value() -> None:
    """Compounding 10% and -10% should produce -1%."""
    result = calculate_cumulative_return(
        [0.10, -0.10]
    )

    assert result == pytest.approx(-0.01)


def test_total_loss_produces_minus_one_cumulative_return() -> None:
    """A -100% period should produce a complete cumulative loss."""
    result = calculate_cumulative_return(
        [0.10, -1.0, 0.50]
    )

    assert result == pytest.approx(-1.0)


def test_cumulative_return_rejects_impossible_loss() -> None:
    """A simple return below -100% is invalid."""
    with pytest.raises(
        ValueError,
        match="cannot be less",
    ):
        calculate_cumulative_return(
            [0.01, -1.01]
        )


def test_calculate_annualized_return() -> None:
    """Arithmetic mean return should scale by frequency."""
    returns = np.array([0.01, 0.02, 0.03])

    result = calculate_annualized_return(
        returns,
        periods_per_year=12,
    )

    assert result == pytest.approx(0.24)


def test_calculate_annualized_volatility() -> None:
    """Volatility should follow the square-root-of-time rule."""
    returns = np.array([0.01, -0.01, 0.02, -0.02])

    result = calculate_annualized_volatility(
        returns,
        periods_per_year=12,
    )

    expected = np.std(
        returns,
        ddof=1,
    ) * np.sqrt(12)

    assert result == pytest.approx(expected)


def test_annualized_volatility_requires_two_returns() -> None:
    """Sample volatility requires at least two observations."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        calculate_annualized_volatility([0.01])


@pytest.mark.parametrize(
    "periods_per_year",
    [0, -1],
)
def test_validate_periods_rejects_non_positive_values(
    periods_per_year: int,
) -> None:
    """Annualization frequency must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_periods_per_year(periods_per_year)


@pytest.mark.parametrize(
    "periods_per_year",
    [252.0, "252", True],
)
def test_validate_periods_rejects_non_integer_values(
    periods_per_year: object,
) -> None:
    """Annualization frequency must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_periods_per_year(
            periods_per_year  # type: ignore[arg-type]
        )


def test_calculate_return_statistics() -> None:
    """The summary function should return all expected metrics."""
    returns = np.array(
        [0.01, -0.02, 0.03, 0.00, 0.02]
    )

    result = calculate_return_statistics(
        returns,
        periods_per_year=12,
    )

    assert isinstance(result, ReturnStatistics)
    assert result.observations == 5
    assert result.mean_daily_return == pytest.approx(
        np.mean(returns)
    )
    assert result.daily_volatility == pytest.approx(
        np.std(returns, ddof=1)
    )
    assert result.annualized_return == pytest.approx(
        np.mean(returns) * 12
    )
    assert result.annualized_volatility == pytest.approx(
        np.std(returns, ddof=1) * np.sqrt(12)
    )
    assert result.cumulative_return == pytest.approx(
        np.prod(1.0 + returns) - 1.0
    )
    assert result.minimum_return == pytest.approx(-0.02)
    assert result.maximum_return == pytest.approx(0.03)
    assert np.isfinite(result.skewness)
    assert np.isfinite(result.excess_kurtosis)


def test_statistics_returns_nan_for_small_shape_samples() -> None:
    """Skewness and kurtosis need larger samples."""
    result = calculate_return_statistics(
        [0.01, -0.01]
    )

    assert np.isnan(result.skewness)
    assert np.isnan(result.excess_kurtosis)


def test_return_statistics_is_immutable() -> None:
    """The result data class should be immutable."""
    result = calculate_return_statistics(
        [0.01, -0.01, 0.02, -0.02]
    )

    with pytest.raises(FrozenInstanceError):
        result.observations = 99  # type: ignore[misc]


def test_price_and_return_observation_relationship() -> None:
    """A price series of length n should produce n-1 returns."""
    prices = np.linspace(100.0, 110.0, 25)

    simple_returns = calculate_simple_returns(prices)
    log_returns = calculate_log_returns(prices)

    assert simple_returns.size == prices.size - 1
    assert log_returns.size == prices.size - 1


def test_simple_and_log_returns_have_same_growth() -> None:
    """Simple and log returns should imply the same total growth."""
    prices = np.array(
        [100.0, 102.0, 101.0, 105.0]
    )

    simple_returns = calculate_simple_returns(prices)
    log_returns = calculate_log_returns(prices)

    simple_growth = np.prod(1.0 + simple_returns)
    log_growth = np.exp(np.sum(log_returns))

    assert simple_growth == pytest.approx(log_growth)
    assert simple_growth == pytest.approx(
        prices[-1] / prices[0]
    )


def test_main_prints_analysis(capsys: pytest.CaptureFixture[str]) -> None:
    """The demonstration should print the principal metrics."""
    main()

    output = capsys.readouterr().out

    assert "Financial return analysis" in output
    assert "Annualized return" in output
    assert "Annualized volatility" in output
    assert "Cumulative return" in output
    assert "Skewness" in output
    assert "Excess kurtosis" in output