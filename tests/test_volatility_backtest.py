"""Tests for volatility-model backtesting."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import src.volatility_backtest as backtest_module
from src.data_loader import MarketData
from src.garch_model import simulate_garch_process
from src.volatility_backtest import (
    VolatilityBacktestMetrics,
    VolatilityBacktestResult,
    VolatilityModelComparison,
    backtest_ewma_volatility,
    backtest_garch_volatility,
    calculate_volatility_backtest_metrics,
    compare_volatility_models,
    main,
    print_backtest_result,
    validate_estimation_window,
    validate_refit_frequency,
)


@pytest.fixture(scope="module")
def synthetic_returns() -> np.ndarray:
    """Create reproducible heteroskedastic returns."""
    simulation = simulate_garch_process(
        observations=400,
        omega=0.000001,
        alpha=0.08,
        beta=0.90,
        seed=321,
    )

    return simulation.returns


@pytest.fixture(scope="module")
def ewma_backtest(
    synthetic_returns: np.ndarray,
) -> VolatilityBacktestResult:
    """Create one reusable EWMA backtest."""
    return backtest_ewma_volatility(
        synthetic_returns,
        estimation_window=250,
        decay_factor=0.94,
    )


@pytest.fixture(scope="module")
def garch_backtest(
    synthetic_returns: np.ndarray,
) -> VolatilityBacktestResult:
    """Create one reusable GARCH backtest."""
    return backtest_garch_volatility(
        synthetic_returns,
        estimation_window=250,
        refit_frequency=50,
    )


@pytest.fixture(scope="module")
def model_comparison(
    ewma_backtest: VolatilityBacktestResult,
    garch_backtest: VolatilityBacktestResult,
) -> VolatilityModelComparison:
    """Create a reusable comparison object."""
    if (
        ewma_backtest.metrics.qlike_loss
        <= garch_backtest.metrics.qlike_loss
    ):
        best_model = "EWMA"
    else:
        best_model = "GARCH"

    return VolatilityModelComparison(
        ewma=ewma_backtest,
        garch=garch_backtest,
        best_model_by_qlike=best_model,
    )


def test_validate_estimation_window() -> None:
    """A valid estimation window should be returned."""
    result = validate_estimation_window(
        250,
        observations=1000,
        minimum_window=100,
    )

    assert result == 250


@pytest.mark.parametrize(
    "estimation_window",
    [99, 50, 0, -1],
)
def test_window_rejects_value_below_minimum(
    estimation_window: int,
) -> None:
    """Estimation window must meet the minimum."""
    with pytest.raises(
        ValueError,
        match="at least 100",
    ):
        validate_estimation_window(
            estimation_window,
            observations=1000,
            minimum_window=100,
        )


@pytest.mark.parametrize(
    "estimation_window",
    [1000, 1001],
)
def test_window_must_be_smaller_than_sample(
    estimation_window: int,
) -> None:
    """The window must leave out-of-sample observations."""
    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        validate_estimation_window(
            estimation_window,
            observations=1000,
            minimum_window=100,
        )


@pytest.mark.parametrize(
    "estimation_window",
    [250.0, "250", True],
)
def test_window_rejects_non_integer_values(
    estimation_window: object,
) -> None:
    """Estimation window must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_estimation_window(
            estimation_window,  # type: ignore[arg-type]
            observations=1000,
        )


@pytest.mark.parametrize(
    "minimum_window",
    [0, 1, -1],
)
def test_window_rejects_invalid_minimum(
    minimum_window: int,
) -> None:
    """Minimum estimation length must be at least two."""
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        validate_estimation_window(
            250,
            observations=1000,
            minimum_window=minimum_window,
        )


def test_refit_frequency_accepts_positive_integer() -> None:
    """A positive refit interval should be accepted."""
    assert validate_refit_frequency(20) == 20


@pytest.mark.parametrize(
    "refit_frequency",
    [0, -1],
)
def test_refit_frequency_rejects_non_positive_values(
    refit_frequency: int,
) -> None:
    """Refit interval must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_refit_frequency(
            refit_frequency
        )


@pytest.mark.parametrize(
    "refit_frequency",
    [20.0, "20", True],
)
def test_refit_frequency_rejects_non_integer_values(
    refit_frequency: object,
) -> None:
    """Refit interval must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_refit_frequency(
            refit_frequency  # type: ignore[arg-type]
        )


def test_backtest_metrics_known_values() -> None:
    """Metrics should match direct calculations."""
    realized = np.array(
        [0.0001, 0.0004, 0.0009]
    )
    forecasts = np.array(
        [0.0002, 0.0003, 0.0010]
    )

    result = (
        calculate_volatility_backtest_metrics(
            realized,
            forecasts,
        )
    )

    errors = forecasts - realized

    assert isinstance(
        result,
        VolatilityBacktestMetrics,
    )
    assert result.observations == 3
    assert (
        result.mean_absolute_error
        == pytest.approx(
            np.mean(np.abs(errors))
        )
    )
    assert (
        result.root_mean_squared_error
        == pytest.approx(
            np.sqrt(np.mean(errors**2))
        )
    )
    assert result.qlike_loss == pytest.approx(
        np.mean(
            np.log(forecasts)
            + realized / forecasts
        )
    )
    assert result.forecast_bias == pytest.approx(
        np.mean(forecasts)
        - np.mean(realized)
    )
    assert result.variance_correlation == pytest.approx(
        np.corrcoef(
            realized,
            forecasts,
        )[0, 1]
    )


def test_metrics_allow_zero_realized_variance() -> None:
    """A zero realized return is valid."""
    result = (
        calculate_volatility_backtest_metrics(
            [0.0, 0.0001, 0.0002],
            [0.0001, 0.0001, 0.0001],
        )
    )

    assert np.isfinite(result.qlike_loss)


def test_metrics_reject_negative_realized_variance() -> None:
    """Realized variance cannot be negative."""
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        calculate_volatility_backtest_metrics(
            [0.0001, -0.0001],
            [0.0001, 0.0001],
        )


@pytest.mark.parametrize(
    "invalid_forecast",
    [0.0, -0.0001],
)
def test_metrics_require_positive_forecasts(
    invalid_forecast: float,
) -> None:
    """Forecast variance must be strictly positive."""
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        calculate_volatility_backtest_metrics(
            [0.0001, 0.0002],
            [0.0001, invalid_forecast],
        )


def test_metrics_require_matching_lengths() -> None:
    """Realizations and forecasts must align."""
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_volatility_backtest_metrics(
            [0.0001, 0.0002, 0.0003],
            [0.0001, 0.0002],
        )


def test_constant_series_produces_nan_correlation() -> None:
    """Correlation is undefined for constant arrays."""
    result = (
        calculate_volatility_backtest_metrics(
            [0.0001, 0.0001, 0.0001],
            [0.0002, 0.0002, 0.0002],
        )
    )

    assert np.isnan(
        result.variance_correlation
    )


def test_ewma_backtest_structure(
    ewma_backtest: VolatilityBacktestResult,
) -> None:
    """EWMA backtest should return aligned forecasts."""
    result = ewma_backtest

    assert isinstance(
        result,
        VolatilityBacktestResult,
    )
    assert result.model_name == "EWMA"
    assert result.estimation_window == 250
    assert result.realized_variances.shape == (150,)
    assert result.variance_forecasts.shape == (150,)
    assert result.volatility_forecasts.shape == (150,)
    assert result.metrics.observations == 150
    assert np.all(result.realized_variances >= 0.0)
    assert np.all(result.variance_forecasts > 0.0)


def test_ewma_volatility_matches_variance(
    ewma_backtest: VolatilityBacktestResult,
) -> None:
    """EWMA volatility should equal square-root variance."""
    np.testing.assert_allclose(
        ewma_backtest.volatility_forecasts,
        np.sqrt(
            ewma_backtest.variance_forecasts
        ),
    )


def test_ewma_backtest_has_no_future_leakage() -> None:
    """Changing the final realization must not alter forecasts."""
    generator = np.random.default_rng(10)
    original_returns = generator.normal(
        scale=0.01,
        size=100,
    )
    changed_returns = original_returns.copy()
    changed_returns[-1] = 0.50

    original_result = backtest_ewma_volatility(
        original_returns,
        estimation_window=50,
    )
    changed_result = backtest_ewma_volatility(
        changed_returns,
        estimation_window=50,
    )

    np.testing.assert_allclose(
        original_result.variance_forecasts,
        changed_result.variance_forecasts,
    )
    assert (
        original_result.realized_variances[-1]
        != changed_result.realized_variances[-1]
    )


def test_ewma_validates_decay_factor(
    synthetic_returns: np.ndarray,
) -> None:
    """EWMA backtest should validate lambda."""
    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        backtest_ewma_volatility(
            synthetic_returns,
            estimation_window=250,
            decay_factor=1.0,
        )


def test_garch_backtest_structure(
    garch_backtest: VolatilityBacktestResult,
) -> None:
    """GARCH backtest should return aligned forecasts."""
    result = garch_backtest

    assert isinstance(
        result,
        VolatilityBacktestResult,
    )
    assert result.model_name == "GARCH"
    assert result.estimation_window == 250
    assert result.realized_variances.shape == (150,)
    assert result.variance_forecasts.shape == (150,)
    assert result.volatility_forecasts.shape == (150,)
    assert result.metrics.observations == 150
    assert np.all(result.variance_forecasts > 0.0)


def test_garch_volatility_matches_variance(
    garch_backtest: VolatilityBacktestResult,
) -> None:
    """GARCH volatility should equal square-root variance."""
    np.testing.assert_allclose(
        garch_backtest.volatility_forecasts,
        np.sqrt(
            garch_backtest.variance_forecasts
        ),
    )


def test_garch_backtest_handles_partial_final_block() -> None:
    """A final block shorter than refit frequency should work."""
    simulation = simulate_garch_process(
        observations=333,
        seed=55,
    )

    result = backtest_garch_volatility(
        simulation.returns,
        estimation_window=250,
        refit_frequency=40,
    )

    assert result.metrics.observations == 83
    assert result.variance_forecasts.shape == (83,)


def test_garch_backtest_validates_window(
    synthetic_returns: np.ndarray,
) -> None:
    """GARCH requires at least 100 estimation observations."""
    with pytest.raises(
        ValueError,
        match="at least 100",
    ):
        backtest_garch_volatility(
            synthetic_returns,
            estimation_window=99,
        )


def test_compare_models_returns_both_results(
    synthetic_returns: np.ndarray,
) -> None:
    """Comparison should include EWMA and GARCH."""
    result = compare_volatility_models(
        synthetic_returns,
        estimation_window=250,
        refit_frequency=75,
    )

    assert isinstance(
        result,
        VolatilityModelComparison,
    )
    assert result.ewma.model_name == "EWMA"
    assert result.garch.model_name == "GARCH"
    assert result.best_model_by_qlike in {
        "EWMA",
        "GARCH",
    }
    assert (
        result.ewma.metrics.observations
        == result.garch.metrics.observations
    )


def test_best_model_matches_lower_qlike(
    model_comparison: VolatilityModelComparison,
) -> None:
    """The selected model should have the lowest QLIKE."""
    comparison = model_comparison

    expected = (
        "EWMA"
        if (
            comparison.ewma.metrics.qlike_loss
            <= comparison.garch.metrics.qlike_loss
        )
        else "GARCH"
    )

    assert (
        comparison.best_model_by_qlike
        == expected
    )


def test_print_backtest_result(
    ewma_backtest: VolatilityBacktestResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Summary printer should include all metrics."""
    print_backtest_result(ewma_backtest)

    output = capsys.readouterr().out

    assert "EWMA" in output
    assert "Out-of-sample observations" in output
    assert "Variance MAE" in output
    assert "Variance RMSE" in output
    assert "QLIKE loss" in output
    assert "Forecast bias" in output
    assert "Variance correlation" in output


def test_backtest_result_is_immutable(
    ewma_backtest: VolatilityBacktestResult,
) -> None:
    """Backtest-result attributes should be immutable."""
    with pytest.raises(FrozenInstanceError):
        ewma_backtest.model_name = "OTHER"  # type: ignore[misc]


def test_metrics_are_immutable(
    ewma_backtest: VolatilityBacktestResult,
) -> None:
    """Backtesting metrics should be immutable."""
    with pytest.raises(FrozenInstanceError):
        ewma_backtest.metrics.observations = 0  # type: ignore[misc]


def test_main_prints_model_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_comparison: VolatilityModelComparison,
) -> None:
    """Command-line demonstration should print both models."""
    prices = pd.Series(
        np.linspace(100.0, 110.0, 401),
        index=pd.date_range(
            "2024-01-01",
            periods=401,
            freq="D",
        ),
        name="SPY",
    )
    simple_returns = prices.pct_change(
        fill_method=None
    ).dropna()
    log_returns = np.log(
        prices / prices.shift(1)
    ).dropna()

    market_data = MarketData(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2025-02-05",
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
    )

    monkeypatch.setattr(
        backtest_module,
        "load_market_data",
        lambda **_: market_data,
    )
    monkeypatch.setattr(
        backtest_module,
        "compare_volatility_models",
        lambda *_, **__: model_comparison,
    )

    main()

    output = capsys.readouterr().out

    assert "Volatility model backtesting" in output
    assert "Asset: SPY" in output
    assert "EWMA" in output
    assert "GARCH" in output
    assert "Best model by QLIKE" in output