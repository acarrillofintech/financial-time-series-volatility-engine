"""Tests for financial time-series visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import src.visualization as visualization
from src.data_loader import MarketData
from src.garch_model import VolatilityForecast
from src.visualization import (
    configure_plot_style,
    generate_analysis_figures,
    main,
    plot_acf_pacf,
    plot_backtest_comparison,
    plot_conditional_volatility,
    plot_price_history,
    plot_return_analysis,
    plot_volatility_forecast,
    save_figure,
    validate_plot_values,
)
from src.volatility_backtest import (
    VolatilityBacktestMetrics,
    VolatilityBacktestResult,
    VolatilityModelComparison,
)


@pytest.fixture
def market_data() -> MarketData:
    """Create reproducible synthetic market data."""
    generator = np.random.default_rng(123)
    returns = generator.normal(
        loc=0.0004,
        scale=0.012,
        size=260,
    )
    prices_array = (
        100.0
        * np.exp(np.cumsum(returns))
    )
    price_index = pd.date_range(
        "2024-01-01",
        periods=260,
        freq="B",
    )
    prices = pd.Series(
        prices_array,
        index=price_index,
        name="SPY",
    )

    simple_returns = prices.pct_change(
        fill_method=None
    ).dropna()
    log_returns = np.log(
        prices / prices.shift(1)
    ).dropna()

    return MarketData(
        ticker="SPY",
        start_date="2024-01-01",
        end_date="2025-01-01",
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
    )


@pytest.fixture
def volatility_forecast() -> VolatilityForecast:
    """Create a deterministic volatility forecast."""
    variances = np.array(
        [0.0001, 0.00011, 0.00012, 0.00013]
    )
    volatilities = np.sqrt(variances)

    return VolatilityForecast(
        steps=4,
        variance_forecasts=variances,
        volatility_forecasts=volatilities,
        annualized_volatility_forecasts=(
            volatilities * np.sqrt(252)
        ),
    )


@pytest.fixture
def model_comparison() -> VolatilityModelComparison:
    """Create deterministic backtesting results."""
    realized = np.array(
        [0.0001, 0.0004, 0.0002, 0.0003]
    )
    ewma_forecasts = np.array(
        [0.0002, 0.0003, 0.00025, 0.00028]
    )
    garch_forecasts = np.array(
        [0.00018, 0.00035, 0.00022, 0.00031]
    )

    ewma_metrics = VolatilityBacktestMetrics(
        observations=4,
        mean_absolute_error=0.0001,
        root_mean_squared_error=0.00012,
        qlike_loss=-7.0,
        mean_realized_variance=float(
            np.mean(realized)
        ),
        mean_forecast_variance=float(
            np.mean(ewma_forecasts)
        ),
        forecast_bias=float(
            np.mean(ewma_forecasts)
            - np.mean(realized)
        ),
        variance_correlation=0.50,
    )
    garch_metrics = VolatilityBacktestMetrics(
        observations=4,
        mean_absolute_error=0.00008,
        root_mean_squared_error=0.00010,
        qlike_loss=-7.2,
        mean_realized_variance=float(
            np.mean(realized)
        ),
        mean_forecast_variance=float(
            np.mean(garch_forecasts)
        ),
        forecast_bias=float(
            np.mean(garch_forecasts)
            - np.mean(realized)
        ),
        variance_correlation=0.60,
    )

    ewma_result = VolatilityBacktestResult(
        model_name="EWMA",
        estimation_window=100,
        realized_variances=realized,
        variance_forecasts=ewma_forecasts,
        volatility_forecasts=np.sqrt(
            ewma_forecasts
        ),
        metrics=ewma_metrics,
    )
    garch_result = VolatilityBacktestResult(
        model_name="GARCH",
        estimation_window=100,
        realized_variances=realized,
        variance_forecasts=garch_forecasts,
        volatility_forecasts=np.sqrt(
            garch_forecasts
        ),
        metrics=garch_metrics,
    )

    return VolatilityModelComparison(
        ewma=ewma_result,
        garch=garch_result,
        best_model_by_qlike="GARCH",
    )


def test_configure_plot_style() -> None:
    """Plot configuration should update Matplotlib settings."""
    configure_plot_style()

    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["axes.facecolor"] == "white"


def test_validate_plot_values() -> None:
    """Valid plotting data should become a float array."""
    result = validate_plot_values(
        [1, 2, 3],
        name="values",
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0, 3.0]),
    )


def test_validate_plot_values_rejects_two_dimensions() -> None:
    """Plot data must be one-dimensional."""
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_plot_values(
            np.ones((2, 2)),
            name="values",
        )


def test_validate_plot_values_rejects_short_sample() -> None:
    """Plot data must meet the minimum length."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        validate_plot_values(
            [1.0],
            name="values",
            minimum_observations=2,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_validate_plot_values_rejects_non_finite_data(
    invalid_value: float,
) -> None:
    """Plot data cannot contain non-finite values."""
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_plot_values(
            [1.0, invalid_value],
            name="values",
        )


def test_plot_price_history(
    market_data: MarketData,
) -> None:
    """Price-history function should return a figure."""
    figure = plot_price_history(
        market_data.prices,
        ticker="SPY",
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert (
        figure.axes[0].get_title()
        == "SPY Adjusted Closing Price"
    )

    plt.close(figure)


def test_plot_price_history_requires_series() -> None:
    """Price history requires a pandas Series."""
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        plot_price_history(
            [100.0, 101.0],  # type: ignore[arg-type]
            ticker="SPY",
        )


def test_plot_return_analysis(
    market_data: MarketData,
) -> None:
    """Return analysis should contain two panels."""
    figure = plot_return_analysis(
        market_data.simple_returns,
        ticker="SPY",
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2
    assert (
        figure.axes[0].get_title()
        == "SPY Daily Simple Returns"
    )

    plt.close(figure)


def test_plot_return_analysis_requires_series() -> None:
    """Return analysis requires a pandas Series."""
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        plot_return_analysis(
            [0.01, -0.01],  # type: ignore[arg-type]
            ticker="SPY",
        )


def test_plot_acf_pacf(
    market_data: MarketData,
) -> None:
    """ACF/PACF function should create two panels."""
    figure = plot_acf_pacf(
        market_data.simple_returns.to_numpy(),
        lags=20,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert "ACF" in figure.axes[0].get_title()
    assert "PACF" in figure.axes[1].get_title()

    plt.close(figure)


def test_plot_acf_pacf_validates_lags(
    market_data: MarketData,
) -> None:
    """Unsupported lag counts should fail."""
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        plot_acf_pacf(
            market_data.simple_returns.to_numpy(),
            lags=200,
        )


def test_plot_conditional_volatility(
    market_data: MarketData,
) -> None:
    """Volatility comparison should return one figure."""
    observations = (
        market_data.simple_returns.size
    )
    ewma = np.full(observations, 0.01)
    garch = np.full(observations, 0.012)

    figure = plot_conditional_volatility(
        market_data.simple_returns.index,
        ewma,
        garch,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert (
        figure.axes[0].get_title()
        == "Estimated Conditional Volatility"
    )
    assert len(figure.axes[0].lines) == 2

    plt.close(figure)


def test_conditional_volatility_requires_matching_lengths(
    market_data: MarketData,
) -> None:
    """Dates and volatility arrays must align."""
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        plot_conditional_volatility(
            market_data.simple_returns.index,
            np.ones(10),
            np.ones(11),
        )


def test_conditional_volatility_rejects_negative_values(
    market_data: MarketData,
) -> None:
    """Volatility cannot be negative."""
    observations = (
        market_data.simple_returns.size
    )
    ewma = np.full(observations, 0.01)
    ewma[0] = -0.01
    garch = np.full(observations, 0.01)

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        plot_conditional_volatility(
            market_data.simple_returns.index,
            ewma,
            garch,
        )


def test_plot_volatility_forecast(
    volatility_forecast: VolatilityForecast,
) -> None:
    """Forecast visualization should contain two panels."""
    figure = plot_volatility_forecast(
        volatility_forecast
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    assert "Daily" in figure.axes[0].get_title()
    assert "Annualized" in figure.axes[1].get_title()

    plt.close(figure)


def test_plot_volatility_forecast_requires_result() -> None:
    """Forecast plot requires a VolatilityForecast."""
    with pytest.raises(
        TypeError,
        match="VolatilityForecast",
    ):
        plot_volatility_forecast(
            "forecast"  # type: ignore[arg-type]
        )


def test_plot_backtest_comparison(
    model_comparison: VolatilityModelComparison,
) -> None:
    """Backtest plot should display three series."""
    dates = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="D",
    )

    figure = plot_backtest_comparison(
        dates,
        model_comparison,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) == 3
    assert "Backtesting" in (
        figure.axes[0].get_title()
    )

    plt.close(figure)


def test_backtest_plot_requires_comparison_object() -> None:
    """Backtest plot requires the comparison data class."""
    with pytest.raises(
        TypeError,
        match="VolatilityModelComparison",
    ):
        plot_backtest_comparison(
            pd.date_range(
                "2025-01-01",
                periods=4,
            ),
            "comparison",  # type: ignore[arg-type]
        )


def test_backtest_plot_requires_matching_dates(
    model_comparison: VolatilityModelComparison,
) -> None:
    """Backtesting dates must align with results."""
    with pytest.raises(
        ValueError,
        match="dates must align",
    ):
        plot_backtest_comparison(
            pd.date_range(
                "2025-01-01",
                periods=3,
            ),
            model_comparison,
        )


def test_save_figure(
    tmp_path: Path,
    market_data: MarketData,
) -> None:
    """A figure should be saved to the requested path."""
    figure = plot_price_history(
        market_data.prices,
        ticker="SPY",
    )
    output_path = (
        tmp_path
        / "figures"
        / "price_history.png"
    )

    result = save_figure(
        figure,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)


def test_save_figure_requires_matplotlib_figure(
    tmp_path: Path,
) -> None:
    """Only Matplotlib Figure objects can be saved."""
    with pytest.raises(
        TypeError,
        match="Matplotlib Figure",
    ):
        save_figure(
            "figure",  # type: ignore[arg-type]
            output_path=tmp_path / "test.png",
        )


def test_generate_analysis_figures(
    tmp_path: Path,
    market_data: MarketData,
) -> None:
    """The complete package should create six PNG files."""
    result = generate_analysis_figures(
        market_data,
        output_directory=tmp_path / "figures",
    )

    expected_names = {
        "price_history",
        "return_analysis",
        "acf_pacf",
        "conditional_volatility",
        "volatility_forecast",
        "backtest_comparison",
    }

    assert set(result) == expected_names

    for path in result.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_generate_figures_requires_market_data(
    tmp_path: Path,
) -> None:
    """Figure generation requires a MarketData object."""
    with pytest.raises(
        TypeError,
        match="MarketData",
    ):
        generate_analysis_figures(
            "market data",  # type: ignore[arg-type]
            output_directory=tmp_path,
        )


def test_main_prints_saved_paths(
    tmp_path: Path,
    market_data: MarketData,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Command-line entry point should report generated figures."""
    fake_paths = {
        "price_history": (
            tmp_path / "price_history.png"
        ),
        "return_analysis": (
            tmp_path / "return_analysis.png"
        ),
    }

    monkeypatch.setattr(
        visualization,
        "load_market_data",
        lambda **_: market_data,
    )
    monkeypatch.setattr(
        visualization,
        "generate_analysis_figures",
        lambda *_args, **_kwargs: fake_paths,
    )

    main()

    output = capsys.readouterr().out

    assert "Financial time-series visualizations" in output
    assert "Asset: SPY" in output
    assert "Figures generated: 2" in output
    assert "price_history" in output
    assert "return_analysis" in output