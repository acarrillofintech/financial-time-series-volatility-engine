"""Professional visualizations for time-series and volatility analysis."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.data_loader import (
    MarketData,
    load_market_data,
)
from src.ewma_volatility import (
    calculate_ewma_volatility,
)
from src.garch_model import (
    FittedVolatilityModel,
    VolatilityForecast,
    fit_volatility_model,
    forecast_volatility_model,
)
from src.stationarity import (
    calculate_correlation_diagnostics,
)
from src.volatility_backtest import (
    VolatilityModelComparison,
    compare_volatility_models,
)


DEFAULT_FIGURE_DIRECTORY = Path(
    "results/figures"
)
PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#DC2626"
TERTIARY_COLOR = "#059669"
NEUTRAL_COLOR = "#475569"
GRID_COLOR = "#CBD5E1"


def configure_plot_style() -> None:
    """Configure a consistent professional plotting style."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#94A3B8",
            "axes.labelcolor": "#1E293B",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.35,
            "legend.frameon": False,
            "font.family": "sans-serif",
        }
    )


def validate_plot_values(
    values: Iterable[float] | np.ndarray,
    *,
    name: str,
    minimum_observations: int = 2,
) -> np.ndarray:
    """Validate a one-dimensional finite plotting array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.size < minimum_observations:
        raise ValueError(
            f"{name} does not contain enough observations."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return array


def plot_price_history(
    prices: pd.Series,
    *,
    ticker: str,
) -> Figure:
    """Plot the adjusted historical price series."""
    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    validated_prices = validate_plot_values(
        prices.to_numpy(),
        name="prices",
    )

    if validated_prices.size != prices.index.size:
        raise ValueError(
            "price values and dates must align."
        )

    configure_plot_style()

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.plot(
        prices.index,
        validated_prices,
        color=PRIMARY_COLOR,
        linewidth=1.8,
    )
    axis.fill_between(
        prices.index,
        validated_prices,
        np.min(validated_prices),
        color=PRIMARY_COLOR,
        alpha=0.08,
    )

    axis.set_title(
        f"{ticker} Adjusted Closing Price"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Adjusted price ($)")
    axis.grid(True, alpha=0.30)

    figure.tight_layout()

    return figure


def plot_return_analysis(
    returns: pd.Series,
    *,
    ticker: str,
) -> Figure:
    """Plot daily returns and their empirical distribution."""
    if not isinstance(returns, pd.Series):
        raise TypeError(
            "returns must be a pandas Series."
        )

    validated_returns = validate_plot_values(
        returns.to_numpy(),
        name="returns",
    )

    configure_plot_style()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 9),
        gridspec_kw={
            "height_ratios": [1.3, 1.0]
        },
    )

    axes[0].plot(
        returns.index,
        validated_returns * 100.0,
        color=NEUTRAL_COLOR,
        linewidth=0.9,
    )
    axes[0].axhline(
        0.0,
        color="#0F172A",
        linewidth=0.8,
    )
    axes[0].set_title(
        f"{ticker} Daily Simple Returns"
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Return (%)")

    sns.histplot(
        validated_returns * 100.0,
        bins=50,
        kde=True,
        color=PRIMARY_COLOR,
        alpha=0.65,
        ax=axes[1],
    )
    axes[1].axvline(
        np.mean(validated_returns) * 100.0,
        color=SECONDARY_COLOR,
        linestyle="--",
        linewidth=1.8,
        label="Sample mean",
    )
    axes[1].set_title(
        "Empirical Return Distribution"
    )
    axes[1].set_xlabel("Daily return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    figure.tight_layout()

    return figure


def plot_acf_pacf(
    returns: Iterable[float] | np.ndarray,
    *,
    lags: int = 30,
) -> Figure:
    """Plot autocorrelation and partial autocorrelation."""
    validated_returns = validate_plot_values(
        returns,
        name="returns",
        minimum_observations=10,
    )

    diagnostics = (
        calculate_correlation_diagnostics(
            validated_returns,
            lags=lags,
        )
    )

    confidence_limit = (
        1.96 / np.sqrt(validated_returns.size)
    )

    configure_plot_style()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].vlines(
        diagnostics.lags,
        0.0,
        diagnostics.autocorrelation,
        color=PRIMARY_COLOR,
        linewidth=2.0,
    )
    axes[0].scatter(
        diagnostics.lags,
        diagnostics.autocorrelation,
        color=PRIMARY_COLOR,
        s=24,
    )
    axes[0].axhline(
        0.0,
        color="#0F172A",
        linewidth=0.8,
    )
    axes[0].axhspan(
        -confidence_limit,
        confidence_limit,
        color=PRIMARY_COLOR,
        alpha=0.12,
    )
    axes[0].set_title(
        "Autocorrelation Function (ACF)"
    )
    axes[0].set_ylabel("Correlation")

    axes[1].vlines(
        diagnostics.lags,
        0.0,
        diagnostics.partial_autocorrelation,
        color=TERTIARY_COLOR,
        linewidth=2.0,
    )
    axes[1].scatter(
        diagnostics.lags,
        diagnostics.partial_autocorrelation,
        color=TERTIARY_COLOR,
        s=24,
    )
    axes[1].axhline(
        0.0,
        color="#0F172A",
        linewidth=0.8,
    )
    axes[1].axhspan(
        -confidence_limit,
        confidence_limit,
        color=TERTIARY_COLOR,
        alpha=0.12,
    )
    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)"
    )
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Partial correlation")

    figure.tight_layout()

    return figure


def plot_conditional_volatility(
    dates: pd.Index,
    ewma_volatility: Iterable[float] | np.ndarray,
    garch_volatility: Iterable[float] | np.ndarray,
) -> Figure:
    """Compare EWMA and GARCH conditional volatility."""
    ewma_values = validate_plot_values(
        ewma_volatility,
        name="ewma_volatility",
    )
    garch_values = validate_plot_values(
        garch_volatility,
        name="garch_volatility",
    )

    if (
        ewma_values.size != garch_values.size
        or ewma_values.size != len(dates)
    ):
        raise ValueError(
            "dates and volatility arrays "
            "must have the same length."
        )

    if (
        np.any(ewma_values < 0.0)
        or np.any(garch_values < 0.0)
    ):
        raise ValueError(
            "volatility values cannot be negative."
        )

    annualization_factor = np.sqrt(252.0)

    configure_plot_style()

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    axis.plot(
        dates,
        ewma_values
        * annualization_factor
        * 100.0,
        color=PRIMARY_COLOR,
        linewidth=1.4,
        label="EWMA",
    )
    axis.plot(
        dates,
        garch_values
        * annualization_factor
        * 100.0,
        color=SECONDARY_COLOR,
        linewidth=1.3,
        alpha=0.90,
        label="GARCH(1,1)",
    )

    axis.set_title(
        "Estimated Conditional Volatility"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel(
        "Annualized volatility (%)"
    )
    axis.legend()

    figure.tight_layout()

    return figure


def plot_volatility_forecast(
    forecast: VolatilityForecast,
) -> Figure:
    """Plot daily and annualized GARCH forecasts."""
    if not isinstance(
        forecast,
        VolatilityForecast,
    ):
        raise TypeError(
            "forecast must be a VolatilityForecast."
        )

    horizons = np.arange(
        1,
        forecast.steps + 1,
    )

    configure_plot_style()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        horizons,
        forecast.volatility_forecasts * 100.0,
        color=PRIMARY_COLOR,
        marker="o",
        linewidth=2.0,
    )
    axes[0].set_title(
        "GARCH Daily Volatility Forecast"
    )
    axes[0].set_ylabel(
        "Daily volatility (%)"
    )

    axes[1].plot(
        horizons,
        forecast.annualized_volatility_forecasts
        * 100.0,
        color=SECONDARY_COLOR,
        marker="o",
        linewidth=2.0,
    )
    axes[1].set_title(
        "GARCH Annualized Volatility Forecast"
    )
    axes[1].set_xlabel("Forecast horizon (days)")
    axes[1].set_ylabel(
        "Annualized volatility (%)"
    )

    axes[1].set_xticks(horizons)

    figure.tight_layout()

    return figure


def plot_backtest_comparison(
    dates: pd.Index,
    comparison: VolatilityModelComparison,
) -> Figure:
    """Plot realized volatility against model forecasts."""
    if not isinstance(
        comparison,
        VolatilityModelComparison,
    ):
        raise TypeError(
            "comparison must be a "
            "VolatilityModelComparison."
        )

    observations = (
        comparison.ewma.metrics.observations
    )

    if (
        comparison.garch.metrics.observations
        != observations
    ):
        raise ValueError(
            "EWMA and GARCH backtests must align."
        )

    if len(dates) != observations:
        raise ValueError(
            "dates must align with backtest results."
        )

    realized_volatility = np.sqrt(
        comparison.ewma.realized_variances
    )

    configure_plot_style()

    figure, axis = plt.subplots(
        figsize=(13, 7)
    )

    axis.plot(
        dates,
        realized_volatility * 100.0,
        color=NEUTRAL_COLOR,
        linewidth=0.8,
        alpha=0.55,
        label="Absolute realized return",
    )
    axis.plot(
        dates,
        comparison.ewma.volatility_forecasts
        * 100.0,
        color=PRIMARY_COLOR,
        linewidth=1.5,
        label="EWMA forecast",
    )
    axis.plot(
        dates,
        comparison.garch.volatility_forecasts
        * 100.0,
        color=SECONDARY_COLOR,
        linewidth=1.4,
        label="GARCH forecast",
    )

    axis.set_title(
        "Out-of-Sample Volatility Backtesting"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Daily volatility (%)")
    axis.legend(
        loc="upper right",
    )

    figure.tight_layout()

    return figure


def save_figure(
    figure: Figure,
    *,
    output_path: str | Path,
) -> Path:
    """Save one Matplotlib figure."""
    if not isinstance(figure, Figure):
        raise TypeError(
            "figure must be a Matplotlib Figure."
        )

    normalized_path = Path(output_path)
    normalized_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        normalized_path,
        dpi=180,
        bbox_inches="tight",
    )

    return normalized_path


def generate_analysis_figures(
    market_data: MarketData,
    *,
    output_directory: str | Path = (
        DEFAULT_FIGURE_DIRECTORY
    ),
) -> dict[str, Path]:
    """Generate and save the complete visualization package."""
    if not isinstance(market_data, MarketData):
        raise TypeError(
            "market_data must be a MarketData object."
        )

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    returns = (
        market_data.simple_returns.to_numpy()
    )

    ewma_result = calculate_ewma_volatility(
        returns,
        decay_factor=0.94,
    )
    garch_result: FittedVolatilityModel = (
        fit_volatility_model(
            returns,
            model_type="GARCH",
            p=1,
            q=1,
            distribution="normal",
            mean_model="zero",
        )
    )
    volatility_forecast = (
        forecast_volatility_model(
            garch_result,
            steps=20,
        )
    )

    estimation_window = min(
        500,
        max(
            100,
            returns.size // 2,
        ),
    )

    comparison = compare_volatility_models(
        returns,
        estimation_window=estimation_window,
        decay_factor=0.94,
        refit_frequency=50,
    )

    out_of_sample_dates = (
        market_data.simple_returns.index[
            estimation_window:
        ]
    )

    figures = {
        "price_history": plot_price_history(
            market_data.prices,
            ticker=market_data.ticker,
        ),
        "return_analysis": plot_return_analysis(
            market_data.simple_returns,
            ticker=market_data.ticker,
        ),
        "acf_pacf": plot_acf_pacf(
            returns,
            lags=30,
        ),
        "conditional_volatility": (
            plot_conditional_volatility(
                market_data.simple_returns.index,
                ewma_result.conditional_volatilities,
                garch_result.conditional_volatilities,
            )
        ),
        "volatility_forecast": (
            plot_volatility_forecast(
                volatility_forecast
            )
        ),
        "backtest_comparison": (
            plot_backtest_comparison(
                out_of_sample_dates,
                comparison,
            )
        ),
    }

    saved_paths: dict[str, Path] = {}

    for name, figure in figures.items():
        saved_paths[name] = save_figure(
            figure,
            output_path=(
                output_path / f"{name}.png"
            ),
        )
        plt.close(figure)

    return saved_paths


def main() -> None:
    """Generate all SPY analytical figures."""
    market_data = load_market_data(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    saved_paths = generate_analysis_figures(
        market_data
    )

    print("Financial time-series visualizations")
    print(f"Asset: {market_data.ticker}")
    print(
        f"Figures generated: "
        f"{len(saved_paths)}"
    )

    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()