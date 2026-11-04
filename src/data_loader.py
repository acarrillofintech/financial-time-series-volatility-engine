"""Historical market-data loading and preparation."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_CACHE_DIRECTORY = Path(".yfinance_cache")


@dataclass(frozen=True)
class MarketData:
    """Validated historical prices and financial returns."""

    ticker: str
    start_date: str
    end_date: str
    prices: pd.Series
    simple_returns: pd.Series
    log_returns: pd.Series


def normalize_ticker(ticker: str) -> str:
    """Validate and normalize a market ticker."""
    if not isinstance(ticker, str):
        raise TypeError("ticker must be a string.")

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        raise ValueError("ticker cannot be empty.")

    return normalized_ticker


def normalize_date(
    value: str | date | datetime,
    *,
    name: str,
) -> str:
    """Validate and normalize a date as ISO YYYY-MM-DD."""
    if not isinstance(value, (str, date, datetime)):
        raise TypeError(
            f"{name} must be a date or ISO date string."
        )

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a valid date."
        ) from error

    if pd.isna(timestamp):
        raise ValueError(
            f"{name} must be a valid date."
        )

    return timestamp.strftime("%Y-%m-%d")


def validate_date_range(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> tuple[str, str]:
    """Validate and normalize a historical-data period."""
    normalized_start = normalize_date(
        start_date,
        name="start_date",
    )
    normalized_end = normalize_date(
        end_date,
        name="end_date",
    )

    if (
        pd.Timestamp(normalized_start)
        >= pd.Timestamp(normalized_end)
    ):
        raise ValueError(
            "start_date must be earlier than end_date."
        )

    return normalized_start, normalized_end


def configure_yfinance_cache(
    cache_directory: str | Path = (
        DEFAULT_CACHE_DIRECTORY
    ),
) -> Path:
    """Configure a project-local yfinance cache directory."""
    cache_path = Path(cache_directory)
    cache_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    yf.set_tz_cache_location(
        str(cache_path.resolve())
    )

    return cache_path


def validate_price_series(
    prices: pd.Series,
    *,
    ticker: str,
) -> pd.Series:
    """Validate and clean a historical price series."""
    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    if prices.empty:
        raise ValueError(
            f"No market data was found for {ticker}."
        )

    cleaned_prices = pd.to_numeric(
        prices,
        errors="coerce",
    )
    cleaned_prices = cleaned_prices.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    cleaned_prices = cleaned_prices.dropna()

    if cleaned_prices.empty:
        raise ValueError(
            f"No finite prices were found for {ticker}."
        )

    if np.any(cleaned_prices.to_numpy() <= 0.0):
        raise ValueError(
            "market prices must be strictly positive."
        )

    cleaned_prices = cleaned_prices.sort_index()
    cleaned_prices = cleaned_prices[
        ~cleaned_prices.index.duplicated(
            keep="last"
        )
    ]

    if isinstance(
        cleaned_prices.index,
        pd.DatetimeIndex,
    ):
        if cleaned_prices.index.tz is not None:
            cleaned_prices.index = (
                cleaned_prices.index.tz_localize(None)
            )

        cleaned_prices.index = (
            cleaned_prices.index.normalize()
        )

    cleaned_prices = cleaned_prices.astype(float)
    cleaned_prices.name = ticker

    return cleaned_prices


def extract_close_prices(
    history: pd.DataFrame,
    *,
    ticker: str,
) -> pd.Series:
    """Extract adjusted closing prices from yfinance output."""
    if not isinstance(history, pd.DataFrame):
        raise TypeError(
            "history must be a pandas DataFrame."
        )

    if history.empty:
        raise RuntimeError(
            f"No market data was found for {ticker}."
        )

    if "Close" in history.columns:
        close_prices = history["Close"]
    elif "Adj Close" in history.columns:
        close_prices = history["Adj Close"]
    else:
        raise RuntimeError(
            "Downloaded data does not contain "
            "a closing-price column."
        )

    if isinstance(close_prices, pd.DataFrame):
        if ticker in close_prices.columns:
            close_prices = close_prices[ticker]
        elif close_prices.shape[1] == 1:
            close_prices = close_prices.iloc[:, 0]
        else:
            raise RuntimeError(
                "Unable to identify the requested "
                "ticker column."
            )

    return validate_price_series(
        close_prices,
        ticker=ticker,
    )


def download_adjusted_prices(
    *,
    ticker: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    cache_directory: str | Path = (
        DEFAULT_CACHE_DIRECTORY
    ),
    attempts: int = 3,
) -> pd.Series:
    """Download adjusted daily closing prices."""
    normalized_ticker = normalize_ticker(ticker)
    normalized_start, normalized_end = (
        validate_date_range(
            start_date,
            end_date,
        )
    )

    if (
        not isinstance(attempts, int)
        or isinstance(attempts, bool)
    ):
        raise TypeError(
            "attempts must be an integer."
        )

    if attempts <= 0:
        raise ValueError(
            "attempts must be greater than zero."
        )

    configure_yfinance_cache(
        cache_directory
    )

    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            history = yf.Ticker(
                normalized_ticker
            ).history(
                start=normalized_start,
                end=normalized_end,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )

            if history.empty:
                last_error = RuntimeError(
                    f"No market data was found "
                    f"for {normalized_ticker}."
                )
                continue

            return extract_close_prices(
                history,
                ticker=normalized_ticker,
            )
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Unable to download market data for "
        f"{normalized_ticker} after {attempts} attempts."
    ) from last_error


def calculate_market_returns(
    prices: pd.Series,
    *,
    ticker: str,
) -> tuple[pd.Series, pd.Series]:
    """Calculate aligned simple and logarithmic returns."""
    normalized_ticker = normalize_ticker(ticker)
    validated_prices = validate_price_series(
        prices,
        ticker=normalized_ticker,
    )

    if validated_prices.size < 2:
        raise ValueError(
            "at least two prices are required "
            "to calculate returns."
        )

    simple_returns = (
        validated_prices
        .pct_change(fill_method=None)
        .dropna()
    )

    log_returns = np.log(
        validated_prices
        / validated_prices.shift(1)
    ).dropna()

    if not np.all(
        np.isfinite(simple_returns.to_numpy())
    ):
        raise RuntimeError(
            "simple returns contain non-finite values."
        )

    if not np.all(
        np.isfinite(log_returns.to_numpy())
    ):
        raise RuntimeError(
            "log returns contain non-finite values."
        )

    simple_returns.name = normalized_ticker
    log_returns.name = normalized_ticker

    return simple_returns, log_returns


def load_market_data(
    *,
    ticker: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    cache_directory: str | Path = (
        DEFAULT_CACHE_DIRECTORY
    ),
) -> MarketData:
    """Download prices and prepare an analysis-ready dataset."""
    normalized_ticker = normalize_ticker(ticker)
    normalized_start, normalized_end = (
        validate_date_range(
            start_date,
            end_date,
        )
    )

    prices = download_adjusted_prices(
        ticker=normalized_ticker,
        start_date=normalized_start,
        end_date=normalized_end,
        cache_directory=cache_directory,
    )

    simple_returns, log_returns = (
        calculate_market_returns(
            prices,
            ticker=normalized_ticker,
        )
    )

    return MarketData(
        ticker=normalized_ticker,
        start_date=normalized_start,
        end_date=normalized_end,
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
    )


def main() -> None:
    """Download and summarize SPY market data."""
    market_data = load_market_data(
        ticker="SPY",
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    print("Historical market data")
    print(f"Ticker: {market_data.ticker}")
    print(
        f"Period: "
        f"{market_data.prices.index[0].date()} "
        f"to "
        f"{market_data.prices.index[-1].date()}"
    )
    print(
        f"Price observations: "
        f"{market_data.prices.size:,}"
    )
    print(
        f"Return observations: "
        f"{market_data.simple_returns.size:,}"
    )

    print("\nLatest adjusted prices")
    print(market_data.prices.tail())

    print("\nReturn statistics")
    print(
        f"Average daily return: "
        f"{market_data.simple_returns.mean():.4%}"
    )
    print(
        f"Daily volatility: "
        f"{market_data.simple_returns.std(ddof=1):.4%}"
    )
    print(
        f"Worst daily return: "
        f"{market_data.simple_returns.min():.4%}"
    )
    print(
        f"Best daily return: "
        f"{market_data.simple_returns.max():.4%}"
    )


if __name__ == "__main__":
    main()