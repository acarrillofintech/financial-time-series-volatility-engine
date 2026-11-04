"""Tests for historical market-data loading."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data_loader as data_loader
from src.data_loader import (
    MarketData,
    calculate_market_returns,
    configure_yfinance_cache,
    download_adjusted_prices,
    extract_close_prices,
    load_market_data,
    main,
    normalize_date,
    normalize_ticker,
    validate_date_range,
    validate_price_series,
)


def sample_prices() -> pd.Series:
    """Create a small valid market-price series."""
    return pd.Series(
        [100.0, 102.0, 101.0, 105.0],
        index=pd.date_range(
            "2025-01-01",
            periods=4,
            freq="D",
        ),
        name="SPY",
    )


def sample_history() -> pd.DataFrame:
    """Create yfinance-style historical data."""
    return pd.DataFrame(
        {
            "Open": [99.0, 101.0, 100.0, 104.0],
            "High": [101.0, 103.0, 102.0, 106.0],
            "Low": [98.0, 100.0, 99.0, 103.0],
            "Close": [100.0, 102.0, 101.0, 105.0],
            "Volume": [1000, 1100, 1200, 1300],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=4,
            freq="D",
        ),
    )


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("spy", "SPY"),
        (" SPY ", "SPY"),
        ("qqq", "QQQ"),
        ("brk-b", "BRK-B"),
    ],
)
def test_normalize_ticker(
    ticker: str,
    expected: str,
) -> None:
    """Ticker names should be stripped and uppercased."""
    assert normalize_ticker(ticker) == expected


def test_normalize_ticker_rejects_empty_value() -> None:
    """Ticker cannot be empty."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        normalize_ticker("   ")


def test_normalize_ticker_rejects_non_string() -> None:
    """Ticker must be textual."""
    with pytest.raises(
        TypeError,
        match="string",
    ):
        normalize_ticker(
            123  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "input_date",
    [
        "2025-01-15",
        date(2025, 1, 15),
        datetime(2025, 1, 15, 10, 30),
        pd.Timestamp("2025-01-15"),
    ],
)
def test_normalize_date(
    input_date: object,
) -> None:
    """Supported date values should become ISO dates."""
    result = normalize_date(
        input_date,  # type: ignore[arg-type]
        name="start_date",
    )

    assert result == "2025-01-15"


def test_normalize_date_rejects_invalid_string() -> None:
    """Invalid date text should be rejected."""
    with pytest.raises(
        ValueError,
        match="valid date",
    ):
        normalize_date(
            "not-a-date",
            name="start_date",
        )


def test_normalize_date_rejects_invalid_type() -> None:
    """Unsupported date types should fail."""
    with pytest.raises(
        TypeError,
        match="date or ISO",
    ):
        normalize_date(
            123,  # type: ignore[arg-type]
            name="start_date",
        )


def test_validate_date_range() -> None:
    """A valid date range should be normalized."""
    result = validate_date_range(
        "2025-01-01",
        "2025-12-31",
    )

    assert result == (
        "2025-01-01",
        "2025-12-31",
    )


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        ("2025-01-01", "2025-01-01"),
        ("2025-12-31", "2025-01-01"),
    ],
)
def test_validate_date_range_requires_chronological_order(
    start_date: str,
    end_date: str,
) -> None:
    """Start date must precede end date."""
    with pytest.raises(
        ValueError,
        match="earlier",
    ):
        validate_date_range(
            start_date,
            end_date,
        )


def test_configure_yfinance_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache configuration should create and register a directory."""
    recorded_paths: list[str] = []

    monkeypatch.setattr(
        data_loader.yf,
        "set_tz_cache_location",
        recorded_paths.append,
    )

    cache_path = configure_yfinance_cache(
        tmp_path / "market-cache"
    )

    assert cache_path.exists()
    assert cache_path.is_dir()
    assert recorded_paths == [
        str(cache_path.resolve())
    ]


def test_validate_price_series() -> None:
    """Valid prices should be returned as floats."""
    result = validate_price_series(
        sample_prices(),
        ticker="SPY",
    )

    assert isinstance(result, pd.Series)
    assert result.dtype == float
    assert result.name == "SPY"
    assert result.size == 4


def test_validate_price_series_sorts_dates() -> None:
    """Market prices should be ordered chronologically."""
    prices = sample_prices().sort_index(
        ascending=False
    )

    result = validate_price_series(
        prices,
        ticker="SPY",
    )

    assert result.index.is_monotonic_increasing


def test_validate_price_series_removes_duplicate_dates() -> None:
    """Duplicate index values should keep the last observation."""
    index = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
        ]
    )
    prices = pd.Series(
        [100.0, 101.0, 102.0],
        index=index,
    )

    result = validate_price_series(
        prices,
        ticker="SPY",
    )

    assert result.size == 2
    assert result.iloc[0] == 101.0


def test_validate_price_series_removes_timezone() -> None:
    """Daily market dates should become timezone-naive."""
    prices = pd.Series(
        [100.0, 101.0],
        index=pd.date_range(
            "2025-01-01",
            periods=2,
            tz="America/New_York",
        ),
    )

    result = validate_price_series(
        prices,
        ticker="SPY",
    )

    assert result.index.tz is None


def test_validate_price_series_drops_non_finite_values() -> None:
    """NaN and infinite prices should be removed."""
    prices = pd.Series(
        [100.0, np.nan, np.inf, 102.0],
        index=pd.date_range(
            "2025-01-01",
            periods=4,
        ),
    )

    result = validate_price_series(
        prices,
        ticker="SPY",
    )

    np.testing.assert_array_equal(
        result.to_numpy(),
        np.array([100.0, 102.0]),
    )


def test_validate_price_series_rejects_empty_series() -> None:
    """An empty market series should fail."""
    with pytest.raises(
        ValueError,
        match="No market data",
    ):
        validate_price_series(
            pd.Series(dtype=float),
            ticker="SPY",
        )


def test_validate_price_series_rejects_all_invalid_values() -> None:
    """A series without finite prices should fail."""
    prices = pd.Series(
        [np.nan, np.inf],
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="No finite prices",
    ):
        validate_price_series(
            prices,
            ticker="SPY",
        )


@pytest.mark.parametrize(
    "invalid_price",
    [0.0, -1.0],
)
def test_validate_price_series_requires_positive_prices(
    invalid_price: float,
) -> None:
    """Market prices must be positive."""
    prices = pd.Series(
        [100.0, invalid_price],
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        validate_price_series(
            prices,
            ticker="SPY",
        )


def test_validate_price_series_requires_pandas_series() -> None:
    """Price input must be a pandas Series."""
    with pytest.raises(
        TypeError,
        match="pandas Series",
    ):
        validate_price_series(
            [100.0, 101.0],  # type: ignore[arg-type]
            ticker="SPY",
        )


def test_extract_close_prices() -> None:
    """The Close column should be extracted."""
    result = extract_close_prices(
        sample_history(),
        ticker="SPY",
    )

    np.testing.assert_array_equal(
        result.to_numpy(),
        np.array([100.0, 102.0, 101.0, 105.0]),
    )


def test_extract_adjusted_close_fallback() -> None:
    """Adj Close should be used when Close is absent."""
    history = pd.DataFrame(
        {
            "Adj Close": [100.0, 101.0],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    result = extract_close_prices(
        history,
        ticker="SPY",
    )

    np.testing.assert_array_equal(
        result.to_numpy(),
        np.array([100.0, 101.0]),
    )


def test_extract_close_from_multiindex() -> None:
    """MultiIndex download output should support one ticker."""
    columns = pd.MultiIndex.from_product(
        [["Close", "Open"], ["SPY"]]
    )
    history = pd.DataFrame(
        [
            [100.0, 99.0],
            [101.0, 100.0],
        ],
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
        columns=columns,
    )

    result = extract_close_prices(
        history,
        ticker="SPY",
    )

    np.testing.assert_array_equal(
        result.to_numpy(),
        np.array([100.0, 101.0]),
    )


def test_extract_close_rejects_empty_history() -> None:
    """An empty download should fail clearly."""
    with pytest.raises(
        RuntimeError,
        match="No market data",
    ):
        extract_close_prices(
            pd.DataFrame(),
            ticker="SPY",
        )


def test_extract_close_requires_dataframe() -> None:
    """Downloaded history must be a DataFrame."""
    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        extract_close_prices(
            [],  # type: ignore[arg-type]
            ticker="SPY",
        )


def test_extract_close_requires_price_column() -> None:
    """Downloaded data must contain a closing-price column."""
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=2,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="closing-price column",
    ):
        extract_close_prices(
            history,
            ticker="SPY",
        )


def test_calculate_market_returns_known_values() -> None:
    """Simple and log returns should match known values."""
    prices = sample_prices()

    simple_returns, log_returns = (
        calculate_market_returns(
            prices,
            ticker="spy",
        )
    )

    expected_simple = np.array(
        [
            102.0 / 100.0 - 1.0,
            101.0 / 102.0 - 1.0,
            105.0 / 101.0 - 1.0,
        ]
    )
    expected_log = np.log(
        np.array(
            [
                102.0 / 100.0,
                101.0 / 102.0,
                105.0 / 101.0,
            ]
        )
    )

    np.testing.assert_allclose(
        simple_returns.to_numpy(),
        expected_simple,
    )
    np.testing.assert_allclose(
        log_returns.to_numpy(),
        expected_log,
    )
    assert simple_returns.name == "SPY"
    assert log_returns.name == "SPY"


def test_returns_are_aligned_with_later_prices() -> None:
    """Return dates should begin with the second price date."""
    prices = sample_prices()

    simple_returns, log_returns = (
        calculate_market_returns(
            prices,
            ticker="SPY",
        )
    )

    expected_index = prices.index[1:]

    pd.testing.assert_index_equal(
        simple_returns.index,
        expected_index,
    )
    pd.testing.assert_index_equal(
        log_returns.index,
        expected_index,
    )


def test_calculate_returns_requires_two_prices() -> None:
    """At least two prices are needed for returns."""
    prices = pd.Series(
        [100.0],
        index=pd.to_datetime(["2025-01-01"]),
    )

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        calculate_market_returns(
            prices,
            ticker="SPY",
        )


def test_download_adjusted_prices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloader should return validated close prices."""
    history = sample_history()

    class FakeTicker:
        """Minimal yfinance ticker replacement."""

        def __init__(self, ticker: str) -> None:
            assert ticker == "SPY"

        def history(self, **kwargs: object) -> pd.DataFrame:
            assert kwargs["start"] == "2025-01-01"
            assert kwargs["end"] == "2025-02-01"
            assert kwargs["auto_adjust"] is True
            return history

    monkeypatch.setattr(
        data_loader.yf,
        "Ticker",
        FakeTicker,
    )
    monkeypatch.setattr(
        data_loader.yf,
        "set_tz_cache_location",
        lambda _: None,
    )

    result = download_adjusted_prices(
        ticker="spy",
        start_date="2025-01-01",
        end_date="2025-02-01",
        cache_directory=tmp_path / "cache",
    )

    assert result.name == "SPY"
    assert result.size == 4


def test_download_retries_after_empty_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloader should retry transient empty responses."""
    history = sample_history()
    call_count = 0

    class FakeTicker:
        """Ticker that returns empty data once."""

        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, **kwargs: object) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return pd.DataFrame()

            return history

    monkeypatch.setattr(
        data_loader.yf,
        "Ticker",
        FakeTicker,
    )
    monkeypatch.setattr(
        data_loader.yf,
        "set_tz_cache_location",
        lambda _: None,
    )

    result = download_adjusted_prices(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-02-01",
        cache_directory=tmp_path / "cache",
        attempts=2,
    )

    assert result.size == 4
    assert call_count == 2


def test_download_fails_after_all_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent empty downloads should raise RuntimeError."""
    class EmptyTicker:
        """Ticker that never returns market data."""

        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        data_loader.yf,
        "Ticker",
        EmptyTicker,
    )
    monkeypatch.setattr(
        data_loader.yf,
        "set_tz_cache_location",
        lambda _: None,
    )

    with pytest.raises(
        RuntimeError,
        match="after 2 attempts",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-02-01",
            cache_directory=tmp_path / "cache",
            attempts=2,
        )


@pytest.mark.parametrize(
    "attempts",
    [0, -1],
)
def test_download_requires_positive_attempt_count(
    attempts: int,
) -> None:
    """Download attempt count must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-02-01",
            attempts=attempts,
        )


@pytest.mark.parametrize(
    "attempts",
    [1.0, True],
)
def test_download_requires_integer_attempt_count(
    attempts: object,
) -> None:
    """Download attempt count must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        download_adjusted_prices(
            ticker="SPY",
            start_date="2025-01-01",
            end_date="2025-02-01",
            attempts=attempts,  # type: ignore[arg-type]
        )


def test_load_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined loader should return an analysis-ready object."""
    prices = sample_prices()

    monkeypatch.setattr(
        data_loader,
        "download_adjusted_prices",
        lambda **_: prices,
    )

    result = load_market_data(
        ticker="spy",
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    assert isinstance(result, MarketData)
    assert result.ticker == "SPY"
    assert result.start_date == "2025-01-01"
    assert result.end_date == "2025-02-01"
    assert result.prices.size == 4
    assert result.simple_returns.size == 3
    assert result.log_returns.size == 3


def test_market_data_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Market-data attributes should be immutable."""
    monkeypatch.setattr(
        data_loader,
        "download_adjusted_prices",
        lambda **_: sample_prices(),
    )

    result = load_market_data(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-02-01",
    )

    with pytest.raises(FrozenInstanceError):
        result.ticker = "QQQ"  # type: ignore[misc]


def test_main_prints_market_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command-line demonstration should print key statistics."""
    prices = sample_prices()
    simple_returns, log_returns = (
        calculate_market_returns(
            prices,
            ticker="SPY",
        )
    )

    market_data = MarketData(
        ticker="SPY",
        start_date="2025-01-01",
        end_date="2025-02-01",
        prices=prices,
        simple_returns=simple_returns,
        log_returns=log_returns,
    )

    monkeypatch.setattr(
        data_loader,
        "load_market_data",
        lambda **_: market_data,
    )

    main()

    output = capsys.readouterr().out

    assert "Historical market data" in output
    assert "Ticker: SPY" in output
    assert "Price observations" in output
    assert "Return observations" in output
    assert "Average daily return" in output
    assert "Daily volatility" in output