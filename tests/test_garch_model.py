"""Tests for ARCH and GARCH volatility models."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.garch_model import (
    FittedVolatilityModel,
    GarchSimulation,
    VolatilityForecast,
    calculate_persistence,
    calculate_unconditional_variance,
    fit_volatility_model,
    forecast_volatility_model,
    main,
    simulate_garch_process,
    validate_distribution,
    validate_forecast_steps,
    validate_mean_model,
    validate_model_orders,
    validate_model_type,
)


@pytest.fixture(scope="module")
def simulated_garch() -> GarchSimulation:
    """Create a reproducible GARCH(1,1) sample."""
    return simulate_garch_process(
        observations=1200,
        omega=0.000001,
        alpha=0.08,
        beta=0.90,
        seed=123,
    )


@pytest.fixture(scope="module")
def fitted_garch(
    simulated_garch: GarchSimulation,
) -> FittedVolatilityModel:
    """Fit one reusable GARCH model."""
    return fit_volatility_model(
        simulated_garch.returns,
        model_type="GARCH",
        p=1,
        q=1,
        distribution="normal",
        mean_model="zero",
    )


@pytest.fixture
def manual_model() -> FittedVolatilityModel:
    """Create a deterministic wrapper for formula tests."""
    return FittedVolatilityModel(
        model_type="GARCH",
        p=1,
        q=1,
        distribution="normal",
        observations=100,
        log_likelihood=-100.0,
        aic=206.0,
        bic=215.0,
        parameters={
            "omega": 0.01,
            "alpha[1]": 0.08,
            "beta[1]": 0.90,
        },
        conditional_volatilities=np.full(
            100,
            0.01,
        ),
        standardized_residuals=np.zeros(100),
        result=None,
    )


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("ARCH", "ARCH"),
        ("arch", "ARCH"),
        (" GARCH ", "GARCH"),
        ("garch", "GARCH"),
    ],
)
def test_validate_model_type_accepts_supported_models(
    model_type: str,
    expected: str,
) -> None:
    """Supported model names should be normalized."""
    assert validate_model_type(model_type) == expected


def test_validate_model_type_rejects_non_string() -> None:
    """Model type must be textual."""
    with pytest.raises(
        TypeError,
        match="string",
    ):
        validate_model_type(
            1  # type: ignore[arg-type]
        )


def test_validate_model_type_rejects_unknown_model() -> None:
    """Unsupported volatility models should be rejected."""
    with pytest.raises(
        ValueError,
        match="ARCH.*GARCH",
    ):
        validate_model_type("EWMA")


@pytest.mark.parametrize(
    ("model_type", "p", "q"),
    [
        ("ARCH", 1, 0),
        ("ARCH", 5, 0),
        ("GARCH", 1, 1),
        ("GARCH", 2, 3),
    ],
)
def test_validate_model_orders_accepts_valid_values(
    model_type: str,
    p: int,
    q: int,
) -> None:
    """Valid ARCH and GARCH orders should be accepted."""
    assert validate_model_orders(
        model_type=model_type,
        p=p,
        q=q,
    ) == (p, q)


@pytest.mark.parametrize(
    "p",
    [0, -1],
)
def test_validate_model_orders_requires_positive_p(
    p: int,
) -> None:
    """The ARCH order must be at least one."""
    with pytest.raises(
        ValueError,
        match="p must be at least one",
    ):
        validate_model_orders(
            model_type="GARCH",
            p=p,
            q=1,
        )


def test_validate_model_orders_rejects_negative_q() -> None:
    """The GARCH order cannot be negative."""
    with pytest.raises(
        ValueError,
        match="q cannot be negative",
    ):
        validate_model_orders(
            model_type="GARCH",
            p=1,
            q=-1,
        )


def test_arch_requires_zero_q() -> None:
    """A pure ARCH model must not contain GARCH lags."""
    with pytest.raises(
        ValueError,
        match="q equal to zero",
    ):
        validate_model_orders(
            model_type="ARCH",
            p=1,
            q=1,
        )


def test_garch_requires_positive_q() -> None:
    """A GARCH model must include a variance lag."""
    with pytest.raises(
        ValueError,
        match="q at least one",
    ):
        validate_model_orders(
            model_type="GARCH",
            p=1,
            q=0,
        )


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (1.0, 1),
        (1, 1.0),
        (True, 1),
        (1, True),
    ],
)
def test_model_orders_reject_non_integer_values(
    p: object,
    q: object,
) -> None:
    """Lag orders must be integers."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_model_orders(
            model_type="GARCH",
            p=p,  # type: ignore[arg-type]
            q=q,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "distribution",
    ["normal", "t", "skewt", "ged"],
)
def test_validate_distribution_accepts_supported_values(
    distribution: str,
) -> None:
    """Supported innovation distributions should be accepted."""
    assert (
        validate_distribution(distribution)
        == distribution
    )


def test_validate_distribution_normalizes_case() -> None:
    """Distribution names should be case-insensitive."""
    assert validate_distribution(" NORMAL ") == "normal"


def test_validate_distribution_rejects_unknown_value() -> None:
    """Unknown distributions should be rejected."""
    with pytest.raises(
        ValueError,
        match="distribution must be one of",
    ):
        validate_distribution("uniform")


def test_validate_distribution_rejects_non_string() -> None:
    """Distribution name must be textual."""
    with pytest.raises(
        TypeError,
        match="string",
    ):
        validate_distribution(
            1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("mean_model", "expected"),
    [
        ("zero", "Zero"),
        ("ZERO", "Zero"),
        ("constant", "Constant"),
        (" Constant ", "Constant"),
    ],
)
def test_validate_mean_model(
    mean_model: str,
    expected: str,
) -> None:
    """Supported mean specifications should be normalized."""
    assert validate_mean_model(mean_model) == expected


def test_validate_mean_model_rejects_unknown_value() -> None:
    """Unsupported mean specifications should fail."""
    with pytest.raises(
        ValueError,
        match="zero.*constant",
    ):
        validate_mean_model("AR")


def test_validate_mean_model_rejects_non_string() -> None:
    """Mean-model name must be textual."""
    with pytest.raises(
        TypeError,
        match="string",
    ):
        validate_mean_model(
            1  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "steps",
    [1, 5, 20],
)
def test_validate_forecast_steps_accepts_positive_values(
    steps: int,
) -> None:
    """Positive forecast horizons should be accepted."""
    assert validate_forecast_steps(steps) == steps


@pytest.mark.parametrize(
    "steps",
    [0, -1],
)
def test_validate_forecast_steps_rejects_non_positive_values(
    steps: int,
) -> None:
    """Forecast horizon must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_forecast_steps(steps)


@pytest.mark.parametrize(
    "steps",
    [5.0, "5", True],
)
def test_validate_forecast_steps_rejects_non_integer_values(
    steps: object,
) -> None:
    """Forecast horizon must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        validate_forecast_steps(
            steps  # type: ignore[arg-type]
        )


def test_simulation_structure(
    simulated_garch: GarchSimulation,
) -> None:
    """Simulation should return aligned finite arrays."""
    result = simulated_garch

    assert isinstance(result, GarchSimulation)
    assert result.returns.shape == (1200,)
    assert result.conditional_variances.shape == (1200,)
    assert result.conditional_volatilities.shape == (1200,)
    assert np.all(np.isfinite(result.returns))
    assert np.all(
        result.conditional_variances > 0.0
    )


def test_simulation_volatility_matches_variance(
    simulated_garch: GarchSimulation,
) -> None:
    """Simulated volatility should equal square-root variance."""
    np.testing.assert_allclose(
        simulated_garch.conditional_volatilities,
        np.sqrt(
            simulated_garch.conditional_variances
        ),
    )


def test_initial_variance_equals_unconditional_variance() -> None:
    """Simulation should initialize at long-run variance."""
    result = simulate_garch_process(
        observations=10,
        omega=0.000001,
        alpha=0.08,
        beta=0.90,
        seed=1,
    )

    expected_variance = (
        0.000001
        / (1.0 - 0.08 - 0.90)
    )

    assert (
        result.conditional_variances[0]
        == pytest.approx(expected_variance)
    )


def test_simulation_variance_recursion() -> None:
    """Every simulated variance should follow GARCH(1,1)."""
    omega = 0.000001
    alpha = 0.08
    beta = 0.90

    result = simulate_garch_process(
        observations=100,
        omega=omega,
        alpha=alpha,
        beta=beta,
        seed=5,
    )

    expected = (
        omega
        + alpha * result.returns[:-1] ** 2
        + beta
        * result.conditional_variances[:-1]
    )

    np.testing.assert_allclose(
        result.conditional_variances[1:],
        expected,
    )


def test_simulation_is_reproducible() -> None:
    """The same random seed should reproduce the process."""
    first = simulate_garch_process(seed=99)
    second = simulate_garch_process(seed=99)

    np.testing.assert_array_equal(
        first.returns,
        second.returns,
    )
    np.testing.assert_array_equal(
        first.conditional_variances,
        second.conditional_variances,
    )


def test_different_seeds_change_simulation() -> None:
    """Different seeds should produce different returns."""
    first = simulate_garch_process(seed=1)
    second = simulate_garch_process(seed=2)

    assert not np.array_equal(
        first.returns,
        second.returns,
    )


@pytest.mark.parametrize(
    "observations",
    [0, 1, -1],
)
def test_simulation_requires_two_observations(
    observations: int,
) -> None:
    """At least two simulated observations are required."""
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        simulate_garch_process(
            observations=observations
        )


@pytest.mark.parametrize(
    "observations",
    [10.0, True],
)
def test_simulation_rejects_non_integer_observations(
    observations: object,
) -> None:
    """Simulation length must be an integer."""
    with pytest.raises(
        TypeError,
        match="integer",
    ):
        simulate_garch_process(
            observations=observations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "omega",
    [0.0, -0.000001],
)
def test_simulation_requires_positive_omega(
    omega: float,
) -> None:
    """The variance intercept must be positive."""
    with pytest.raises(
        ValueError,
        match="omega must be greater",
    ):
        simulate_garch_process(omega=omega)


@pytest.mark.parametrize(
    "alpha",
    [-0.01, -1.0],
)
def test_simulation_rejects_negative_alpha(
    alpha: float,
) -> None:
    """ARCH coefficients cannot be negative."""
    with pytest.raises(
        ValueError,
        match="alpha cannot be negative",
    ):
        simulate_garch_process(alpha=alpha)


@pytest.mark.parametrize(
    "beta",
    [-0.01, -1.0],
)
def test_simulation_rejects_negative_beta(
    beta: float,
) -> None:
    """GARCH coefficients cannot be negative."""
    with pytest.raises(
        ValueError,
        match="beta cannot be negative",
    ):
        simulate_garch_process(beta=beta)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (0.10, 0.90),
        (0.20, 0.90),
        (1.00, 0.00),
    ],
)
def test_simulation_requires_stationary_persistence(
    alpha: float,
    beta: float,
) -> None:
    """Stationarity requires alpha plus beta below one."""
    with pytest.raises(
        ValueError,
        match="less than one",
    ):
        simulate_garch_process(
            alpha=alpha,
            beta=beta,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("omega", np.nan),
        ("alpha", np.inf),
        ("beta", -np.inf),
    ],
)
def test_simulation_rejects_non_finite_parameters(
    parameter: str,
    value: float,
) -> None:
    """Simulation parameters must be finite."""
    arguments = {
        "omega": 0.000001,
        "alpha": 0.08,
        "beta": 0.90,
    }
    arguments[parameter] = value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        simulate_garch_process(**arguments)


def test_fitted_garch_structure(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """GARCH fitting should return complete diagnostics."""
    result = fitted_garch

    assert isinstance(result, FittedVolatilityModel)
    assert result.model_type == "GARCH"
    assert result.p == 1
    assert result.q == 1
    assert result.distribution == "normal"
    assert result.observations == 1200
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert result.conditional_volatilities.shape == (1200,)
    assert result.standardized_residuals.shape == (1200,)
    assert np.all(result.conditional_volatilities > 0.0)


def test_fitted_garch_contains_required_parameters(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """A GARCH(1,1) model should contain omega, alpha, beta."""
    assert "omega" in fitted_garch.parameters
    assert "alpha[1]" in fitted_garch.parameters
    assert "beta[1]" in fitted_garch.parameters


def test_fitted_garch_persistence_is_plausible(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """Estimated persistence should be positive and stationary."""
    persistence = calculate_persistence(
        fitted_garch
    )

    assert 0.0 < persistence < 1.0


def test_fit_arch_model(
    simulated_garch: GarchSimulation,
) -> None:
    """A pure ARCH model should fit successfully."""
    result = fit_volatility_model(
        simulated_garch.returns,
        model_type="ARCH",
        p=1,
        q=0,
    )

    assert result.model_type == "ARCH"
    assert result.p == 1
    assert result.q == 0
    assert "alpha[1]" in result.parameters
    assert not any(
        name.startswith("beta[")
        for name in result.parameters
    )


def test_fit_with_constant_mean(
    simulated_garch: GarchSimulation,
) -> None:
    """The model should support a constant conditional mean."""
    result = fit_volatility_model(
        simulated_garch.returns,
        mean_model="constant",
    )

    assert "mu" in result.parameters


def test_fit_rejects_short_series() -> None:
    """Volatility fitting requires at least 100 returns."""
    with pytest.raises(
        ValueError,
        match="enough observations",
    ):
        fit_volatility_model(
            np.linspace(-0.01, 0.01, 99)
        )


def test_fit_rejects_constant_returns() -> None:
    """A constant return series has no estimable volatility."""
    with pytest.raises(
        ValueError,
        match="non-constant",
    ):
        fit_volatility_model(
            np.full(200, 0.01)
        )


def test_forecast_structure(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """Forecast arrays should match the requested horizon."""
    result = forecast_volatility_model(
        fitted_garch,
        steps=10,
    )

    assert isinstance(result, VolatilityForecast)
    assert result.steps == 10
    assert result.variance_forecasts.shape == (10,)
    assert result.volatility_forecasts.shape == (10,)
    assert (
        result.annualized_volatility_forecasts.shape
        == (10,)
    )
    assert np.all(result.variance_forecasts > 0.0)


def test_forecast_volatility_matches_variance(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """Forecast volatility should equal square-root variance."""
    result = forecast_volatility_model(
        fitted_garch,
        steps=10,
    )

    np.testing.assert_allclose(
        result.volatility_forecasts,
        np.sqrt(result.variance_forecasts),
    )


def test_forecast_annualization(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """Annualized forecast should use square-root-of-time."""
    result = forecast_volatility_model(
        fitted_garch,
        steps=10,
        periods_per_year=12,
    )

    np.testing.assert_allclose(
        result.annualized_volatility_forecasts,
        result.volatility_forecasts
        * np.sqrt(12),
    )


def test_forecast_rejects_wrong_model_type() -> None:
    """Forecasting requires the fitted-model wrapper."""
    with pytest.raises(
        TypeError,
        match="FittedVolatilityModel",
    ):
        forecast_volatility_model(
            "model",  # type: ignore[arg-type]
            steps=5,
        )


def test_calculate_persistence_known_value(
    manual_model: FittedVolatilityModel,
) -> None:
    """Persistence should equal sum of alpha and beta."""
    result = calculate_persistence(
        manual_model
    )

    assert result == pytest.approx(0.98)


def test_unconditional_variance_known_value(
    manual_model: FittedVolatilityModel,
) -> None:
    """Long-run variance should match omega/(1-alpha-beta)."""
    result = calculate_unconditional_variance(
        manual_model
    )

    expected = (
        0.01
        / (1.0 - 0.08 - 0.90)
        / 10000.0
    )

    assert result == pytest.approx(expected)


def test_unconditional_variance_rejects_nonstationary_model(
    manual_model: FittedVolatilityModel,
) -> None:
    """Long-run variance is undefined at persistence one."""
    nonstationary_model = FittedVolatilityModel(
        model_type=manual_model.model_type,
        p=manual_model.p,
        q=manual_model.q,
        distribution=manual_model.distribution,
        observations=manual_model.observations,
        log_likelihood=manual_model.log_likelihood,
        aic=manual_model.aic,
        bic=manual_model.bic,
        parameters={
            "omega": 0.01,
            "alpha[1]": 0.10,
            "beta[1]": 0.90,
        },
        conditional_volatilities=(
            manual_model.conditional_volatilities
        ),
        standardized_residuals=(
            manual_model.standardized_residuals
        ),
        result=None,
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        calculate_unconditional_variance(
            nonstationary_model
        )


def test_fitted_model_is_immutable(
    fitted_garch: FittedVolatilityModel,
) -> None:
    """Fitted-model attributes should be immutable."""
    with pytest.raises(FrozenInstanceError):
        fitted_garch.aic = 0.0  # type: ignore[misc]


def test_simulation_result_is_immutable(
    simulated_garch: GarchSimulation,
) -> None:
    """Simulation-result attributes should be immutable."""
    with pytest.raises(FrozenInstanceError):
        simulated_garch.returns = np.array([])  # type: ignore[misc]


def test_main_prints_garch_analysis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The demonstration should print model diagnostics."""
    main()

    output = capsys.readouterr().out

    assert "GARCH volatility model" in output
    assert "Estimated parameters" in output
    assert "Model diagnostics" in output
    assert "Persistence" in output
    assert "Long-run daily volatility" in output
    assert "Ten-day volatility forecast" in output