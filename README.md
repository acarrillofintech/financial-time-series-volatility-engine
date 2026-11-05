# Financial Time Series & Volatility Engine

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-397%20passed-brightgreen)](#testing)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success)](#project-status)

A quantitative Python engine for financial time-series analysis, ARIMA forecasting, EWMA volatility, ARCH/GARCH modeling, market-data preparation, professional visualization, and out-of-sample volatility backtesting.

> Educational quantitative-finance software. It is not investment advice, a trading recommendation, or a guarantee of future performance.

---

## English

### Overview

The Financial Time Series & Volatility Engine is a modular quantitative-finance project developed in Python.

It studies the behavior of financial prices, returns, conditional means, and time-varying volatility through:

- historical adjusted market prices;
- simple and logarithmic returns;
- descriptive return statistics;
- stationarity analysis;
- Augmented Dickey-Fuller testing;
- autocorrelation and partial autocorrelation;
- ARIMA modeling;
- out-of-sample return forecasting;
- EWMA conditional volatility;
- ARCH and GARCH models;
- volatility persistence;
- long-run variance;
- multi-period volatility forecasts;
- rolling out-of-sample backtesting;
- MAE, RMSE, QLIKE, bias, and correlation;
- professional analytical visualizations.

The project emphasizes mathematical transparency, reproducibility, modular software design, strict input validation, and automated testing.

---

### Main objectives

The engine was designed to answer questions such as:

- How are financial prices converted into returns?
- Are prices and returns statistically stationary?
- Do past returns contain linear forecasting information?
- Can ARIMA forecast the conditional mean of returns?
- How does recent market activity affect current volatility?
- How quickly does volatility react to a new shock?
- How long does a volatility shock remain influential?
- What is the long-run GARCH variance?
- How does EWMA differ from GARCH?
- Which volatility model performs better outside the estimation sample?
- How can volatility forecasts be evaluated without future-data leakage?

---

### Mathematical framework

#### Simple return

\[
R_t
=
\frac{P_t}{P_{t-1}}-1
\]

#### Logarithmic return

\[
r_t
=
\ln
\left(
\frac{P_t}{P_{t-1}}
\right)
\]

#### Annualized volatility

\[
\sigma_{annual}
=
\sigma_{daily}\sqrt{252}
\]

#### Augmented Dickey-Fuller test

\[
\Delta X_t
=
c+\gamma X_{t-1}
+
\sum_{i=1}^{p}
\psi_i\Delta X_{t-i}
+
u_t
\]

The null hypothesis indicates a unit root and non-stationarity.

#### ARIMA model

\[
\phi(B)(1-B)^dX_t
=
c+\theta(B)\varepsilon_t
\]

#### EWMA variance

\[
\sigma_t^2
=
\lambda\sigma_{t-1}^2
+
(1-\lambda)R_{t-1}^2
\]

#### ARCH model

\[
\sigma_t^2
=
\omega
+
\sum_{i=1}^{p}
\alpha_i\varepsilon_{t-i}^2
\]

#### GARCH model

\[
\sigma_t^2
=
\omega
+
\sum_{i=1}^{p}
\alpha_i\varepsilon_{t-i}^2
+
\sum_{j=1}^{q}
\beta_j\sigma_{t-j}^2
\]

For GARCH(1,1):

\[
\sigma_t^2
=
\omega
+
\alpha\varepsilon_{t-1}^2
+
\beta\sigma_{t-1}^2
\]

#### Volatility persistence

\[
\pi=\alpha+\beta
\]

#### Long-run variance

\[
\sigma_{\infty}^2
=
\frac{\omega}
{1-\alpha-\beta}
\]

#### QLIKE loss

\[
QLIKE
=
\frac{1}{n}
\sum_{t=1}^{n}
\left[
\ln(\widehat{\sigma}_t^2)
+
\frac{R_t^2}
{\widehat{\sigma}_t^2}
\right]
\]

A complete derivation and interpretation of every model is available in [docs/methodology.md](docs/methodology.md).

---

### Project modules

| Module | Purpose |
|---|---|
| `returns.py` | Simple returns, log returns, compounding, annualization, skewness, and kurtosis |
| `stationarity.py` | ADF testing, differencing, ACF, and PACF |
| `arima_model.py` | ARIMA estimation, forecasting, confidence intervals, and forecast metrics |
| `ewma_volatility.py` | EWMA weights, conditional volatility, and volatility forecasts |
| `garch_model.py` | ARCH/GARCH simulation, estimation, persistence, long-run variance, and forecasts |
| `data_loader.py` | Historical adjusted-price download, cache management, cleaning, and return preparation |
| `volatility_backtest.py` | Rolling EWMA/GARCH backtesting and forecast comparison |
| `visualization.py` | Professional market, time-series, volatility, and backtesting charts |

---

### Main capabilities

#### 1. Return analysis

The engine calculates:

- simple returns;
- logarithmic returns;
- compounded cumulative return;
- arithmetic mean;
- daily volatility;
- annualized return;
- annualized volatility;
- minimum and maximum return;
- skewness;
- excess kurtosis.

#### 2. Stationarity diagnostics

The stationarity module:

- validates univariate time series;
- differences non-stationary series;
- performs the Augmented Dickey-Fuller test;
- supports constants and deterministic trends;
- calculates ACF;
- calculates PACF;
- reports statistical stationarity decisions.

#### 3. ARIMA forecasting

The ARIMA component:

- validates \((p,d,q)\) orders;
- fits univariate ARIMA models;
- calculates AIC, BIC, and HQIC;
- generates multi-period point forecasts;
- calculates confidence intervals;
- evaluates forecasts using MAE, RMSE, and MAPE;
- supports reproducible simulated AR processes.

#### 4. EWMA volatility

The EWMA component:

- applies geometric weights to past squared returns;
- emphasizes recent market information;
- supports configurable decay factors;
- estimates conditional variance;
- estimates daily and annualized volatility;
- generates multi-period volatility forecasts;
- illustrates the integrated nature of EWMA variance.

#### 5. ARCH and GARCH models

The conditional-volatility component:

- simulates stationary GARCH(1,1) returns;
- fits ARCH models;
- fits GARCH models;
- supports normal, Student-t, skewed Student-t, and GED innovations;
- estimates model parameters;
- calculates log-likelihood, AIC, and BIC;
- measures volatility persistence;
- calculates long-run variance;
- produces daily and annualized volatility forecasts.

#### 6. Historical market data

The data loader:

- downloads adjusted prices from Yahoo Finance;
- uses a project-local yfinance cache;
- retries transient failed downloads;
- normalizes tickers and dates;
- removes missing and non-finite prices;
- validates positive prices;
- removes duplicated dates;
- handles timezone-aware indexes;
- produces aligned simple and logarithmic returns.

#### 7. Volatility backtesting

The backtesting framework:

- uses rolling estimation windows;
- prevents future-data leakage;
- generates one-step EWMA forecasts;
- periodically re-estimates GARCH;
- supports block forecasts;
- compares forecasts with squared realized returns;
- calculates MAE;
- calculates RMSE;
- calculates QLIKE;
- calculates forecast bias;
- calculates variance correlation;
- automatically selects the lowest-QLIKE model.

#### 8. Visualization

The project generates figures for:

- historical adjusted prices;
- daily returns;
- empirical return distribution;
- autocorrelation;
- partial autocorrelation;
- EWMA conditional volatility;
- GARCH conditional volatility;
- GARCH volatility forecasts;
- out-of-sample model comparison.

---

### Example market analysis

The included analysis uses SPY adjusted prices from January 2021 through December 2025.

| Market statistic | Example result |
|---|---:|
| Price observations | 1,255 |
| Return observations | 1,254 |
| Average daily return | 0.0603% |
| Daily volatility | 1.0779% |
| Worst daily return | -5.8543% |
| Best daily return | 10.5019% |

---

### Example volatility backtest

The rolling evaluation uses a 500-day estimation window and 754 out-of-sample observations.

| Metric | EWMA | GARCH |
|---|---:|---:|
| Variance MAE | 0.0001112959 | 0.0001071991 |
| Variance RMSE | 0.0004407850 | 0.0004482502 |
| QLIKE loss | -8.463936 | -8.168038 |
| Forecast bias | 0.0000028897 | -0.0000063261 |
| Variance correlation | 0.188054 | -0.020038 |

In this sample:

```text
Best model by QLIKE: EWMA
```

A negative QLIKE value is valid when forecast variances are smaller than one. The preferred model is the one with the lower loss.

These results are sample-dependent and should not be interpreted as guaranteed future performance.

---

### Generated figures

The visualization module creates:

```text
results/figures/
├── acf_pacf.png
├── backtest_comparison.png
├── conditional_volatility.png
├── price_history.png
├── return_analysis.png
└── volatility_forecast.png
```

---

### Repository structure

```text
financial-time-series-volatility-engine/
├── docs/
│   └── methodology.md
├── notebooks/
│   └── time_series_volatility_analysis.ipynb
├── results/
│   └── figures/
│       ├── acf_pacf.png
│       ├── backtest_comparison.png
│       ├── conditional_volatility.png
│       ├── price_history.png
│       ├── return_analysis.png
│       └── volatility_forecast.png
├── src/
│   ├── __init__.py
│   ├── arima_model.py
│   ├── data_loader.py
│   ├── ewma_volatility.py
│   ├── garch_model.py
│   ├── returns.py
│   ├── stationarity.py
│   ├── visualization.py
│   └── volatility_backtest.py
├── tests/
│   ├── test_arima_model.py
│   ├── test_data_loader.py
│   ├── test_ewma_volatility.py
│   ├── test_garch_model.py
│   ├── test_returns.py
│   ├── test_stationarity.py
│   ├── test_visualization.py
│   └── test_volatility_backtest.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/acarrillofintech/financial-time-series-volatility-engine.git
cd financial-time-series-volatility-engine
```

#### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 4. Validate the environment

```bash
python -m pip check
```

---

### Running the modules

Return analysis:

```bash
python -m src.returns
```

Stationarity and correlation diagnostics:

```bash
python -m src.stationarity
```

ARIMA modeling:

```bash
python -m src.arima_model
```

EWMA volatility:

```bash
python -m src.ewma_volatility
```

ARCH/GARCH modeling:

```bash
python -m src.garch_model
```

Historical market data:

```bash
python -m src.data_loader
```

Volatility backtesting:

```bash
python -m src.volatility_backtest
```

Generate analytical figures:

```bash
python -m src.visualization
```

---

### Running the notebook

Start Jupyter:

```bash
python -m jupyter notebook
```

Open:

```text
notebooks/time_series_volatility_analysis.ipynb
```

Select the project kernel:

```text
Python 3.14 (.venv Time Series)
```

Then execute:

```text
Run All
```

The notebook integrates:

- historical data;
- formulas;
- descriptive statistics;
- stationarity testing;
- ACF and PACF;
- ARIMA forecasts;
- EWMA estimates;
- GARCH parameters;
- volatility persistence;
- long-run variance;
- forecast backtesting;
- tables;
- charts;
- financial interpretation.

---

### Testing

Run the complete automated suite:

```bash
python -m pytest -q -W error
```

Current result:

```text
397 passed
```

Run with detailed output:

```bash
python -m pytest -v
```

Run an individual module:

```bash
python -m pytest tests/test_garch_model.py -v
```

Treating warnings as errors helps detect deprecated library behavior and future compatibility problems.

The test suite covers:

- exact mathematical calculations;
- invalid-input handling;
- boundary conditions;
- numerical precision;
- seeded simulation reproducibility;
- stationarity decisions;
- ARIMA model structures;
- forecast horizons;
- EWMA recursions;
- GARCH recursions;
- persistence and long-run variance;
- yfinance retry behavior;
- prevention of future-data leakage;
- forecast-loss metrics;
- graphical output;
- immutable result objects.

---

### Technology stack

- Python 3.14
- NumPy
- pandas
- SciPy
- statsmodels
- ARCH
- Matplotlib
- Seaborn
- yfinance
- Jupyter
- pytest
- Git and GitHub

---

### Model assumptions

The project uses the following principal assumptions:

- adjusted prices correctly represent the asset’s economic price history;
- returns are aligned chronologically;
- return-series dynamics are sufficiently stable within estimation windows;
- ARIMA captures linear conditional-mean relationships;
- EWMA uses a constant decay factor;
- GARCH parameters remain stable during each estimation window;
- standardized innovations follow the selected distribution;
- conditional variances remain positive;
- squared returns are useful daily realized-variance proxies;
- annualization uses 252 trading days;
- historical relationships may contain information relevant to future volatility.

---

### Limitations

Important limitations include:

- squared daily returns are noisy variance proxies;
- daily prices omit intraday volatility information;
- ADF tests can have limited finite-sample power;
- ARIMA does not capture every nonlinear relationship;
- EWMA has no independent long-run variance target;
- standard GARCH treats positive and negative shocks symmetrically;
- normal innovations may underestimate extreme tails;
- structural breaks can invalidate historical parameters;
- block GARCH forecasts adapt only when the model is refitted;
- in-sample fit does not guarantee out-of-sample performance;
- volatility forecasts do not predict return direction;
- transaction costs, liquidity, and market impact are excluded.

See [docs/methodology.md](docs/methodology.md) for full derivations, assumptions, validations, and limitations.

---

### Project status

Implemented components:

- [x] Historical market-data loading
- [x] Simple and logarithmic returns
- [x] Descriptive return statistics
- [x] Stationarity analysis
- [x] Augmented Dickey-Fuller test
- [x] ACF and PACF
- [x] ARIMA estimation
- [x] Return forecasting
- [x] EWMA volatility
- [x] ARCH models
- [x] GARCH models
- [x] Volatility persistence
- [x] Long-run variance
- [x] Conditional-volatility forecasts
- [x] Rolling out-of-sample backtesting
- [x] MAE, RMSE, QLIKE, bias, and correlation
- [x] Professional visualizations
- [x] Analytical notebook
- [x] Detailed mathematical methodology
- [x] Automated test suite

Potential extensions:

- [ ] Student-t GARCH comparison
- [ ] EGARCH
- [ ] GJR-GARCH
- [ ] APARCH
- [ ] Realized volatility
- [ ] Regime-switching models
- [ ] Multivariate GARCH
- [ ] Dynamic conditional correlation
- [ ] Implied-volatility integration
- [ ] Diebold-Mariano tests
- [ ] Interactive web dashboard
- [ ] Production model monitoring

---

## Español

### Descripción general

Financial Time Series & Volatility Engine es un motor cuantitativo desarrollado en Python para analizar precios financieros, rendimientos, dependencia temporal y volatilidad cambiante.

El proyecto integra:

- precios históricos ajustados;
- rendimientos simples y logarítmicos;
- estadística descriptiva;
- estacionariedad;
- prueba ADF;
- ACF y PACF;
- modelos ARIMA;
- volatilidad EWMA;
- modelos ARCH y GARCH;
- persistencia de volatilidad;
- varianza de largo plazo;
- pronósticos;
- backtesting fuera de muestra;
- visualizaciones profesionales.

---

### Preguntas que responde

El motor permite estudiar:

- ¿Cómo se transforma un precio en rendimiento?
- ¿Los precios son estacionarios?
- ¿Los rendimientos son estacionarios?
- ¿Existe dependencia entre rendimientos pasados y futuros?
- ¿Puede ARIMA pronosticar el rendimiento promedio?
- ¿Cómo reacciona EWMA ante un movimiento reciente?
- ¿Cómo modela GARCH los grupos de volatilidad?
- ¿Cuánto tarda un choque de volatilidad en desaparecer?
- ¿Hacia qué nivel de riesgo regresa GARCH?
- ¿Cuál modelo funciona mejor fuera de muestra?
- ¿Cómo se evita utilizar información futura?

---

### Interpretación de los modelos

| Modelo o medida | Pregunta que responde |
|---|---|
| Rendimiento simple | ¿Cuánto ganó o perdió el activo? |
| Rendimiento logarítmico | ¿Cuál fue el cambio continuamente compuesto? |
| Volatilidad | ¿Qué tan variables fueron los rendimientos? |
| ADF | ¿La serie es estacionaria? |
| ACF | ¿Existe relación con observaciones anteriores? |
| PACF | ¿Qué rezagos tienen una relación directa? |
| ARIMA | ¿Puede modelarse el rendimiento promedio? |
| EWMA | ¿Cuánto pesan los movimientos recientes? |
| ARCH | ¿Cómo afectan los choques anteriores a la volatilidad? |
| GARCH | ¿Cómo influyen los choques y la varianza pasada? |
| Persistencia | ¿Cuánto dura un choque de volatilidad? |
| Varianza de largo plazo | ¿Hacia qué nivel de riesgo regresa el modelo? |
| MAE | ¿Cuál es el error absoluto promedio? |
| RMSE | ¿Qué tan importantes son los errores grandes? |
| QLIKE | ¿Cuál pronóstico de varianza es preferible? |
| Sesgo | ¿El modelo sobreestima o subestima el riesgo? |

---

### Ejecución rápida

Activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```powershell
python -m pip install -r requirements.txt
```

Ejecutar todas las pruebas:

```powershell
python -m pytest -q -W error
```

Descargar datos históricos:

```powershell
python -m src.data_loader
```

Ejecutar GARCH:

```powershell
python -m src.garch_model
```

Comparar EWMA y GARCH:

```powershell
python -m src.volatility_backtest
```

Generar gráficos:

```powershell
python -m src.visualization
```

---

### Metodología matemática

La documentación completa incluye:

- rendimientos;
- compounding;
- annualización;
- asimetría y curtosis;
- estacionariedad;
- prueba ADF;
- diferenciación;
- ACF y PACF;
- AR, MA y ARIMA;
- criterios AIC, BIC y HQIC;
- EWMA;
- ARCH;
- GARCH;
- persistencia;
- varianza de largo plazo;
- distribuciones de innovaciones;
- escalamiento numérico;
- backtesting sin información futura;
- MAE, RMSE, QLIKE, sesgo y correlación;
- supuestos;
- validaciones;
- limitaciones.

Consulte:

[docs/methodology.md](docs/methodology.md)

---

### Uso educativo

Este repositorio forma parte de un portafolio profesional de matemática financiera, ingeniería de software y desarrollo cuantitativo.

Demuestra competencias en:

- programación científica;
- series temporales;
- econometría financiera;
- modelado de volatilidad;
- simulación;
- pronósticos;
- validación de modelos;
- backtesting;
- diseño modular;
- pruebas automatizadas;
- visualización;
- documentación matemática;
- Git y GitHub.

---

### Aviso importante

Este software es un simulador cuantitativo con fines educativos y de investigación.

Los resultados dependen de los datos, la muestra, los parámetros y los supuestos utilizados. No deben interpretarse como:

- asesoría financiera;
- señal de compra o venta;
- estrategia garantizada;
- predicción segura;
- recomendación de inversión;
- cálculo regulatorio oficial.

El uso en producción requeriría datos validados, calibración, validación independiente, gobierno de modelos, controles de seguridad, auditoría y monitoreo continuo.

---

## Author / Autor

**Alex Carrillo**

Systems Engineer and MSc in Financial Mathematics, focused on quantitative finance, financial risk, banking technology, data analytics, and Python software development.

GitHub: [acarrillofintech](https://github.com/acarrillofintech)

---

## License / Licencia

This project is distributed under the MIT License.

Este proyecto se distribuye bajo la licencia MIT.

See [LICENSE](LICENSE) for details.