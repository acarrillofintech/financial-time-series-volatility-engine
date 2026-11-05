# Financial Time Series & Volatility Engine — Mathematical Methodology

## 1. Purpose and scope

The Financial Time Series & Volatility Engine provides a transparent and reproducible framework for analyzing financial prices, returns, temporal dependence, conditional volatility, and out-of-sample forecast performance.

The engine combines:

- historical market-data preparation;
- simple and logarithmic returns;
- descriptive return statistics;
- stationarity testing;
- autocorrelation diagnostics;
- ARIMA mean modeling;
- EWMA volatility estimation;
- ARCH and GARCH models;
- conditional-variance forecasting;
- volatility persistence;
- long-run variance;
- rolling out-of-sample backtesting;
- forecast-loss metrics;
- professional financial visualizations.

The software is designed for education, quantitative research, model development, and portfolio demonstration. It is not a trading recommendation, investment advisory system, or guaranteed forecasting service.

---

## 2. Mathematical notation

Let:

| Symbol | Meaning |
|---|---|
| \(P_t\) | Adjusted asset price at time \(t\) |
| \(R_t\) | Simple return at time \(t\) |
| \(r_t\) | Logarithmic return at time \(t\) |
| \(\mu\) | Expected periodic return |
| \(\sigma^2\) | Unconditional variance |
| \(\sigma_t^2\) | Conditional variance at time \(t\) |
| \(\varepsilon_t\) | Innovation or unexpected shock |
| \(z_t\) | Standardized innovation |
| \(\phi_i\) | Autoregressive coefficient |
| \(\theta_j\) | Moving-average coefficient |
| \(\omega\) | ARCH/GARCH variance intercept |
| \(\alpha_i\) | ARCH shock coefficient |
| \(\beta_j\) | GARCH persistence coefficient |
| \(\lambda\) | EWMA decay factor |
| \(h\) | Forecast horizon |
| \(T\) | Number of observations |
| \(\widehat{\sigma}_{t|t-1}^{2}\) | One-step-ahead variance forecast |

Daily financial data are annualized using:

\[
N=252
\]

trading periods per year.

---

## 3. Historical market data

The engine downloads adjusted daily closing prices using `yfinance`.

Adjusted prices are used because they provide a consistent economic price series after accounting for relevant corporate actions.

The preparation process includes:

1. normalizing the ticker;
2. validating the start and end dates;
3. configuring a project-local cache;
4. downloading adjusted prices;
5. sorting observations chronologically;
6. removing missing or infinite values;
7. eliminating duplicated dates;
8. verifying that prices are strictly positive;
9. calculating simple returns;
10. calculating logarithmic returns;
11. aligning returns with trading dates.

If the price series contains \(n\) observations, the return series contains:

\[
n-1
\]

observations because the first price has no previous price.

---

## 4. Financial returns

### 4.1 Simple return

The simple return is:

\[
R_t
=
\frac{P_t}{P_{t-1}}-1
\]

It represents the proportional gain or loss between two consecutive observations.

For example, if the price changes from 100 to 103:

\[
R_t
=
\frac{103}{100}-1
=
0.03
\]

or 3%.

The associated gross return is:

\[
1+R_t
=
\frac{P_t}{P_{t-1}}
\]

---

### 4.2 Logarithmic return

The continuously compounded return is:

\[
r_t
=
\ln
\left(
\frac{P_t}{P_{t-1}}
\right)
\]

Because:

\[
\ln
\left(
\frac{P_t}{P_{t-1}}
\right)
=
\ln(P_t)-\ln(P_{t-1})
\]

log returns are first differences of logarithmic prices.

---

### 4.3 Conversion between return definitions

Simple return can be obtained from log return:

\[
R_t=e^{r_t}-1
\]

Log return can be obtained from simple return:

\[
r_t=\ln(1+R_t)
\]

provided:

\[
R_t>-1
\]

A simple return of \(-1\) represents a complete loss and has no finite logarithmic equivalent.

---

### 4.4 Multi-period aggregation

Simple returns compound multiplicatively:

\[
1+R_{0,T}
=
\prod_{t=1}^{T}(1+R_t)
\]

Therefore:

\[
R_{0,T}
=
\prod_{t=1}^{T}(1+R_t)-1
\]

Log returns aggregate additively:

\[
r_{0,T}
=
\sum_{t=1}^{T}r_t
\]

and:

\[
R_{0,T}
=
e^{r_{0,T}}-1
\]

This additive property makes log returns convenient for time-series modeling.

---

## 5. Descriptive return statistics

### 5.1 Arithmetic mean

The sample mean return is:

\[
\bar{R}
=
\frac{1}{T}
\sum_{t=1}^{T}R_t
\]

The annualized arithmetic return is:

\[
\bar{R}_{annual}
=
N\bar{R}
\]

This calculation assumes that the periodic expected return remains stable.

---

### 5.2 Sample variance

Sample variance is:

\[
s^2
=
\frac{1}{T-1}
\sum_{t=1}^{T}
(R_t-\bar{R})^2
\]

Sample volatility is:

\[
s=\sqrt{s^2}
\]

Annualized volatility uses the square-root-of-time rule:

\[
s_{annual}
=
s\sqrt{N}
\]

This approximation assumes sufficiently weak serial dependence and a stable variance process over the aggregation horizon.

---

### 5.3 Skewness

Skewness measures distributional asymmetry.

A standardized representation is:

\[
\operatorname{Skew}(R)
=
E
\left[
\left(
\frac{R-\mu}{\sigma}
\right)^3
\right]
\]

Interpretation:

- zero skewness indicates approximate symmetry;
- negative skewness indicates a longer or more severe left tail;
- positive skewness indicates a longer right tail.

Negative skewness is important in risk management because large negative returns may occur more frequently than implied by a symmetric distribution.

---

### 5.4 Excess kurtosis

Kurtosis measures tail heaviness and concentration around the center:

\[
\operatorname{Kurt}(R)
=
E
\left[
\left(
\frac{R-\mu}{\sigma}
\right)^4
\right]
\]

Normal-distribution kurtosis equals 3.

Excess kurtosis is:

\[
\operatorname{ExcessKurt}(R)
=
\operatorname{Kurt}(R)-3
\]

Positive excess kurtosis indicates heavier tails than the normal distribution.

Financial returns commonly exhibit:

- volatility clustering;
- negative skewness;
- positive excess kurtosis;
- more extreme observations than predicted by normality.

---

## 6. Stationarity

### 6.1 Weak stationarity

A time series \(X_t\) is weakly stationary when:

\[
E[X_t]=\mu
\]

for every \(t\),

\[
\operatorname{Var}(X_t)=\sigma^2
\]

for every \(t\), and:

\[
\operatorname{Cov}(X_t,X_{t-k})
=
\gamma_k
\]

depends only on the lag \(k\), not on the calendar date.

Many time-series models require stationary input because their coefficients describe relationships assumed to remain stable.

Financial prices are commonly non-stationary. Returns are generally more stationary.

---

### 6.2 Random walk

A simple random walk is:

\[
P_t=P_{t-1}+\varepsilon_t
\]

or:

\[
P_t=P_0+\sum_{i=1}^{t}\varepsilon_i
\]

The effect of every shock remains permanently embedded in the price level. Consequently, the variance increases through time:

\[
\operatorname{Var}(P_t)
=
t\sigma_{\varepsilon}^{2}
\]

This process is non-stationary.

Its first difference is:

\[
\Delta P_t
=
P_t-P_{t-1}
=
\varepsilon_t
\]

which can be stationary.

---

## 7. Augmented Dickey-Fuller test

The Augmented Dickey-Fuller test evaluates whether a series contains a unit root.

A general ADF regression is:

\[
\Delta X_t
=
c
+
\delta t
+
\gamma X_{t-1}
+
\sum_{i=1}^{p}
\psi_i\Delta X_{t-i}
+
u_t
\]

where:

- \(c\) is an optional constant;
- \(\delta t\) is an optional deterministic trend;
- \(p\) is the number of lagged differences;
- \(u_t\) is an innovation.

The hypotheses are:

\[
H_0:\gamma=0
\]

The series contains a unit root and is non-stationary.

\[
H_1:\gamma<0
\]

The series is stationary.

At significance level \(\alpha_s\):

\[
p\text{-value}<\alpha_s
\]

leads to rejection of the unit-root null hypothesis.

The engine normally uses:

\[
\alpha_s=0.05
\]

The test decision is:

\[
\text{Stationary}
=
\mathbf{1}
\left\{
p\text{-value}<0.05
\right\}
\]

Failure to reject the null does not prove that the series is a random walk. It means that the available sample does not provide sufficient evidence against the unit-root hypothesis.

---

## 8. Differencing

A first difference is:

\[
\Delta X_t
=
X_t-X_{t-1}
\]

A difference over \(d\) periods is:

\[
\Delta_dX_t
=
X_t-X_{t-d}
\]

Differencing can remove stochastic trends and transform an integrated series into a stationary series.

The ARIMA integration parameter \(d\) indicates how many differences are used before fitting the AR and MA components.

---

## 9. Autocorrelation diagnostics

### 9.1 Autocovariance

The lag-\(k\) autocovariance is:

\[
\gamma_k
=
\operatorname{Cov}(X_t,X_{t-k})
\]

### 9.2 Autocorrelation function

The autocorrelation function is:

\[
\rho_k
=
\frac{\gamma_k}{\gamma_0}
\]

where:

\[
\gamma_0=\operatorname{Var}(X_t)
\]

The sample autocorrelation indicates the linear relationship between the present value and a lagged value.

An approximate 95% confidence band under white noise is:

\[
\pm\frac{1.96}{\sqrt{T}}
\]

Values outside this band may indicate statistically relevant serial dependence.

---

### 9.3 Partial autocorrelation function

The partial autocorrelation at lag \(k\) measures the direct relationship between \(X_t\) and \(X_{t-k}\) after controlling for lags:

\[
1,2,\ldots,k-1
\]

ACF and PACF are useful for identifying candidate mean models:

| Pattern | Possible model |
|---|---|
| PACF cuts off after lag \(p\) | AR(\(p\)) |
| ACF cuts off after lag \(q\) | MA(\(q\)) |
| Both decay gradually | ARMA or ARIMA |

These patterns are diagnostic guidance rather than automatic proof of the correct specification.

---

## 10. Autoregressive models

An autoregressive process of order \(p\) is:

\[
X_t
=
c
+
\phi_1X_{t-1}
+
\phi_2X_{t-2}
+\cdots+
\phi_pX_{t-p}
+
\varepsilon_t
\]

For AR(1):

\[
X_t=c+\phi X_{t-1}+\varepsilon_t
\]

A zero-mean AR(1) is covariance-stationary when:

\[
|\phi|<1
\]

The unconditional variance is:

\[
\operatorname{Var}(X_t)
=
\frac{\sigma_{\varepsilon}^2}
{1-\phi^2}
\]

The \(h\)-step forecast of a zero-mean AR(1) is:

\[
\widehat{X}_{t+h|t}
=
\phi^hX_t
\]

When \(|\phi|<1\), forecasts converge toward the long-run mean as \(h\) increases.

---

## 11. Moving-average models

A moving-average process of order \(q\) is:

\[
X_t
=
\mu
+
\varepsilon_t
+
\theta_1\varepsilon_{t-1}
+\cdots+
\theta_q\varepsilon_{t-q}
\]

An MA model represents the current observation as a function of current and previous shocks.

Unlike an AR model, the effect of an MA shock disappears after a finite number of periods.

---

## 12. ARIMA models

An ARIMA model is represented as:

\[
ARIMA(p,d,q)
\]

where:

- \(p\) is the autoregressive order;
- \(d\) is the integration order;
- \(q\) is the moving-average order.

Using the backshift operator \(B\):

\[
BX_t=X_{t-1}
\]

the model can be written as:

\[
\phi(B)(1-B)^dX_t
=
c+\theta(B)\varepsilon_t
\]

where:

\[
\phi(B)
=
1-\phi_1B-\cdots-\phi_pB^p
\]

and:

\[
\theta(B)
=
1+\theta_1B+\cdots+\theta_qB^q
\]

For stationary returns:

\[
d=0
\]

and ARIMA becomes an ARMA model.

The notebook uses:

\[
ARIMA(1,0,1)
\]

for return forecasting.

---

## 13. ARIMA estimation and information criteria

The ARIMA model is estimated using likelihood-based methods.

Let \(k\) be the number of estimated parameters and \(\widehat{L}\) the maximized likelihood.

### Akaike Information Criterion

\[
AIC
=
2k-2\ln(\widehat{L})
\]

### Bayesian Information Criterion

\[
BIC
=
k\ln(T)-2\ln(\widehat{L})
\]

### Hannan-Quinn Information Criterion

\[
HQIC
=
2k\ln(\ln T)-2\ln(\widehat{L})
\]

Lower values indicate a better balance between fit and complexity when comparing models estimated on the same data.

These criteria do not guarantee superior out-of-sample forecasting.

---

## 14. ARIMA forecast evaluation

For actual values \(y_t\) and forecasts \(\widehat{y}_t\), the error is:

\[
e_t=y_t-\widehat{y}_t
\]

### Mean Absolute Error

\[
MAE
=
\frac{1}{n}
\sum_{t=1}^{n}|e_t|
\]

### Root Mean Squared Error

\[
RMSE
=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}e_t^2
}
\]

RMSE penalizes large errors more heavily than MAE.

### Mean Absolute Percentage Error

\[
MAPE
=
\frac{1}{n}
\sum_{t=1}^{n}
\left|
\frac{e_t}{y_t}
\right|
\]

MAPE is unstable when actual values are close to zero. This is particularly relevant for daily financial returns, so it should not be interpreted in isolation.

---

## 15. Conditional heteroskedasticity

Traditional constant-variance models assume:

\[
\operatorname{Var}(\varepsilon_t)
=
\sigma^2
\]

Financial returns frequently violate this assumption.

Instead:

\[
\operatorname{Var}
(\varepsilon_t\mid\mathcal{F}_{t-1})
=
\sigma_t^2
\]

where \(\mathcal{F}_{t-1}\) represents the information available before time \(t\).

Conditional variance changes through time and tends to form clusters.

A standard return representation is:

\[
R_t
=
\mu_t+\varepsilon_t
\]

with:

\[
\varepsilon_t
=
\sigma_tz_t
\]

and:

\[
E[z_t]=0
\]

\[
\operatorname{Var}(z_t)=1
\]

---

## 16. EWMA volatility

The EWMA variance recursion is:

\[
\sigma_t^2
=
\lambda\sigma_{t-1}^2
+
(1-\lambda)R_{t-1}^2
\]

where:

\[
0<\lambda<1
\]

The engine uses the common daily value:

\[
\lambda=0.94
\]

The weight assigned to a squared return \(k\) periods in the past is:

\[
w_k
=
(1-\lambda)\lambda^k
\]

The infinite sequence of weights sums to one:

\[
\sum_{k=0}^{\infty}
(1-\lambda)\lambda^k
=
1
\]

A high \(\lambda\):

- produces slower volatility adjustment;
- gives more persistence to historical observations;
- assigns a smaller weight to the latest return.

A lower \(\lambda\):

- reacts more strongly to recent shocks;
- forgets older observations faster.

---

### 16.1 EWMA initialization

The engine initializes EWMA with sample variance:

\[
\sigma_0^2
=
\frac{1}{T-1}
\sum_{t=1}^{T}
(R_t-\bar{R})^2
\]

A user-specified positive initial variance may also be supplied.

---

### 16.2 EWMA forecasting

The next variance forecast is:

\[
\widehat{\sigma}_{t+1|t}^{2}
=
\lambda\sigma_t^2
+
(1-\lambda)R_t^2
\]

EWMA corresponds to an integrated GARCH structure with no independent long-run variance target.

Under expected future squared returns equal to forecast variance:

\[
\widehat{\sigma}_{t+h|t}^{2}
=
\widehat{\sigma}_{t+1|t}^{2}
\]

for:

\[
h\geq1
\]

Therefore, expected multi-step EWMA variance forecasts remain constant.

---

## 17. ARCH models

An ARCH(\(p\)) process is:

\[
\sigma_t^2
=
\omega
+
\sum_{i=1}^{p}
\alpha_i\varepsilon_{t-i}^{2}
\]

with constraints:

\[
\omega>0
\]

and:

\[
\alpha_i\geq0
\]

The conditional variance responds to previous squared shocks.

For ARCH(1):

\[
\sigma_t^2
=
\omega
+
\alpha\varepsilon_{t-1}^{2}
\]

A large shock increases the next-period variance regardless of whether the shock was positive or negative.

---

## 18. GARCH models

A GARCH(\(p,q\)) model is:

\[
\sigma_t^2
=
\omega
+
\sum_{i=1}^{p}
\alpha_i\varepsilon_{t-i}^{2}
+
\sum_{j=1}^{q}
\beta_j\sigma_{t-j}^{2}
\]

The engine focuses on GARCH(1,1):

\[
\sigma_t^2
=
\omega
+
\alpha\varepsilon_{t-1}^{2}
+
\beta\sigma_{t-1}^{2}
\]

Interpretation:

- \(\omega\): baseline variance component;
- \(\alpha\): immediate sensitivity to new shocks;
- \(\beta\): persistence of previous conditional variance.

Parameter constraints are:

\[
\omega>0
\]

\[
\alpha\geq0
\]

\[
\beta\geq0
\]

For covariance stationarity:

\[
\alpha+\beta<1
\]

---

## 19. Volatility persistence

GARCH(1,1) persistence is:

\[
\pi=\alpha+\beta
\]

Interpretation:

- low persistence implies rapid shock decay;
- high persistence implies slow volatility decay;
- persistence near one implies that volatility shocks remain influential for a long time.

The expected decay after \(h\) periods is approximately:

\[
\pi^h
\]

A volatility half-life can be approximated by:

\[
HL
=
\frac{\ln(0.5)}
{\ln(\pi)}
\]

when:

\[
0<\pi<1
\]

---

## 20. Long-run GARCH variance

Taking unconditional expectations of:

\[
\sigma_t^2
=
\omega
+
\alpha\varepsilon_{t-1}^{2}
+
\beta\sigma_{t-1}^{2}
\]

and using:

\[
E[\varepsilon_t^2]
=
E[\sigma_t^2]
=
\sigma_{\infty}^2
\]

gives:

\[
\sigma_{\infty}^2
=
\omega
+
\alpha\sigma_{\infty}^2
+
\beta\sigma_{\infty}^2
\]

Therefore:

\[
\sigma_{\infty}^2
=
\frac{\omega}
{1-\alpha-\beta}
\]

This quantity is defined only when:

\[
\alpha+\beta<1
\]

Long-run daily volatility is:

\[
\sigma_{\infty}
=
\sqrt{
\frac{\omega}
{1-\alpha-\beta}
}
\]

Annualized long-run volatility is:

\[
\sigma_{\infty,annual}
=
\sigma_{\infty}\sqrt{252}
\]

---

## 21. GARCH variance forecasting

The one-step GARCH(1,1) forecast is:

\[
\widehat{\sigma}_{t+1|t}^{2}
=
\omega
+
\alpha\varepsilon_t^2
+
\beta\sigma_t^2
\]

For horizons greater than one, future squared innovations are replaced by their conditional expectations.

The forecast can be expressed relative to long-run variance:

\[
\widehat{\sigma}_{t+h|t}^{2}
-
\sigma_{\infty}^{2}
=
(\alpha+\beta)^{h-1}
\left(
\widehat{\sigma}_{t+1|t}^{2}
-
\sigma_{\infty}^{2}
\right)
\]

Thus, stationary GARCH forecasts converge toward long-run variance.

---

## 22. Innovation distributions

The engine supports several standardized innovation distributions.

### Normal

\[
z_t\sim N(0,1)
\]

This distribution is simple but may underestimate extreme financial returns.

### Student-t

The Student-t distribution provides heavier tails than the normal distribution and can better represent extreme observations.

### Skewed Student-t

The skewed Student-t distribution permits both:

- heavy tails;
- asymmetric innovations.

### Generalized Error Distribution

The GED allows flexible tail thickness through a shape parameter.

Distribution choice affects likelihood, parameter estimates, tail forecasts, and risk measures.

---

## 23. Numerical return scaling

Daily decimal returns are often numerically small:

\[
R_t\approx0.01
\]

Before ARCH/GARCH estimation, the engine scales returns as percentages:

\[
R_t^{scaled}=100R_t
\]

This improves numerical conditioning during optimization.

Estimated percentage variance is converted back to decimal variance using:

\[
\sigma_{decimal}^{2}
=
\frac{
\sigma_{percentage}^{2}
}{
100^2
}
\]

Volatility is converted using:

\[
\sigma_{decimal}
=
\frac{
\sigma_{percentage}
}{
100
}
\]

This scaling changes numerical units but not the underlying economic model.

---

## 24. Realized-variance proxy

True conditional variance is latent and cannot be directly observed.

The backtesting framework uses squared return as a daily realized-variance proxy:

\[
RV_t=R_t^2
\]

This proxy is unbiased under appropriate assumptions but is noisy because one daily observation contains limited information about the latent variance.

Higher-frequency realized-volatility measures could provide more precise proxies.

---

## 25. Rolling out-of-sample backtesting

An honest forecast evaluation must prevent future information from entering model estimation.

For every forecast origin \(t\):

1. select returns from the historical window;
2. estimate or update the volatility model;
3. generate forecasts using information through \(t-1\);
4. observe \(R_t^2\);
5. compare the forecast with the realization;
6. move the window forward.

For estimation-window length \(W\):

\[
\mathcal{D}_t
=
\{R_{t-W},\ldots,R_{t-1}\}
\]

The forecast is:

\[
\widehat{\sigma}_{t|t-1}^{2}
=
f(\mathcal{D}_t)
\]

The realized variance:

\[
R_t^2
\]

is not included in \(\mathcal{D}_t\).

This prevents look-ahead bias.

---

### 25.1 GARCH refitting frequency

Repeatedly estimating GARCH for every observation can be computationally expensive.

The engine supports block forecasting:

1. estimate GARCH at a block origin;
2. forecast several future horizons;
3. advance to the next origin;
4. refit using only data then available.

The `refit_frequency` parameter controls the tradeoff between:

- computational cost;
- responsiveness to parameter changes.

A lower frequency value means more frequent re-estimation.

---

## 26. Volatility forecast metrics

Let:

\[
RV_t=R_t^2
\]

and:

\[
\widehat{V}_t
=
\widehat{\sigma}_{t|t-1}^2
\]

The variance forecast error is:

\[
e_t
=
\widehat{V}_t-RV_t
\]

---

### 26.1 Mean Absolute Error

\[
MAE
=
\frac{1}{n}
\sum_{t=1}^{n}|e_t|
\]

MAE measures average absolute deviation and is relatively easy to interpret.

---

### 26.2 Root Mean Squared Error

\[
RMSE
=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}e_t^2
}
\]

RMSE gives greater weight to large errors.

---

### 26.3 QLIKE loss

The Gaussian quasi-likelihood loss is:

\[
QLIKE
=
\frac{1}{n}
\sum_{t=1}^{n}
\left[
\ln(\widehat{V}_t)
+
\frac{RV_t}{\widehat{V}_t}
\right]
\]

Lower QLIKE values indicate better forecasts.

QLIKE can be negative because daily variances are often smaller than one:

\[
\ln(\widehat{V}_t)<0
\]

A negative QLIKE value is not an error. Model comparison depends on relative loss: the lower value is preferred.

---

### 26.4 Forecast bias

Average bias is:

\[
Bias
=
\frac{1}{n}
\sum_{t=1}^{n}
(\widehat{V}_t-RV_t)
\]

Equivalently:

\[
Bias
=
\overline{\widehat{V}}
-
\overline{RV}
\]

Interpretation:

- positive bias indicates average overforecasting;
- negative bias indicates average underforecasting;
- bias near zero indicates average calibration.

A low bias does not guarantee accurate individual forecasts.

---

### 26.5 Variance correlation

Forecast correlation is:

\[
\rho
=
\operatorname{Corr}
(RV_t,\widehat{V}_t)
\]

Higher positive correlation indicates that the model better tracks changes in realized variance.

Correlation is undefined when either series is constant.

---

## 27. Model comparison

The engine compares EWMA and GARCH using identical out-of-sample observations.

The preferred model by QLIKE is:

\[
M^*
=
\arg\min_{M\in\{EWMA,GARCH\}}
QLIKE_M
\]

Different metrics can select different models because they penalize errors differently.

A sound decision should consider:

- QLIKE;
- MAE;
- RMSE;
- bias;
- correlation;
- stability;
- interpretability;
- computational cost;
- intended risk application.

No single metric provides a complete model assessment.

---

## 28. Reproducibility

Simulation functions accept a random seed.

With identical:

- inputs;
- parameters;
- software versions;
- random seed;

the simulation produces identical values.

This is essential for:

- testing;
- debugging;
- model review;
- auditability;
- research replication.

Historical downloads may change if the data provider revises adjusted prices.

---

## 29. Input validation

The engine validates:

- one-dimensional arrays;
- finite observations;
- positive prices;
- valid return sample sizes;
- positive annualization frequencies;
- valid ARIMA orders;
- valid ARCH and GARCH orders;
- positive forecast horizons;
- EWMA decay factors between zero and one;
- positive initial variance;
- positive GARCH \(\omega\);
- non-negative \(\alpha\) and \(\beta\);
- stationary simulation persistence;
- aligned forecasts and realizations;
- strictly positive variance forecasts;
- chronological date ranges.

Invalid inputs raise explicit errors rather than producing silent numerical results.

---

## 30. Automated testing

The automated suite verifies:

- known return calculations;
- simple/log return conversion;
- cumulative compounding;
- annualization;
- skewness and kurtosis;
- ADF behavior on stationary and non-stationary processes;
- ACF and PACF dimensions;
- ARIMA validation and forecasting;
- EWMA recursion;
- geometric EWMA weights;
- ARCH/GARCH simulation;
- variance recursion;
- persistence;
- long-run variance;
- forecast shapes;
- numerical scaling;
- data cleaning;
- retry behavior;
- out-of-sample alignment;
- absence of future leakage;
- QLIKE and error metrics;
- visualization creation;
- saved graphical artifacts;
- immutable result objects.

Warnings are treated as errors during full validation:

```text
python -m pytest -q -W error
```

This helps identify deprecated library behavior and numerical issues before release.

---

## 31. Principal assumptions

The framework relies on the following assumptions:

1. Adjusted closing prices accurately represent the selected asset.
2. Daily observations are chronologically aligned.
3. Missing observations correspond primarily to non-trading days.
4. Returns are sufficiently stationary for the selected models.
5. ARIMA describes linear conditional-mean dependence.
6. EWMA uses a constant decay factor.
7. ARCH/GARCH parameters remain stable within each estimation window.
8. Standardized innovations are independent and identically distributed under the selected distribution.
9. Conditional variance is positive.
10. Squared return is an acceptable daily realized-variance proxy.
11. Annualization uses 252 trading days.
12. Historical relationships contain information relevant to the forecast horizon.
13. Transaction costs and liquidity effects are excluded.
14. Forecasts are statistical estimates rather than guaranteed outcomes.

---

## 32. Limitations

Important limitations include:

- Daily squared returns are noisy variance proxies.
- ADF tests can have limited power in finite samples.
- ACF and PACF identify linear dependence only.
- ARIMA cannot capture every nonlinear return pattern.
- EWMA has no independent long-run variance target.
- Standard GARCH treats positive and negative shocks symmetrically.
- Normal innovations may underestimate extreme tails.
- Parameters may change after structural breaks.
- Long historical windows can react slowly to new regimes.
- Short windows can produce unstable parameter estimates.
- Block GARCH forecasts may adapt less quickly between refits.
- Market-data providers may revise historical adjusted prices.
- In-sample likelihood does not guarantee out-of-sample performance.
- Statistical significance does not automatically imply economic value.
- Volatility forecasts do not predict return direction.
- The framework excludes trading costs, bid-ask spreads, market impact, and liquidity risk.

---

## 33. Potential extensions

Future versions could include:

- Student-t GARCH;
- skewed Student-t GARCH;
- EGARCH;
- GJR-GARCH;
- APARCH;
- stochastic-volatility models;
- HAR realized-volatility models;
- intraday realized variance;
- regime-switching volatility;
- multivariate GARCH;
- dynamic conditional correlation;
- ARIMAX with macroeconomic variables;
- volatility forecast combinations;
- Diebold-Mariano forecast tests;
- VaR and Expected Shortfall integration;
- options-implied volatility;
- machine-learning forecasts;
- automated model selection;
- interactive web dashboards;
- scheduled market-data updates;
- database persistence;
- production model monitoring.

---

## 34. Practical interpretation framework

| Measure | Question answered |
|---|---|
| Simple return | How much did the asset gain or lose? |
| Log return | What was the continuously compounded change? |
| Volatility | How variable were returns? |
| Skewness | Is the distribution asymmetric? |
| Excess kurtosis | Are extreme returns unusually frequent? |
| ADF test | Is the series statistically stationary? |
| ACF | Is the series correlated with previous values? |
| PACF | Which lags have a direct relationship? |
| ARIMA | Can past returns and errors explain the conditional mean? |
| EWMA | How does recent market activity affect risk? |
| ARCH | How do previous shocks affect volatility? |
| GARCH | How do shocks and previous variance jointly affect volatility? |
| Persistence | How slowly does a volatility shock disappear? |
| Long-run variance | Toward what variance does GARCH revert? |
| MAE | What is the average absolute variance error? |
| RMSE | How large are errors when large misses receive extra weight? |
| QLIKE | Which variance model performs better under quasi-likelihood loss? |
| Bias | Does the model overforecast or underforecast on average? |
| Correlation | Does the forecast track changes in realized variance? |

---

## 35. Educational disclaimer

This software is a quantitative educational simulator.

Results depend on:

- historical data;
- sample period;
- model specification;
- estimation window;
- distributional assumptions;
- numerical optimization;
- forecast horizon;
- market regime.

Outputs must not be interpreted as:

- guaranteed forecasts;
- investment recommendations;
- trading signals;
- regulatory calculations;
- accounting valuations;
- promises of future performance.

Production use would require:

- validated institutional data;
- independent model validation;
- parameter-stability analysis;
- formal model governance;
- security controls;
- access management;
- audit trails;
- monitoring;
- documented approvals;
- regulatory review where applicable.
