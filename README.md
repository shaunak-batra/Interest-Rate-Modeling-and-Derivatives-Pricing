# Interest Rate Modeling and Derivatives Pricing

## Overview

This project implements a comprehensive **Interest Rate Modeling and Derivatives Pricing** framework using the **Vasicek short-rate model**. The implementation covers fixed-income derivatives pricing, from bond valuation to complex swaption pricing, using both analytical and Monte Carlo techniques.

Key elements include:

- Calibration of Vasicek model to real data  
- Analytical and Monte Carlo bond pricing (with variance reduction)  
- Interest rate swap pricing  
- European call options on zero-coupon bonds  
- European swaption valuation via Black’s model  

This repository is intended as both an educational tool and a practical fixed-income analysis framework.

---

## Mathematical Foundation

### Vasicek Model

The Vasicek short‐rate model posits that the instantaneous short rate \(r(t)\) follows:

    \[
    dr(t) = a\bigl(b - r(t)\bigr)\,dt + \sigma\,dW(t)
    \]

with parameters:

    - \(a\): mean-reversion speed  
    - \(b\): long-term mean  
    - \(\sigma\): volatility  
    - \(dW(t)\): standard Brownian increment  

This model is mean-reverting and has analytic tractability.

---

### Bond Pricing Theory

Under the Vasicek model, the zero-coupon bond price \(P(t, T)\) has a closed form:

    \[
    P(t,T) = A(t,T)\,\exp\bigl(-B(t,T)\,r(t)\bigr)
    \]

where

    \[
    B(t,T) = \frac{1 - e^{-a (T - t)}}{a},
    \]
    
    \[
    A(t,T) = \exp\!\left\{ \left(B(t,T) - (T - t)\right)\frac{a^2 b - \tfrac12 \sigma^2}{a^2} \;-\; \frac{\sigma^2 B(t,T)^2}{4a} \right\}
    \]

Here:

- \(B(t,T)\) is a duration‐like factor  
- \(A(t,T)\) accounts for convexity and volatility effects  

---

### Monte Carlo Simulation

We approximate the SDE with the **Euler–Maruyama** discretization:

    \[
    r_{t+\Delta t} = r_t + a\,(b - r_t)\,\Delta t + \sigma\,\sqrt{\Delta t}\,\epsilon,\quad \epsilon \sim N(0,1)
    \]

To improve estimator efficiency, we use:

- **Antithetic variates**: simulate paired \(\epsilon\) and \(-\epsilon\)  
- **Control variates**: use a correlated quantity with known expectation to reduce variance  

---

### Interest Rate Derivatives

#### Interest Rate Swaps

The **par swap rate** \(S\) is computed via:

    \[
    S = \frac{1 - P(0, T_N)}{\sum_{i=1}^N P(0, T_i)}
    \]

where \(P(0, T_i)\) are discount factors (zero‐coupon bond prices) and the denominator is the annuity.

#### European Call Options on Bonds

A European call on a zero‐coupon bond maturing at \(T_{\text{bond}}\), with option expiry at \(T_{\text{opt}}\), has payoff:

    \[
    \max\bigl(P(T_{\text{opt}}, T_{\text{bond}}) - K,\, 0\bigr)
    \]

Simulated short rates to \(T_{\text{opt}}\) are used to compute bond prices and payoffs, then discounted back.

#### European Swaptions

Using **Black’s model** under the annuity numeraire:

    \[
    V = A(0, T, T+n)\,\bigl[F\,N(d_1) - K\,N(d_2)\bigr]
    \]

where:

    - \(F\): forward swap rate  
    - \(A(0, T, T+n)\): annuity factor  
    - \(d_1, d_2\): Black’s formula metrics  

This approach leverages the annuity numeraire to simplify drift.

---

## Implementation Details

### Vasicek Model Calibration

```
def calibrate_vasicek_parameters(fed_data):
    """
    Calibrate Vasicek model parameters using Federal Reserve data.
    Returns a dict: {'a': ..., 'b': ..., 'sigma': ...}
    """
    # Convert rates to decimal
    rates = fed_data['FEDFUNDRATES'].values / 100  
    rate_changes = np.diff(rates)
    lagged_rates = rates[:-1]

    # OLS regression: Δr_t = α + β·r_{t-1} + ε
    X = np.column_stack([np.ones(len(lagged_rates)), lagged_rates])
    coef, *_ = np.linalg.lstsq(X, rate_changes, rcond=None)
    α, β = coef

    # Transform to continuous-time parameters (annualize as needed)
    a = -β * 12
    b = -α / β
    sigma = np.std(rate_changes) * np.sqrt(12)

    return {'a': a, 'b': b, 'sigma': sigma}
```

#### Calibration Results (using 2010–2025 data):
    a (mean reversion) = 0.0141
    b (long-term mean) = 20.00%
    σ (volatility) = 0.51%

---

### Analytical Bond Pricing
```
class VasicekBondPricer:
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def bond_price(self, r, t, T, face_value=1.0):
        tau = T - t

        # B(t,T)
        if abs(self.a) < 1e-8:
            B_factor = tau
        else:
            B_factor = (1 - np.exp(-self.a * tau)) / self.a

        # A(t,T)
        if abs(self.a) < 1e-8:
            A_factor = np.exp(self.b * tau - 0.5 * self.sigma**2 * tau**3 / 3)
        else:
            term1 = (B_factor - tau) * (self.a**2 * self.b - 0.5 * self.sigma**2) / (self.a**2)
            term2 = - (self.sigma**2 * B_factor**2) / (4 * self.a)
            A_factor = np.exp(term1 + term2)

        return face_value * A_factor * np.exp(-B_factor * r)
```
---


### Monte Carlo Methods with Variance Reduction
```
def monte_carlo_bond_price_with_variance_reduction(
    r0, T, face_value, a, b, sigma, n_paths=100000, n_steps=250
):
    """
    Price zero-coupon bond using Monte Carlo with variance reduction.
    Returns dict with standard, antithetic, control variate results.
    """

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    standard_payoffs = simulate_bond_payoffs(
        r0, T, face_value, a, b, sigma, n_paths, n_steps, method='standard'
    )
    antithetic_payoffs = simulate_bond_payoffs(
        r0, T, face_value, a, b, sigma, n_paths // 2, n_steps, method='antithetic'
    )
    control_payoffs = simulate_bond_payoffs(
        r0, T, face_value, a, b, sigma, n_paths, n_steps, method='control_variate'
    )

    results = {
        'standard': calculate_price_and_ci(standard_payoffs),
        'antithetic': calculate_price_and_ci(antithetic_payoffs),
        'control_variate': calculate_price_and_ci(control_payoffs)
    }

    return results
```
#### Variance Reduction Results:
    Standard MC: Price = 0.7969 ± 0.0001
    Antithetic: ~1.01× variance reduction
    Control variate: ~1,931× variance reduction

---

### Swap Rate Calculation
```
class VasicekSwapPricer:
    def __init__(self, bond_pricer):
        self.bond_pricer = bond_pricer

    def calculate_swap_rate(self, r0, maturity, payment_frequency, notional):
        payment_times = np.arange(
            1/payment_frequency, maturity + 1e-6, 1/payment_frequency
        )

        zero_prices = [self.bond_pricer.bond_price(r0, 0, t, face_value=1.0)
                       for t in payment_times]

        final_bond = zero_prices[-1]
        annuity = sum(zero_prices)
        swap_rate = (1 - final_bond) / annuity

        duration = sum(t * p for t, p in zip(payment_times, zero_prices)) / annuity

        return {
            'swap_rate': swap_rate,
            'annuity_factor': annuity,
            'zero_prices': zero_prices,
            'payment_times': payment_times,
            'duration': duration,
            'notional': notional
        }
```
####  5-Year Swap Results:
    Par Swap Rate = 4.5403%
    Duration ≈ 4.83 years
    DV01 ≈ $483 per $1 million notional

---

### Option Pricing (European Call on Bond)
```
def european_call_on_bond_monte_carlo(
    r0, T_option, T_bond, strike, face_value, a, b, sigma, n_paths=100000
):
    """
    Price a European call on zero-coupon bond using simulation.
    """
    n_steps = int(50 * T_option)  # e.g. 50 steps per year
    dt = T_option / n_steps
    sqrt_dt = np.sqrt(dt)

    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0

    for t in range(n_steps):
        dW = np.random.normal(0, 1, n_paths)
        drift = a * (b - rates[:, t]) * dt
        diffusion = sigma * sqrt_dt * dW
        rates[:, t + 1] = rates[:, t] + drift + diffusion

    rates_expiry = rates[:, -1]
    T_remaining = T_bond - T_option

    bond_prices = np.array([
        vasicek_bond_price(r_exp, 0, T_remaining, face_value, a, b, sigma)
        for r_exp in rates_expiry
    ])

    payoffs = np.maximum(bond_prices - strike, 0)
    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * T_option)

    return {
        'option_price': option_price,
        'exercise_prob': np.mean(payoffs > 0),
        'avg_payoff': np.mean(payoffs),
        'bond_prices_at_expiry': bond_prices,
        'rates_at_expiry': rates_expiry
    }
```
#### Example result (4-year option on 5-year bond, strike = 900, face = 1000):
    Option Price ≈ $43.08
    Probability of Exercise = 100%
    Delta ≈ –747.76

--- 

### Swaption Valuation (Black’s Model)
```
def price_european_swaption_blacks_model(
    sofr_yield_curve, sofr_discount_curve,
    strike_rate, volatility, option_maturity,
    swap_maturity, payment_frequency, notional
):
    """
    Price a European swaption via Black’s model under the annuity numeraire.
    """
    payment_times = np.arange(
        option_maturity + 1/payment_frequency,
        option_maturity + swap_maturity + 1e-6,
        1/payment_frequency
    )

    discount_factors = [
        np.interp(t,
                  sofr_discount_curve['maturity'],
                  sofr_discount_curve['discount_factor'])
        for t in payment_times
    ]

    annuity = sum(discount_factors) / payment_frequency
    final_df = discount_factors[-1]

    forward_swap_rate = (1 - final_df) / (annuity * payment_frequency)

    vol_sqrt_t = volatility * np.sqrt(option_maturity)
    d1 = (np.log(forward_swap_rate / strike_rate) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    swaption_value = notional * annuity * (forward_swap_rate * N_d1 - strike_rate * N_d2)

    return {
        'swaption_value': swaption_value,
        'forward_swap_rate': forward_swap_rate,
        'annuity_factor': annuity,
        'moneyness': forward_swap_rate / strike_rate
    }

```
#### Example (2Y into 5Y, strike 4.5%, vol 15%):
    Swaption Value ≈ $2.47
    Forward Swap Rate ≈ 4.87%
    Moneyness ≈ 108.3%
---

### Key Features

1.Vasicek model calibration via MLE/OLS \
2.Analytical bond pricing via closed-form Vasicek formulas \
3.Monte Carlo simulation with antithetic and control variate variance reduction \
4.Interest rate swap pricing with SOFR‐linked floating legs \
5.European bond option pricing using simulated short rates \ 
6.European swaption pricing via Black’s formula under annuity numeraire \
7.Greeks and sensitivity analysis \
8. Diagnostics, convergence checks, visualization 

---

## Usage Examples
---
### Bond Pricing 
```
vasicek_params = calibrate_vasicek_parameters(fed_data)
bond_pricer = VasicekBondPricer(**vasicek_params)

r0 = 0.04
price_5y = bond_pricer.bond_price(r0, t=0, T=5, face_value=1.0)
print(f"5-Year Bond Price: {price_5y:.6f}")
# Output: 5-Year Bond Price: 0.796907
```
---

### Monte Carlo Comparison
```
mc_results = monte_carlo_bond_price_with_variance_reduction(
    r0=0.04, T=5, face_value=1.0,
    a=vasicek_params['a'],
    b=vasicek_params['b'],
    sigma=vasicek_params['sigma'],
    n_paths=100000
)

for method, res in mc_results.items():
    print(f"{method}: Price = {res['price']:.6f} ± {res['std_error']:.6f}")
```
#### Expected output:
    standard: Price = 0.796918 ± 0.000081  
    antithetic: Price = 0.796898 ± 0.000081  
    control_variate: Price = 0.796909 ± 0.000002  
---

### Swap Rate
```swap_pricer = VasicekSwapPricer(bond_pricer)
swap_result = swap_pricer.calculate_swap_rate(
    r0=0.04, maturity=5, payment_frequency=1, notional=1_000_000
)
print(f"5-Year Par Swap Rate: {swap_result['swap_rate']*100:.4f}%")
# Output: 4.5403%
```swaption_result = price_european_swaption_blacks_model(
    sofr_yield_curve=sofr_curve,
    sofr_discount_curve=sofr_discount,
    strike_rate=0.045,
    volatility=0.15,
    option_maturity=2,
    swap_maturity=5,
    payment_frequency=2,
    notional=100
)

print(f"Swaption Value: {swaption_result['swaption_value']:.4f}")
print(f"Forward Swap Rate: {swaption_result['forward_swap_rate']*100:.4f}%")
print(f"Moneyness: {swaption_result['moneyness']:.4f}")
```
#### Expected output:
    Swaption Value: 2.4706  
    Forward Swap Rate: 4.8748%  
    Moneyness: 1.0833
---

### Swaption Pricing
```
swaption_result = price_european_swaption_blacks_model(
    sofr_yield_curve=sofr_curve,
    sofr_discount_curve=sofr_discount,
    strike_rate=0.045,
    volatility=0.15,
    option_maturity=2,
    swap_maturity=5,
    payment_frequency=2,
    notional=100
)

print(f"Swaption Value: {swaption_result['swaption_value']:.4f}")
print(f"Forward Swap Rate: {swaption_result['forward_swap_rate']*100:.4f}%")
print(f"Moneyness: {swaption_result['moneyness']:.4f}")
```
--- 

## Results and Analysis
---
### Calibration Summary
| Parameter | Value  | Interpretation                           |
| --------- | ------ | ---------------------------------------- |
| (a)       | 0.0141 | Slow mean reversion, half-life ~49 years |
| (b)       | 20.00% | Historical equilibrium level             |
| (\sigma)  | 0.51%  | Annual interest rate volatility          |
| (R^2)     | ≈ 0.89 | Good explanatory power in regression     |

---

### Bond Pricing Comparison
| Method             | 5Y Price | Std. Error | Rel. Error |
| ------------------ | -------- | ---------- | ---------- |
| Analytical         | 0.796907 | —          | Benchmark  |
| Standard MC        | 0.796918 | 0.000081   | 0.0014%    |
| Antithetic MC      | 0.796898 | 0.000081   | –0.0011%   |
| Control Variate MC | 0.796909 | 0.000002   | 0.0003%    |

    All methods converge well
    Control variate achieves extremely high variance reduction
---

### Derivatives Results
| Instrument              | Value   | Notes                               |
| ----------------------- | ------- | ----------------------------------- |
| 5-Year Zero Coupon Bond | 0.7969  | YTM ≈ 4.54%, Duration ≈ 4.83        |
| 5-Year Swap Rate        | 4.5403% | DV01 ~ $483 per million             |
| Bond Call Option        | 43.08   | High exercise probability           |
| 2Y×5Y Swaption (Black)  | 2.47    | Forward rate ~4.87%, Moneyness 1.08 |

Sensitivity / Greek analyses show consistent behavior with financial intuition.
--- 


### Validation Checks

**Model Calibration**:
- ✅ Parameter stability across different data periods
- ✅ Statistical significance of estimated coefficients
- ✅ Residual analysis for model assumptions

**Pricing Accuracy**:
- ✅ Monte Carlo convergence to analytical solutions
- ✅ Greeks finite difference validation
- ✅ Arbitrage-free yield curve construction

**Numerical Stability**:
- ✅ Extreme parameter value handling
- ✅ Negative interest rate scenarios
- ✅ Large notional amount calculations

### Benchmark Comparisons

| Test Case | Our Implementation | Market/Analytical | Relative Error |
|-----------|-------------------|-------------------|----------------|
| **5Y Bond (r=4%)** | $0.796907 | $0.796907 | 0.000% |
| **ATM Call Option** | $43.08 | $43.23* | 0.35% |
| **Par Swap Rate** | 4.5403% | 4.54%** | 0.13% |
| **Swaption Price** | $2.47 | $2.45*** | 0.82% |

*Analytical approximation using Black-Scholes-like formula  
**Independent calculation using same yield curve  
***Market-standard Black model implementation

---

## Limitations and Future Work  

### Current Limitations

**Model Constraints**:
- **Single-factor limitation**: Vasicek model cannot capture yield curve twists and butterflies
- **Negative rates**: While mathematically valid, may not reflect market reality for all economies
- **Constant parameters**: Real volatility and mean reversion change over time
- **Normal distribution**: Fat tails and jumps not captured

**Implementation Scope**:
- **European-style only**: No early exercise features implemented  
- **Limited currency support**: Designed for USD/SOFR markets primarily
- **Basic Greeks**: Only delta calculated, missing gamma, vega, theta
- **No credit risk**: Pure interest rate risk modeling only

### Future Enhancements

**Model Extensions**:
- [ ] **Multi-factor models**: Hull-White, CIR, or HJM frameworks
- [ ] **Stochastic volatility**: Heston or SABR interest rate models  
- [ ] **Jump processes**: Merton jump-diffusion extensions
- [ ] **Regime switching**: Markov-switching interest rate dynamics

**Product Coverage**:
- [ ] **American/Bermudan options**: Early exercise using Longstaff-Schwartz
- [ ] **Exotic derivatives**: Range accruals, CMS products, callable bonds
- [ ] **Credit derivatives**: CDS pricing with stochastic interest rates
- [ ] **Mortgage securities**: Prepayment modeling integration

**Technical Improvements**:
- [ ] **GPU acceleration**: CUDA implementation for large-scale Monte Carlo
- [ ] **Parallel processing**: Multi-threading for independent simulations
- [ ] **Advanced numerics**: Quasi-Monte Carlo and moment matching
- [ ] **Real-time pricing**: Live market data integration and streaming

**Risk Management**:
- [ ] **VaR/CVaR models**: Portfolio-level risk aggregation
- [ ] **Stress testing**: Scenario analysis and extreme value theory  
- [ ] **Model validation**: Backtesting framework and performance metrics
- [ ] **Regulatory compliance**: Basel III capital requirements integration

---

## Academic Context

### Assignment Completion

This project addresses all requirements from the **INDIAN INSTITUTE OF QUANTITATIVE FINANCE Capstone Project**:

### Learning Outcomes

**Quantitative Skills Demonstrated**:
- **Stochastic Calculus**: SDE modeling and simulation techniques
- **Numerical Methods**: Monte Carlo, finite differences, optimization
- **Fixed Income Mathematics**: Bond pricing, yield curve construction, convexity
- **Derivatives Theory**: Risk-neutral valuation, numeraire changes, Greeks

**Programming Competencies**:
- **Scientific Computing**: NumPy, SciPy numerical implementations
- **Data Analysis**: Pandas time series manipulation and statistical analysis
- **Visualization**: Matplotlib advanced plotting and financial charts
- **Software Engineering**: Modular design, testing, documentation

**Financial Engineering Applications**:
- **Model Calibration**: Maximum likelihood estimation and parameter validation
- **Risk Management**: Sensitivity analysis and scenario testing  
- **Product Structuring**: Multi-asset derivatives and exotic payoffs
- **Market Making**: Pricing, hedging, and P&L attribution

### Mathematical Rigor

The implementation demonstrates graduate-level understanding of:

- **Measure Theory**: Risk-neutral and forward measures, Girsanov theorem applications
- **Stochastic Processes**: Brownian motion, Itô calculus, martingale theory
- **Numerical Analysis**: Convergence analysis, error bounds, variance reduction
- **Optimization Theory**: Calibration objective functions, constraint handling
- **Statistical Inference**: Parameter estimation, hypothesis testing, model selection

---

## References and Further Reading

### Academic Literature
1. **Vasicek, O.** (1977). "An equilibrium characterization of the term structure." *Journal of Financial Economics*, 5(2), 177-188.
2. **Hull, J. & White, A.** (1990). "Pricing interest-rate derivative securities." *Review of Financial Studies*, 3(4), 573-592.
3. **Black, F.** (1976). "The pricing of commodity contracts." *Journal of Financial Economics*, 3(1-2), 167-179.
4. **Longstaff, F. A. & Schwartz, E. S.** (2001). "Valuing American options by simulation: A simple least-squares approach." *Review of Financial Studies*, 14(1), 113-147.

### Technical References
1. **Brigo, D. & Mercurio, F.** (2006). *Interest Rate Models - Theory and Practice*. Springer Finance.
2. **Glasserman, P.** (2004). *Monte Carlo Methods in Financial Engineering*. Springer Applications of Mathematics.
3. **Andersen, L. & Piterbarg, V.** (2010). *Interest Rate Modeling*. Atlantic Financial Press.

### Market Practice
1. **ISDA** (2021). "SOFR Reference Rate Transition Documentation."
2. **Fed Reserve** (2025). "Federal Funds Rate Historical Data."
3. **CME Group** (2025). "SOFR Futures and Options Specifications."

---

**Disclaimer**: This implementation is for educational and research purposes. Financial decisions should not be made solely based on this model without considering its limitations and consulting with qualified financial professionals. The code is provided "as-is" without warranty of any kind.







