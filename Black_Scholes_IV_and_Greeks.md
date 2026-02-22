# Implied Volatility (IV) and Greeks -- Black-Scholes Model

This document contains all required formulas and information to
calculate **Implied Volatility (IV)** and **Option Greeks** using the
Black-Scholes model for European stock options (e.g., NSE stock options
like TCS).

------------------------------------------------------------------------

## 1. Required Inputs

From dataset:

-   **S** = Underlying Price (`UndrlygPric`)
-   **K** = Strike Price (`StrkPric`)
-   **Market Price (C or P)** = Option Close/Last Price (`ClsPric` or
    `LastPric`)
-   **T** = Time to Expiry (in years)
-   **r** = Risk-Free Interest Rate (annualized, decimal form)
-   **q** = Dividend Yield (annualized, decimal form)
-   **σ** = Implied Volatility (unknown, must be solved)

------------------------------------------------------------------------

## 2. Time to Expiry

T = (Expiry Date − Trade Date) / 365

------------------------------------------------------------------------

## 3. Black-Scholes Variables

d1 = \[ ln(S/K) + (r − q + 0.5σ²)T \] / (σ√T)

d2 = d1 − σ√T

------------------------------------------------------------------------

## 4. Black-Scholes Pricing Formula

### Call Option (CE)

C = S e\^(−qT) N(d1) − K e\^(−rT) N(d2)

### Put Option (PE)

P = K e\^(−rT) N(−d2) − S e\^(−qT) N(−d1)

Where: - N(.) = Cumulative normal distribution - φ(.) = Standard normal
probability density function

------------------------------------------------------------------------

## 5. Implied Volatility (IV)

There is no closed-form formula.

Solve numerically:

Market Price = Black-Scholes Price

Using methods like: - Newton-Raphson - Brent method - Bisection

Newton-Raphson Update:

σ_new = σ − ( BS(σ) − MarketPrice ) / Vega

------------------------------------------------------------------------

## 6. Greeks Formulas

Let:

φ(d1) = (1 / √(2π)) e\^(−d1²/2)

------------------------------------------------------------------------

### Delta

Call: Δc = e\^(−qT) N(d1)

Put: Δp = e\^(−qT)(N(d1) − 1)

------------------------------------------------------------------------

### Gamma

Γ = e\^(−qT) φ(d1) / (S σ √T)

(Same for Call and Put)

------------------------------------------------------------------------

### Vega

Vega = S e\^(−qT) φ(d1) √T

(Note: Divide by 100 for per 1% volatility change)

------------------------------------------------------------------------

### Theta

Call:

Θc = − (S φ(d1) σ e\^(−qT)) / (2√T) − r K e\^(−rT) N(d2) + q S e\^(−qT)
N(d1)

Put:

Θp = − (S φ(d1) σ e\^(−qT)) / (2√T) + r K e\^(−rT) N(−d2) − q S e\^(−qT)
N(−d1)

------------------------------------------------------------------------

### Rho

Call:

ρc = K T e\^(−rT) N(d2)

Put:

ρp = −K T e\^(−rT) N(−d2)

------------------------------------------------------------------------

## 7. Practical Notes

-   Use annualized volatility.
-   Ensure T is in years.
-   Use settlement price for higher accuracy.
-   For Indian stock options (European style), Black-Scholes is
    appropriate.
-   Use consistent day count (365 or trading days, but remain
    consistent).
-   Dividend yield is important for stocks like TCS.

------------------------------------------------------------------------

End of Document.
