# Interactive Real-Time Portfolio Optimization Dashboard
### 📊 Production-Grade Markowitz Mean-Variance Optimization Engine

An institutional-grade mathematical dashboard built using **Streamlit**, **SciPy Scientific Optimization Suites**, and **Plotly Dark Engines**. The terminal computes live asset covariance structures to map out the non-linear Markowitz Efficient Frontier, locating absolute optimization anchor points via Sequential Least Squares Programming (SLSQP).

---

## 🔬 Core Portfolio Engineering Mathematics

This framework executes structural matrix calculus and quadratic programming natively using NumPy and SciPy arrays rather than relying on high-level abstract finance packages, demonstrating basic financial engineering logic.

### 1. The Vectorized Risk-Return Space
Given a weight allocation vector for \(N\) assets, \(\mathbf{w} = [w_1, w_2, \dots, w_N]^T\), the asset allocation must obey strict standard budget limits:

```math
\sum_{i=1}^{N} w_i = \mathbf{w}^T \mathbf{1} = 1.0 \quad \text{where } w_i \geq 0 \ \forall i
```

*   **Expected Portfolio Return (\(E(R_p)\)):** Expressed as the inner product of the weights and annualized log mean returns vector \(\boldsymbol{\mu}\):
    ```math
    E(R_p) = \mathbf{w}^T \boldsymbol{\mu} \times 252
    ```
*   **Portfolio Volatility (\(\sigma_p\)):** Calculated by projecting weights across the annualized sample covariance matrix \(\boldsymbol{\Sigma}\):
    ```math
    \sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} \times 252}
    ```

### 2. Non-Linear Optimization Objective Functions

The backend optimization engine utilizes the **SLSQP (Sequential Least Squares Programming)** algorithm to handle localized multi-variable constraints:

*   **A. Maximum Sharpe Ratio Tangency Portfolio:** Maximizes the excess return per unit of total risk relative to the risk-free baseline (\(R_f\)):
    ```math
    \max_{\mathbf{w}} \quad \frac{\mathbf{w}^T \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}}
    ```
    *Note: The optimizer minimizes the negative Sharpe Ratio inside execution loops to converge on the target allocation profile.*

*   **B. Global Minimum Variance (GMV) Portfolio:** Minimizes absolute historical portfolio variance, completely ignoring expected return vectors to establish a lower boundary for the risk profile:
    ```math
    \min_{\mathbf{w}} \quad \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}
    ```

### 3. Hyperbolic Efficient Frontier Mapping
The outer boundary curve represents the optimal risk-return tradeoff space. The script samples 40 distinct risk horizons between the GMV profile and maximum individual return parameters. For each target step \(R_{\text{target}}\), it executes a conditional minimization loop:

```math
\min_{\mathbf{w}} \quad \sigma_p(\mathbf{w}) \quad \text{subject to } \mathbf{w}^T \mathbf{1} = 1 \text{ and } \mathbf{w}^T \boldsymbol{\mu} = R_{\text{target}}
```

---

## 🎨 Interactive Interface & System Visualizations

The terminal features a premium **Academic Dark Terminal UI** custom-styled via raw CSS injections to match standard institutional risk layouts.

```text
       ┌────────────────────────────────────────────────────────┐
       │               USER CONFIGURATION SIDEBAR               │
       ├────────────────────────────────────────────────────────┤
       │ 🖥️ Stock Ticker Input Basket (Exactly 5 Assets)       │
       │ 🎚️ Live Risk-Free Interest Margin Slider (R_f)         │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                SCIPY OPTIMIZATION ENGINE               │
       ├────────────────────────────────────────────────────────┤
       │ 📥 Data Extraction: Live yFinance Daily Close Arrays   │
       │ 🧮 Matrix Processing: Annualized Covariance Matrix     │
       │ 🎯 SLSQP Solver Loops: Max Sharpe vs GMV Convergence   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                  FRONTEND RENDER LYS                   │
       ├────────────────────────────────────────────────────────┤
       │ 📊 4-Column Consolidated Performance Scorecard         │
       │ 📈 Interactive Plotly Hyperbolic Solution Chart        │
       │ 📋 Gradient-Mapped Asset Weights Allocation Matrix     │
       └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Production Tech Stack & Dependencies

*   **Dashboard Framework:** `Streamlit` (Dynamic Dark Mode UI Architecture)
*   **Scientific Compute Library:** `SciPy.optimize` (Sequential Least Squares Programming)
*   **Vectorization & Analytics:** `NumPy` & `Pandas` (Matrix Calculus Structures)
*   **Data Scraper Engine:** `yFinance` (Market Close Array Extraction)
*   **Graphics Engine:** `Plotly Graph Objects` (Interactive Hyperbola Mapping Space)

---

## 🚀 Native Local Execution Blueprint

### 1. Clone Repo & Initialize Environment
```bash
git clone https://github.com
cd Quantitative-Asset-Allocation-Terminal
pip install streamlit yfinance pandas numpy scipy plotly
```

### 2. Launch the Application Terminal Natively
```bash
streamlit run app.py
```
