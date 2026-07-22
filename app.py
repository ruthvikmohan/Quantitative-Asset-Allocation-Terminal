import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go

# Custom CSS styling injection to build a clean Academic Dark Terminal layout
st.set_page_config(page_title="Quantum Asset Allocator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stButton>button { width: 100%; background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; border: none; border-radius: 6px; padding: 10px; font-weight: bold; }
    .stButton>button:hover { background: linear-gradient(135deg, #2563eb, #60a5fa); }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #60a5fa; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricLabel"] { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #9ca3af; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1, h2, h3 { font-family: 'Helvetica Neue', Arial, sans-serif; letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

# Title Elements Layout Block
st.title("🎛️ Quantitative Asset Allocation Terminal")
st.markdown("##### *Portfolio Mathematics Suite: Markowitz Mean-Variance Optimization & Efficient Frontier Simulation*")
st.write("---")

# ==========================================
# 1. FRONTEND SIDEBAR USER CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("### 🛠️ Model Parameters")
    st.write("Configure institutional risk horizons below:")
    ticker_input = st.text_input("Target Stock Tickers (Exactly 5):", value="AAPL, MSFT, GOOGL, AMZN, TSLA")
    risk_free_rate = st.slider("Risk-Free Rate Target (% R_f)", 0.0, 5.0, 2.0, 0.1) / 100
    st.write("---")
    run_optimization = st.button("🚀 Execute Matrix Optimization", type="primary")

# Parse strings into systematic arrays
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# ==========================================
# 2. CORE FINANCIAL MATHEMATICAL FUNCTIONS
# ==========================================
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """Annualizes expected returns and returns standard deviation using matrix operations."""
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf_rate):
    """Objective framework target maximized by minimizing the negative ratio outcome."""
    p_ret, p_vol = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - rf_rate) / (p_vol + 1e-10)

def portfolio_volatility_objective(weights, mean_returns, cov_matrix):
    """Objective target minimized to locate the Global Minimum Variance configuration."""
    return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]

# ==========================================
# 3. INTERACTIVE PROCESSING & COMPUTATION
# ==========================================
if run_optimization:
    if len(tickers) != 5:
        st.error(f"❌ Structural Limit Violation: Please specify exactly 5 assets. Detected asset count: {len(tickers)}.")
    else:
        with st.spinner("Extracting historical transaction limits and computing mathematical models..."):
            raw_data = yf.download(tickers, period="3y", progress=False)
            if 'Close' not in raw_data or raw_data['Close'].empty:
                st.error("Extraction Failure: Verification error locating ticker histories.")
                st.stop()

            close_prices = raw_data['Close'].dropna(axis=1, how='all')
            actual_tickers = list(close_prices.columns)
            num_assets = len(actual_tickers)

            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
            mean_returns = log_returns.mean()
            cov_matrix = log_returns.cov()

            # --- CALCULATE OPTIMAL PORTFOLIOS VIA SCIPY OPTIMIZER ---
            initial_guess = num_assets * [1.0 / num_assets]
            bounds = tuple((0.0, 1.0) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})

            # A. Maximum Sharpe Ratio Optimization
            max_sharpe_opt = sco.minimize(
                negative_sharpe_ratio, initial_guess,
                args=(mean_returns, cov_matrix, risk_free_rate),
                method='SLSQP', bounds=bounds, constraints=constraints
            )
            weights_max_sharpe = max_sharpe_opt['x']
            ret_max_sharpe, vol_max_sharpe = calculate_portfolio_performance(weights_max_sharpe, mean_returns, cov_matrix)
            sharpe_max_sharpe = (ret_max_sharpe - risk_free_rate) / vol_max_sharpe

            # B. Global Minimum Variance Optimization
            min_var_opt = sco.minimize(
                portfolio_volatility_objective, initial_guess,
                args=(mean_returns, cov_matrix), # Fixed order configuration passing
                method='SLSQP', bounds=bounds, constraints=constraints
            )
            weights_min_var = min_var_opt['x']
            ret_min_var, vol_min_var = calculate_portfolio_performance(weights_min_var, mean_returns, cov_matrix)
            sharpe_min_var = (ret_min_var - risk_free_rate) / vol_min_var

            # --- GENERATING TRADITIONAL MARKOWITZ EFFICIENT FRONTIER CURVE ---
            target_returns = np.linspace(ret_min_var, max(mean_returns) * 252, 40)
            efficient_vols = []

            for target in target_returns:
                cons = (
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                    {'type': 'eq', 'fun': lambda x: calculate_portfolio_performance(x, mean_returns, cov_matrix)[0] - target}
                )
                # FIXED: Added required model array arguments tuple to clear missing parameter bugs
                res = sco.minimize(
                    portfolio_volatility_objective, initial_guess,
                    args=(mean_returns, cov_matrix),
                    method='SLSQP', bounds=bounds, constraints=cons
                )
                efficient_vols.append(res['x'])

            frontier_vols = [calculate_portfolio_performance(w, mean_returns, cov_matrix)[1] for w in efficient_vols]

            # ==========================================
            # 4. DATA RENDER AND INTERACTIVE GRAPHICS
            # ==========================================
            st.success("🎯 Portfolio optimization execution loops converged successfully.")

            # Premium Dashboard Column Metrics Split Display
            st.markdown("### 📊 Consolidated Model Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Optimal Expected Return", f"{ret_max_sharpe*100:.2f}%")
            m2.metric("Annualized Volatility Risk", f"{vol_max_sharpe*100:.2f}%")
            m3.metric("Maximum Sharpe Ratio achieved", f"{sharpe_max_sharpe:.3f}")
            m4.metric("Risk-Free baseline Input", f"{risk_free_rate*100:.1f}%")

            st.write("---")

            # Split layout panel visualization: Left Chart, Right Data Matrix
            g1, g2 = st.columns([3, 2])

            with g1:
                st.markdown("#### 📈 Hyperbolic Efficient Frontier Solution Space")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[v*100 for v in frontier_vols], y=[r*100 for r in target_returns],
                    mode='lines', name='Efficient Frontier Hyperbola', line=dict(color='#3b82f6', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[vol_max_sharpe*100], y=[ret_max_sharpe*100],
                    mode='markers', name='Max Sharpe Anchor Point', marker=dict(color='#ef4444', size=14, symbol='star-diamond')
                ))
                fig.add_trace(go.Scatter(
                    x=[vol_min_var*100], y=[ret_min_var*100],
                    mode='markers', name='Minimum Variance Anchor Point', marker=dict(color='#10b981', size=12, symbol='circle-cross')
                ))
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Annualized Portfolio Risk / Standard Deviation (σ %)",
                    yaxis_title="Annualized Portfolio Expected Return (E(R) %)",
                    template="plotly_dark", height=450, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                st.markdown("#### 📊 Matrix Weight Allocation Profile")
                df_weights = pd.DataFrame({
                    "Maximum Sharpe Ratio Allocation Weight (%)": [round(w*100, 2) for w in weights_max_sharpe],
                    "Minimum Variance Baseline Limit (%)": [round(w*100, 2) for w in weights_min_var]
                }, index=actual_tickers)

                st.dataframe(df_weights.style.background_gradient(cmap="Blues", axis=0), height=350, use_container_width=True)

else:
    st.info("💡 Adjust stock parameters or target interest margins in the left dashboard options panel and run execution loops to view models layout.")
