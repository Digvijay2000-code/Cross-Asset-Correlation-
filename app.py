import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

# --- 1. PLUG & PLAY CONFIGURATION ---
# Add or remove assets here. The rest of the app dynamically adjusts.
ASSET_MAP = {
    "S&P 500": "^GSPC",
    "US Treasuries (IEF)": "IEF", # 7-10 Year Treasury ETF proxy
    "DXY Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "WTI Crude": "CL=F",
    "Bitcoin": "BTC-USD"
}

# Timeframes mapped to approximate trading days
TIMEFRAMES = {
    "6 Months": 126,
    "1 Year": 252,
    "5 Years": 1260,
    "10 Years": 2520
}

# --- 2. AUTO-UPDATING DATA ENGINE ---
# ttl=3600 means the app fetches new data hourly automatically on page load
@st.cache_data(ttl=3600)
def load_data(tickers):
    # Fetch 10 years of data to support long-term baseline calculations
    raw_data = yf.download(list(tickers.values()), period="10y")['Close']

    # Rename columns to human-readable names
    raw_data = raw_data.rename(columns={v: k for k, v in tickers.items()})

    # Forward fill to align assets that trade on different holiday schedules
    df_clean = raw_data.ffill().dropna()

    # Calculate daily log returns for accurate statistical modeling
    returns = np.log(df_clean / df_clean.shift(1)).dropna()
    return returns

# --- 3. DASHBOARD UI ---
st.set_page_config(layout="wide", page_title="Macro Correlation Detector")
st.title("Cross-Asset Correlation & Anomaly Detector")
st.markdown("Flags statistical breakdowns between macro asset relationships.")

returns_df = load_data(ASSET_MAP)

col1, col2, col3, col4 = st.columns(4)
base_asset = col1.selectbox("Base Asset (Market)", list(ASSET_MAP.keys()), index=0)
target_asset = col2.selectbox("Target Asset", list(ASSET_MAP.keys()), index=1)
rolling_window = col3.number_input("Rolling Window (Days)", min_value=10, max_value=252, value=60)
z_threshold = col4.number_input("Anomaly Z-Score Threshold", min_value=1.0, max_value=5.0, value=2.0)

if base_asset == target_asset:
    st.warning("Please select two different assets.")
    st.stop()

# --- 4. ANALYTICS & MATH ENGINE ---
pair_returns = returns_df[[base_asset, target_asset]].copy()

# Rolling Correlation
rolling_corr = pair_returns[base_asset].rolling(window=rolling_window).corr(pair_returns[target_asset])

# Rolling Beta
rolling_cov = pair_returns[base_asset].rolling(window=rolling_window).cov(pair_returns[target_asset])
rolling_var = pair_returns[base_asset].rolling(window=rolling_window).var()
rolling_beta = rolling_cov / rolling_var

pair_returns['Rolling Corr'] = rolling_corr
pair_returns['Rolling Beta'] = rolling_beta

st.subheader(f"Anomaly Detection: {target_asset} vs {base_asset}")

results = []
current_corr = rolling_corr.iloc[-1]
current_beta = rolling_beta.iloc[-1]

for tf_name, days in TIMEFRAMES.items():
    if len(pair_returns) < days:
        continue

    tf_slice = pair_returns.iloc[-days:].dropna()

    # Overall significance (Pearson r and p-value) for the timeframe
    if len(tf_slice) > 2:
        r, p_value = stats.pearsonr(tf_slice[base_asset], tf_slice[target_asset])
        is_sig = "Yes" if p_value < 0.05 else "No"
    else:
        r, p_value, is_sig = np.nan, np.nan, "N/A"

    # Historical baseline for the rolling correlation
    hist_corr_mean = tf_slice['Rolling Corr'].mean()
    hist_corr_std = tf_slice['Rolling Corr'].std()

    # Z-Score Anomaly Detection
    if hist_corr_std > 0:
        z_score = (current_corr - hist_corr_mean) / hist_corr_std
        is_anomaly = "🚨 YES" if abs(z_score) > z_threshold else "No"
    else:
        z_score = 0
        is_anomaly = "No"

    results.append({
        "Lookback": tf_name,
        "Current Rolling Corr": f"{current_corr:.2f}",
        "Hist. Norm (Mean)": f"{hist_corr_mean:.2f}",
        "Z-Score (Deviation)": f"{z_score:.2f}",
        "Anomaly Flag": is_anomaly,
        "Overall Beta": f"{current_beta:.2f}",
        "Statistically Sig? (p < 0.05)": is_sig
    })

st.table(pd.DataFrame(results))

# Plotting the dynamic relationship
st.line_chart(pair_returns[['Rolling Corr', 'Rolling Beta']].dropna(), height=400)