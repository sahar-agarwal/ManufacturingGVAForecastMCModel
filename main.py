import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Introduction to Data Analytics and Machine Learning (ECO-3401): India Future Trajectory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .kpi-card {
            background: #ffffff;
            box-shadow: 0 1px 6px rgba(0,0,0,0.04);
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.25rem;
            margin-bottom: 0.15rem;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("India Future Trajectory")
st.caption("Manufacturing sector forecasting with ARIMAX, Monte Carlo simulation, and scenario analysis")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "IDAML_Compiled.csv"
REQUIRED_COLS = ["MFG_GVA", "MFG_IIP", "WPI", "BANK_CREDIT", "REER", "EXPORTS", "GOVT_CAPEX"]


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found in the app folder")

    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError("DATE column not found in the file.")

    def parse_quarter(x):
        x = str(x).strip().upper().replace(" ", "")
        if "Q" in x:
            parts = x.split("Q")
            if len(parts) != 2:
                return pd.NaT
            try:
                year, q = int(parts[0]), int(parts[1])
                month = {1: 3, 2: 6, 3: 9, 4: 12}.get(q)
                if month is None:
                    return pd.NaT
                return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.QuarterEnd(0)
            except ValueError:
                return pd.NaT
        return pd.NaT

    df["DATE"] = df["DATE"].apply(parse_quarter)
    df = df.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE")
    return df


try:
    df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

clean = df[REQUIRED_COLS].copy()
clean = clean.replace([np.inf, -np.inf], np.nan)

for col in REQUIRED_COLS:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

for col in [c for c in REQUIRED_COLS if c != "MFG_GVA"]:
    clean[col] = clean[col].interpolate(method="linear", limit_direction="both")
    clean[col] = clean[col].ffill().bfill()

clean["MFG_GVA"] = clean["MFG_GVA"].interpolate(method="linear", limit_direction="both")
clean = clean.dropna(subset=["MFG_GVA"])

if len(clean) < 12:
    st.error("Not enough usable rows after cleaning. Check the DATE parsing or missing target values.")
    st.stop()

y = clean["MFG_GVA"]
exog = clean.drop(columns=["MFG_GVA"])

#  Sidebar 
st.sidebar.header("Model controls")

with st.sidebar.expander("ARIMAX settings", expanded=True):
    p = st.slider("AR order (p)", 0, 3, 1, help="Autoregressive lag order.")
    d = st.slider("Difference (d)", 0, 2, 1, help="Number of differencing steps.")
    q_order = st.slider("MA order (q)", 0, 3, 1, help="Moving average order.")
    sp = st.slider("Seasonal AR (P)", 0, 2, 1, help="Seasonal autoregressive order.")
    sd = st.slider("Seasonal difference (D)", 0, 1, 0, help="Seasonal differencing.")
    sq = st.slider("Seasonal MA (Q)", 0, 2, 1, help="Seasonal moving average order.")
    season_len = st.selectbox("Season length", [4], index=0, help="Quarterly data uses season length 4.")

with st.sidebar.expander("Simulation settings", expanded=True):
    mc_sims = st.slider("Monte Carlo simulations", 1000, 20000, 2000, step=500,
                         help="Number of simulated forecast paths.")
    run_mc = st.button("Run Monte Carlo", use_container_width=True,
                        help="Generate the 3-year simulation.")

with st.sidebar.expander("Display options", expanded=False):
    show_raw = st.checkbox("Show raw data sample", value=True,
                            help="Display a sample of the original dataset.")
    show_clean = st.checkbox("Show cleaned data sample", value=True,
                              help="Display a sample of cleaned data.")
    show_summary = st.checkbox("Show model summary", value=True,
                                help="Display the fitted ARIMAX summary.")

#  Dataset overview ─
st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Rows", f"{len(clean):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Start date", str(clean.index.min().date()))
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("End date", str(clean.index.max().date()))
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Features", str(len(REQUIRED_COLS) - 1))
    st.markdown('</div>', unsafe_allow_html=True)

#  Model fitting 
# FIX: cache_data (not cache_resource) so the cache key includes the param values
@st.cache_data
def fit_model(y_values, exog_values, exog_index, exog_columns, order, seasonal_order):
    """Re-fit whenever any slider changes. Accepts plain arrays for hashability."""
    y_ser = pd.Series(y_values, index=pd.to_datetime(exog_index))
    exog_df = pd.DataFrame(exog_values, index=pd.to_datetime(exog_index), columns=exog_columns)
    model = SARIMAX(
        y_ser,
        exog=exog_df,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


order = (p, d, q_order)
seasonal_order = (sp, sd, sq, season_len)

with st.spinner("Fitting ARIMAX model..."):
    fitted = fit_model(
        y.values,
        exog.values,
        exog.index.astype(str).tolist(),
        list(exog.columns),
        order,
        seasonal_order,
    )

st.success("Model fitted successfully")

#  Tabs ─
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecasts", "Scenarios", "Diagnostics", "Results & Interpretation"])

#  Tab 1 - Overview ─
with tab1:
    st.markdown('<div class="section-title">Data preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Inspect raw and cleaned inputs used by the model.</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if show_raw:
            st.write("Raw data sample")
            st.dataframe(df.tail(10), use_container_width=True)
    with col_b:
        if show_clean:
            st.write("Cleaned data sample")
            st.dataframe(clean.tail(10), use_container_width=True)

    st.markdown("#### Target series")
    st.line_chart(clean[["MFG_GVA"]], height=300)

    
#  Tab 2 · Forecasts 
with tab2:
    st.subheader("1-year forecast")
    h1 = 4
    future_dates_1y = pd.date_range(
        clean.index[-1] + pd.offsets.QuarterEnd(1), periods=h1, freq="Q"
    )

    exog_future_1y = pd.DataFrame(
        np.tile(exog.tail(4).mean().to_numpy(), (h1, 1)),
        index=future_dates_1y,
        columns=exog.columns,
    )

    fc1 = fitted.get_forecast(steps=h1, exog=exog_future_1y)
    mean1 = fc1.predicted_mean
    ci1 = fc1.conf_int()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=y.index, y=y, name="History",
        line=dict(color="steelblue", width=2)
    ))
    fig1.add_trace(go.Scatter(
        x=mean1.index, y=mean1.values, name="Forecast",
        line=dict(color="crimson", width=3)
    ))
    fig1.add_trace(go.Scatter(
        x=ci1.index, y=ci1.iloc[:, 1].values,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False
    ))
    fig1.add_trace(go.Scatter(
        x=ci1.index,
        y=ci1.iloc[:, 0].values,
        fill="tonexty",
        fillcolor="rgba(220,20,60,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI"
    ))
    fig1.update_layout(
        template="plotly_white",
        xaxis_title="Quarter",
        yaxis_title="MFG_GVA_GROWTH_RATE_%",
        hovermode="x unified",
        showlegend=True
    )
    fig1.update_xaxes(type="date")
    st.plotly_chart(fig1, use_container_width=True)

    st.download_button(
        "Download 1-year forecast CSV",
        pd.DataFrame({
            "quarter": mean1.index,
            "forecast": mean1.values,
            "lower": ci1.iloc[:, 0].values,
            "upper": ci1.iloc[:, 1].values,
        }).to_csv(index=False),
        file_name="short_term_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("3-year Monte Carlo forecast")
    h3 = 12
    future_dates_3y = pd.date_range(
        clean.index[-1] + pd.offsets.QuarterEnd(1), periods=h3, freq="Q"
    )

    resid = fitted.resid.dropna().to_numpy()
    resid = resid[np.isfinite(resid)]

    exog_last = exog.iloc[-1].to_numpy()
    exog_diff = exog.diff().dropna()
    exog_mu = exog_diff.mean().to_numpy()
    exog_sigma = exog_diff.std().to_numpy()
    exog_sigma = np.where(np.isfinite(exog_sigma) & (exog_sigma > 0), exog_sigma, 1.0)

    if run_mc:
        mc = np.zeros((mc_sims, h3))
        progress = st.progress(0)

        for i in range(mc_sims):
            x_path = np.zeros((h3, exog.shape[1]))
            prev = exog_last.copy()

            for t in range(h3):
                shock = np.random.normal(exog_mu, exog_sigma * 0.5)
                prev = prev + shock
                x_path[t] = prev

            x_df = pd.DataFrame(x_path, index=future_dates_3y, columns=exog.columns)
            pred = fitted.get_forecast(steps=h3, exog=x_df).predicted_mean.to_numpy()

            noise = np.random.choice(resid, size=h3, replace=True) if len(resid) > 0 else np.zeros(h3)
            mc[i] = pred + noise

            if i % max(1, mc_sims // 100) == 0:
                progress.progress(i / mc_sims)

        progress.progress(1.0)

        mc_mean = mc.mean(axis=0)
        mc_lo = np.percentile(mc, 2.5, axis=0)
        mc_hi = np.percentile(mc, 97.5, axis=0)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y.index, y=y, name="History",
            line=dict(color="steelblue", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=future_dates_3y, y=mc_mean, name="MC mean",
            line=dict(color="green", width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=future_dates_3y, y=mc_hi,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False
        ))
        fig2.add_trace(go.Scatter(
            x=future_dates_3y,
            y=mc_lo,
            fill="tonexty",
            fillcolor="rgba(0,128,0,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% interval"
        ))
        fig2.update_layout(
            template="plotly_white",
            xaxis_title="Quarter",
            yaxis_title="MFG_GVA_GROWTH_RATE_%",
            hovermode="x unified",
            showlegend=True
        )
        fig2.update_xaxes(type="date")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click 'Run Monte Carlo' in the sidebar to generate the 3-year simulation.")
        
#  Tab 3 - Scenarios 
#  Tab 3 - Fan chart from Monte Carlo ─
with tab3:
    st.subheader("10-year fan chart")

    h10 = 40
    future_dates_10y = pd.date_range(
        clean.index[-1] + pd.offsets.QuarterEnd(1), periods=h10, freq="Q"
    )

    exog_cols = list(exog.columns)
    exog_last = exog.iloc[-1].astype(float).to_numpy()
    exog_diff = exog.diff().dropna()
    exog_mu = exog_diff.mean().to_numpy()
    exog_sigma = exog_diff.std().to_numpy()
    exog_sigma = np.where(np.isfinite(exog_sigma) & (exog_sigma > 0), exog_sigma, 1.0)

    resid = fitted.resid.dropna().to_numpy()
    resid = resid[np.isfinite(resid)]

    # Monte Carlo settings for long-horizon fan chart
    mc_fan_sims = max(1000, min(mc_sims, 5000))
    st.caption(f"Fan chart based on {mc_fan_sims} Monte Carlo paths.")

    fan_mc = np.zeros((mc_fan_sims, h10))
    progress = st.progress(0)

    for i in range(mc_fan_sims):
        x_path = np.zeros((h10, exog.shape[1]))
        prev = exog_last.copy()

        for t in range(h10):
            shock = np.random.normal(exog_mu, exog_sigma * 0.5)
            prev = prev + shock
            x_path[t] = prev

        x_df = pd.DataFrame(x_path, index=future_dates_10y, columns=exog_cols).astype(float)
        pred = fitted.get_forecast(steps=h10, exog=x_df).predicted_mean.to_numpy()

        noise = np.random.choice(resid, size=h10, replace=True) if len(resid) > 0 else np.zeros(h10)
        fan_mc[i] = pred + noise

        if i % max(1, mc_fan_sims // 100) == 0:
            progress.progress(i / mc_fan_sims)

    progress.progress(1.0)

    p05 = np.percentile(fan_mc, 5, axis=0)
    p10 = np.percentile(fan_mc, 10, axis=0)
    p25 = np.percentile(fan_mc, 25, axis=0)
    p50 = np.percentile(fan_mc, 50, axis=0)
    p75 = np.percentile(fan_mc, 75, axis=0)
    p90 = np.percentile(fan_mc, 90, axis=0)
    p95 = np.percentile(fan_mc, 95, axis=0)

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=y.index,
            y=y,
            name="History",
            mode="lines",
            line=dict(color="black", width=2),
        )
    )

    # 90% band
    fig3.add_trace(
        go.Scatter(
            x=list(future_dates_10y) + list(future_dates_10y[::-1]),
            y=list(p10) + list(p90[::-1]),
            fill="toself",
            fillcolor="rgba(30,144,255,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10-90%",
            hoverinfo="skip",
        )
    )

    # 50% band
    fig3.add_trace(
        go.Scatter(
            x=list(future_dates_10y) + list(future_dates_10y[::-1]),
            y=list(p25) + list(p75[::-1]),
            fill="toself",
            fillcolor="rgba(30,144,255,0.22)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25-75%",
            hoverinfo="skip",
        )
    )

    # Median line
    fig3.add_trace(
        go.Scatter(
            x=future_dates_10y,
            y=p50,
            name="Median forecast",
            mode="lines",
            line=dict(color="royalblue", width=4),
        )
    )

    # Optional outer uncertainty line
    fig3.add_trace(
        go.Scatter(
            x=future_dates_10y,
            y=p05,
            name="5%",
            mode="lines",
            line=dict(color="rgba(220,20,60,0.6)", width=1, dash="dot"),
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=future_dates_10y,
            y=p95,
            name="95%",
            mode="lines",
            line=dict(color="rgba(220,20,60,0.6)", width=1, dash="dot"),
        )
    )

    fig3.update_layout(
        template="plotly_white",
        xaxis_title="Quarter",
        yaxis_title="MFG_GVA Growth Rate % Forecast",
        hovermode="x unified",
        legend_title="Forecast bands",
    )
    fig3.update_xaxes(type="date")
    fig3.update_yaxes(autorange=True)

    st.plotly_chart(fig3, use_container_width=True)

    fan_df = pd.DataFrame(
        {
            "quarter": future_dates_10y,
            "p05": p05,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
        }
    )

    with st.expander("Fan chart percentiles", expanded=False):
        st.dataframe(fan_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download fan chart percentiles CSV",
        fan_df.to_csv(index=False),
        file_name="fan_chart_percentiles.csv",
        mime="text/csv",
        use_container_width=True,
    )
#  Tab 4 - Diagnostics 
with tab4:
    st.subheader("Diagnostics")
    col3, col4 = st.columns(2)

    resid_clean = fitted.resid.dropna()
    max_lags = min(20, len(resid_clean) - 2)   # guard against short series

    with col3:
        fig_acf, ax_acf = plt.subplots(figsize=(6, 3))
        plot_acf(resid_clean, ax=ax_acf, lags=max_lags)
        ax_acf.set_title("ACF of residuals")
        st.pyplot(fig_acf, clear_figure=True)

    with col4:
        fig_pacf, ax_pacf = plt.subplots(figsize=(6, 3))
        plot_pacf(resid_clean, ax=ax_pacf, lags=max_lags, method="ywm")
        ax_pacf.set_title("PACF of residuals")
        st.pyplot(fig_pacf, clear_figure=True)

    # Residual distribution
    st.markdown("#### Residual distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
    ax_hist.hist(resid_clean, bins=30, edgecolor="white", color="steelblue", alpha=0.85)
    ax_hist.axvline(0, color="crimson", linewidth=1.5, linestyle="--")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of model residuals")
    st.pyplot(fig_hist, clear_figure=True)

    if show_summary:
        with st.expander("Model summary", expanded=False):
            st.text(str(fitted.summary()))

#  Tab 5 - Results & Interpretation
with tab5:
    st.subheader("Results & Interpretation")

    st.markdown("### Model summary")
    st.text(str(fitted.summary()))

    st.markdown("### 1-year forecast results")
    forecast_1y = pd.DataFrame({
        "quarter": mean1.index,
        "forecast": mean1.values,
        "lower_95": ci1.iloc[:, 0].values,
        "upper_95": ci1.iloc[:, 1].values,
    })
    st.dataframe(forecast_1y, use_container_width=True, hide_index=True)

    st.markdown("### 3-year Monte Carlo results")
    if run_mc:
        mc_results = pd.DataFrame({
            "quarter": future_dates_3y,
            "mc_mean": mc_mean,
            "mc_lower_95": mc_lo,
            "mc_upper_95": mc_hi,
        })
        st.dataframe(mc_results, use_container_width=True, hide_index=True)
    else:
        st.info("Run Monte Carlo from the sidebar to populate the 3-year results table.")

    st.markdown("### 10-year fan chart results")
    st.dataframe(fan_df, use_container_width=True, hide_index=True)

st.caption("Built for quarterly manufacturing forecasting with ARIMAX, Monte Carlo simulation, and scenario analysis.")
