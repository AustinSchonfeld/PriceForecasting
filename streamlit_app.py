"""
Day-Ahead Electricity Price Forecasting + Battery Trading Backtest (Proof of Concept)

Run with:
    pip install streamlit pandas numpy scikit-learn
    streamlit run app.py

What this demo shows:
1. A forecast of hourly day-ahead prices using gradient boosting.
2. A fair comparison against "naive" baselines (same hour yesterday / last week).
3. Prediction intervals (P10-P90) from quantile models.
4. A simple battery-trading backtest: what is the forecast worth in money?
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------


def _smooth_noise(rng, n, span):
    """Slowly-varying standardised noise (mimics weather persistence)."""
    z = pd.Series(rng.normal(0, 1, n)).ewm(span=span).mean().values
    return (z - z.mean()) / z.std()


@st.cache_data
def make_synthetic_data(days: int = 900, seed: int = 42) -> pd.DataFrame:
    """
    Build a fake-but-plausible hourly power market so the demo always works.
    Prices respond to gas price and 'residual load' (load minus wind and solar),
    which is the core driver of real power prices (merit order effect).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values

    # Weather
    temp = (
        12
        + 10 * np.cos(2 * np.pi * (doy - 200) / 365)
        + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
        + 2.5 * _smooth_noise(rng, n, 48)
    )
    wind = np.clip(12 + 8 * _smooth_noise(rng, n, 36), 0, 30)
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    cloud = np.clip(0.6 + 0.25 * _smooth_noise(rng, n, 24), 0.1, 1.0)
    solar = 15 * daylight * cloud * (0.6 + 0.4 * np.cos(2 * np.pi * (doy - 172) / 365))

    # Demand (GW): morning and evening peaks, lower on weekends,
    # higher when it is very hot or very cold
    load = (
        40
        + 5 * np.exp(-((hour - 8) ** 2) / 8)
        + 7 * np.exp(-((hour - 19) ** 2) / 8)
        - 4 * (hour < 5)
        - 4 * (dow >= 5)
        + 0.35 * np.abs(temp - 18)
        + rng.normal(0, 0.8, n)
    )

    # Gas price (EUR/MWh): slow random walk
    gas = np.clip(30 + np.cumsum(rng.normal(0, 0.05, n)), 15, 80)

    # Price formation
    residual = load - wind - solar
    price = (
        0.8 * gas
        + 1.2 * (residual - 30)
        + 1.5 * np.clip(residual - 42, 0, None) ** 2 / 4  # scarcity: steep curve
        + rng.normal(0, 4, n)
    )
    spikes = rng.random(n) < 0.003
    price = price + spikes * rng.exponential(80, n)

    # What a trader would actually have at gate closure: FORECASTS, not actuals
    return pd.DataFrame(
        {
            "price": price,
            "load_fc": load + rng.normal(0, 1.2, n),
            "wind_fc": np.clip(wind + rng.normal(0, 2.5, n), 0, None),
            "solar_fc": np.clip(solar + rng.normal(0, 1.0, n), 0, None),
            "temp_fc": temp + rng.normal(0, 1.0, n),
            "gas_price": gas,
        },
        index=idx,
    )


def load_csv(file) -> pd.DataFrame:
    """Read a user CSV. Needs 'timestamp' and 'price'; other numeric columns become features."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("CSV needs at least two columns: 'timestamp' and 'price'.")
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.drop(columns="timestamp").set_index(ts).sort_index()
    df = df.select_dtypes("number")
    df = df.resample("h").mean().interpolate(limit=3)
    return df.dropna(subset=["price"])


# ----------------------------------------------------------------------------
# 2. FEATURES
# ----------------------------------------------------------------------------


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Day-ahead auctions clear around noon on day D-1 for every hour of day D.
    At that moment all of day D-1's prices are known, so a 24h lag is legitimate.
    """
    out = df.copy()
    for lag in (24, 48, 168):
        out[f"price_lag_{lag}"] = out["price"].shift(lag)
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    if {"load_fc", "wind_fc", "solar_fc"} <= set(out.columns):
        out["residual_load_fc"] = out["load_fc"] - out["wind_fc"] - out["solar_fc"]
    return out.dropna()


# ----------------------------------------------------------------------------
# 3. MODEL: walk-forward backtest
# ----------------------------------------------------------------------------


def _model(**kwargs):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0, **kwargs)


@st.cache_data(show_spinner="Training models (walk-forward)...")
def run_walk_forward(feat: pd.DataFrame, test_days: int, retrain_every: int):
    """
    Mimic real use: every `retrain_every` days, retrain on all data seen so far,
    then forecast the next block. The model never sees the future.
    """
    cols = [c for c in feat.columns if c != "price"]
    test_start = feat.index[-1].normalize() - pd.Timedelta(days=test_days - 1)
    test = feat[feat.index >= test_start]
    days = sorted(test.index.normalize().unique())

    pieces, last_model, last_block = [], None, None
    for i in range(0, len(days), retrain_every):
        block_days = days[i : i + retrain_every]
        train = feat[feat.index < block_days[0]]
        block = test[test.index.normalize().isin(block_days)]

        mean_model = _model().fit(train[cols], train["price"])
        lo_model = _model(loss="quantile", quantile=0.1).fit(train[cols], train["price"])
        hi_model = _model(loss="quantile", quantile=0.9).fit(train[cols], train["price"])

        pieces.append(
            pd.DataFrame(
                {
                    "actual": block["price"],
                    "model": mean_model.predict(block[cols]),
                    "p10": lo_model.predict(block[cols]),
                    "p90": hi_model.predict(block[cols]),
                    "naive_24": block["price_lag_24"],
                    "naive_168": block["price_lag_168"],
                },
                index=block.index,
            )
        )
        last_model, last_block = mean_model, block

    preds = pd.concat(pieces)

    # Which inputs matter most? (permutation importance on the last two weeks)
    recent = last_block.tail(24 * 14)
    imp = permutation_importance(
        last_model, recent[cols], recent["price"],
        n_repeats=3, random_state=0, scoring="neg_mean_absolute_error",
    )
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    return preds, importance


# ----------------------------------------------------------------------------
# 4. TRADING BACKTEST
# ----------------------------------------------------------------------------


def day_profit(actual: pd.Series, signal: pd.Series, k: int, efficiency: float) -> float:
    """
    One battery, 1 MW, k hours of storage, one cycle per day.
    Buy in the k cheapest hours (according to the signal), sell in the k most
    expensive. Only trade if the forecast spread covers round-trip losses.
    Profit is then settled at ACTUAL prices.
    """
    buy_hours = signal.nsmallest(k).index
    sell_hours = signal.nlargest(k).index
    if efficiency * signal[sell_hours].mean() <= signal[buy_hours].mean():
        return 0.0
    return efficiency * actual[sell_hours].sum() - actual[buy_hours].sum()


def backtest(preds: pd.DataFrame, k: int, efficiency: float) -> pd.DataFrame:
    rows = {}
    for day, g in preds.groupby(preds.index.normalize()):
        if len(g) < 24:
            continue
        rows[day] = {
            "Perfect foresight": day_profit(g["actual"], g["actual"], k, efficiency),
            "Our model": day_profit(g["actual"], g["model"], k, efficiency),
            "Naive (last week)": day_profit(g["actual"], g["naive_168"], k, efficiency),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# 5. APP
# ----------------------------------------------------------------------------


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    st.set_page_config(page_title="Power Price Forecaster", layout="wide")
    st.title("Day-Ahead Power Price Forecasting and Battery Trading")
    st.caption(
        "Proof of concept: forecast tomorrow's hourly prices, test it honestly "
        "against naive baselines, then ask what the forecast is worth to a trader."
    )

    # ---- Sidebar controls
    st.sidebar.header("Settings")
    source = st.sidebar.radio("Data source", ["Synthetic demo data", "Upload CSV"])
    test_days = st.sidebar.slider("Backtest length (days)", 30, 180, 90, step=10)
    retrain_every = st.sidebar.slider("Retrain every (days)", 1, 30, 7)
    st.sidebar.header("Battery assumptions")
    k = st.sidebar.slider("Storage duration (hours at full power)", 1, 6, 2)
    efficiency = st.sidebar.slider("Round-trip efficiency", 0.70, 1.00, 0.88, step=0.01)

    if source == "Synthetic demo data":
        raw = make_synthetic_data()
        st.sidebar.info("Synthetic data lets the demo run anywhere. Swap in real data via CSV.")
    else:
        up = st.sidebar.file_uploader("CSV with 'timestamp' and 'price' columns", type="csv")
        if up is None:
            st.info(
                "Upload a CSV with a `timestamp` column, a `price` column and (optionally) "
                "numeric forecast columns such as `load_fc`, `wind_fc`, `solar_fc`, `temp_fc`. "
                "Use at least ~1 year of hourly data."
            )
            st.stop()
        try:
            raw = load_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

    feat = add_features(raw)
    if len(feat) < (test_days + 90) * 24:
        st.error("Not enough history for this backtest length. Shorten the backtest or add data.")
        st.stop()

    preds, importance = run_walk_forward(feat, test_days, retrain_every)

    tab1, tab2, tab3 = st.tabs(["Forecast", "Trading value", "Concepts and limitations"])

    # ---- Tab 1: forecast accuracy
    with tab1:
        a = preds["actual"]
        m_mae, n24_mae, n168_mae = mae(a, preds["model"]), mae(a, preds["naive_24"]), mae(a, preds["naive_168"])
        coverage = float(((a >= preds["p10"]) & (a <= preds["p90"])).mean())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model MAE", f"{m_mae:.2f}")
        c2.metric("Naive (yesterday) MAE", f"{n24_mae:.2f}")
        c3.metric("Naive (last week) MAE", f"{n168_mae:.2f}")
        c4.metric("Improvement vs best naive", f"{(1 - m_mae / min(n24_mae, n168_mae)) * 100:.0f}%")
        st.caption(
            f"RMSE: model {rmse(a, preds['model']):.2f}, naive-168 {rmse(a, preds['naive_168']):.2f}. "
            f"The P10-P90 band contained {coverage:.0%} of actual prices (ideal is about 80%). "
            "Units are the price units of your data (EUR/MWh for the demo)."
        )

        day = st.selectbox("Pick a day to inspect", sorted(preds.index.normalize().unique())[::-1],
                           format_func=lambda d: d.strftime("%a %d %b %Y"))
        d = preds[preds.index.normalize() == day]
        st.line_chart(d[["actual", "model", "naive_168", "p10", "p90"]])

        left, right = st.columns(2)
        with left:
            st.subheader("Error by hour of day")
            by_hour = pd.DataFrame(
                {
                    "Model": (preds["model"] - a).abs().groupby(preds.index.hour).mean(),
                    "Naive (last week)": (preds["naive_168"] - a).abs().groupby(preds.index.hour).mean(),
                }
            )
            st.bar_chart(by_hour)
        with right:
            st.subheader("What drives the forecast?")
            st.bar_chart(importance.head(10))
            st.caption("Permutation importance: how much the error grows if a feature is shuffled.")

    # ---- Tab 2: trading value
    with tab2:
        st.subheader("What is the forecast worth? Simple battery arbitrage")
        pnl = backtest(preds, k, efficiency)
        total = pnl.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Perfect foresight", f"{total['Perfect foresight']:,.0f}")
        c2.metric("Our model", f"{total['Our model']:,.0f}",
                  f"{total['Our model'] / total['Perfect foresight']:.0%} of perfect")
        c3.metric("Naive (last week)", f"{total['Naive (last week)']:,.0f}",
                  f"{total['Naive (last week)'] / total['Perfect foresight']:.0%} of perfect")
        st.line_chart(pnl.cumsum())
        st.caption(
            "Profit per 1 MW of battery power over the backtest, in price-currency units. "
            "The battery buys in the hours the forecast says are cheapest, sells in the hours it says "
            "are most expensive, and is settled at actual prices. 'Perfect foresight' is the upper bound; "
            "the gap between it and our model is the value still left on the table."
        )
        st.warning(
            "Simplifications: one cycle per day, no ordering constraint between charge and discharge "
            "hours, no degradation, fees, intraday re-trading or price impact. A real dispatch model "
            "would solve a proper optimisation (for example linear programming)."
        )

    # ---- Tab 3: concepts
    with tab3:
        st.markdown(
            """
### Market concepts used here
- **Day-ahead market:** an auction run around midday on day D-1 that sets one price per hour for day D.
  We forecast these prices, so only information available before gate closure may be used.
- **Merit order:** generators are dispatched cheapest-first, and the last one needed sets the price.
  Gas is often the marginal plant, so gas price is a key driver.
- **Residual load:** demand minus wind and solar. When it is high, expensive plants run and prices spike.
  When it is low, prices can fall to zero or go negative.
- **Spread:** the gap between cheap and expensive hours. Storage and flexible assets earn by capturing it.
- **Forecast vs actual inputs:** we feed in *forecasts* of load, wind and solar, because actuals are
  unknown when you must trade. Using actuals would leak the future and flatter the results.

### Methodology
- **Walk-forward validation:** retrain periodically on past data only, then predict the next block.
- **Baselines:** "same hour yesterday" and "same hour last week" are hard to beat. A model must beat them to add value.
- **Quantile models** give the P10 and P90 band, which matters for risk. A trader needs the range, not only the mean.
- **MAE and RMSE** rather than MAPE, because prices near zero or negative break percentage errors.

### Limitations
- Synthetic data proves the pipeline, not real-world performance. Re-run on a real market before drawing conclusions.
- Price spikes are the hardest part and dominate RMSE.
- Market regimes shift (gas shocks, new renewables, rule changes), so models need monitoring and retraining.
- Real trading also needs intraday and balancing markets, transaction costs and risk limits.
"""
        )


if __name__ == "__main__":
    main()
