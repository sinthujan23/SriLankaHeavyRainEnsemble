"""
Sri Lanka Heavy-Rain Early-Warning: Ensemble Learning Dashboard
Run:  streamlit run app.py
(Requires the artefacts produced by Ensemble_Learning_SL_Heavy_Rain.ipynb in ./models and ./data)

Generative-AI disclosure: this dashboard was generated with the help of Claude (Anthropic) via Claude Code.
"""
import inspect
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR, DATA_DIR = ROOT / "models", ROOT / "data"
FAM_COLORS = {"Baseline": "#9e9e9e", "Bagging": "#2ca02c", "Boosting": "#1f77b4", "Stacking": "#9467bd", "Hybrid": "#d62728"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_CITY, DEFAULT_DATE = "Kandy", date(2025, 11, 26)

st.set_page_config(page_title="Sri Lanka Heavy-Rain Ensemble Dashboard", page_icon="🌧️", layout="wide")


# ------------------------------------------------------------------ loading (cached)
@st.cache_data
def load_artifacts():
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    roc = json.loads((MODEL_DIR / "roc_curves.json").read_text())
    cms = json.loads((MODEL_DIR / "confusion_matrices.json").read_text())
    fi = json.loads((MODEL_DIR / "feature_importance.json").read_text())
    interp_path = MODEL_DIR / "interpretation.md"
    interp = interp_path.read_text(encoding="utf-8") if interp_path.exists() else ""
    return meta, metrics, roc, cms, fi, interp


@st.cache_data
def load_data():
    d = pd.read_parquet(DATA_DIR / "dashboard_features.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


@st.cache_resource(show_spinner="Loading trained ensemble models …")
def load_models(model_list):
    return {m["name"]: joblib.load(MODEL_DIR / m["file"]) for m in model_list}


meta, metrics, roc, cms, fi, interpretation = load_artifacts()
data = load_data()
models = load_models(meta["models"])
FEATURES = meta["num_features"] + meta["cat_features"]
FAMILY = {m["name"]: m["family"] for m in meta["models"]}
BEST = meta["best_model"]
towns = (data.groupby("city")[["district", "province", "climate_zone", "latitude", "longitude", "elevation"]].first())


def plot(fig, **kw):
    # width="stretch" on new Streamlit versions, use_container_width on older ones
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", **kw)
    else:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def season_of(month):
    if month in (12, 1, 2):
        return "NE Monsoon"
    if month in (3, 4):
        return "First Inter-monsoon"
    if month in (5, 6, 7, 8, 9):
        return "SW Monsoon"
    return "Second Inter-monsoon"


# ------------------------------------------------------------------ header
st.title("🌧️ Sri Lanka Heavy-Rain Early Warning: Ensemble Learning Dashboard")
st.caption(f"Predicts whether a Sri Lankan town will receive **≥ {meta['heavy_rain_mm']:.0f} mm of rain tomorrow** "
           f"using Bagging, Boosting, Stacking and a Hybrid RF + ANN. Data: NASA POWER daily weather for 20 towns in all "
           f"9 provinces, {meta['train_period'][0]} → {meta['test_period'][1]}.")

# Navigation: a keyed radio keeps the selected page across reruns (st.tabs jumps back to the first tab
# whenever a slider on another tab changes).
PAGES = ["🏠 Overview", "🔮 Predict", "📊 Compare models", "🌲 Feature importance", "ℹ️ Method & AI disclosure"]
PAGE = st.radio("Page", PAGES, horizontal=True, key="page", label_visibility="collapsed")
st.divider()

# ================================================================== OVERVIEW
if PAGE == "🏠 Overview":
    best_row = metrics.set_index("model").loc[BEST]
    k = st.columns(5)
    k[0].metric("Towns / provinces", f"{data.city.nunique()} / {data.province.nunique()}")
    k[1].metric("Town-days of weather", f"{len(data):,}")
    k[2].metric("Heavy-rain days", f"{data.heavy_rain_tomorrow.mean():.1%}")
    k[3].metric("Models compared", f"{len(metrics)}")
    k[4].metric(f"Best model (F1): {BEST}", f"{best_row.f1:.3f}", f"ROC-AUC {best_row.roc_auc:.3f}", delta_color="off")

    c1, c2 = st.columns([1, 1.35])
    with c1:
        yearly = (data[data.date.dt.year.between(2016, 2025)].assign(y=data.date.dt.year)
                  .groupby(["city", "y"]).rain_mm.sum().groupby("city").mean().rename("annual_rain_mm"))
        town_stats = towns.join(yearly).join(data.groupby("city").heavy_rain_tomorrow.mean().rename("heavy_rain_rate")).reset_index()
        town_stats["heavy_rain_pct"] = (town_stats.heavy_rain_rate * 100).round(1)
        fig = px.scatter_map(town_stats, lat="latitude", lon="longitude", size="annual_rain_mm", color="heavy_rain_pct",
                             color_continuous_scale="Blues", range_color=[8, 25], size_max=26, zoom=6.2,
                             center={"lat": 7.85, "lon": 80.75}, map_style="carto-positron", hover_name="city",
                             hover_data={"province": True, "climate_zone": True, "annual_rain_mm": ":.0f",
                                         "heavy_rain_pct": ":.1f", "latitude": False, "longitude": False},
                             labels={"heavy_rain_pct": "Heavy-rain days %", "annual_rain_mm": "mm / year"},
                             height=560, title="20 towns · bubble = annual rainfall · colour = % heavy-rain days")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        plot(fig)
    with c2:
        heat = (data.assign(m=data.date.dt.month).pivot_table(index="city", columns="m", values="heavy_rain_tomorrow", aggfunc="mean"))
        heat = heat.loc[town_stats.sort_values("annual_rain_mm", ascending=False).city]
        fig = px.imshow(heat * 100, x=MONTHS, y=heat.index, color_continuous_scale="Blues", aspect="auto", text_auto=".0f",
                        labels=dict(color="P(heavy rain) %"), height=560,
                        title="Monsoon fingerprint: probability of a heavy-rain day by town × month (%)")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        plot(fig)

    zone_month = (data.assign(m=data.date.dt.month, y=data.date.dt.year).groupby(["climate_zone", "city", "y", "m"]).rain_mm.sum()
                  .groupby(["climate_zone", "m"]).mean().reset_index())
    zone_month["month"] = zone_month.m.map(lambda i: MONTHS[i - 1])
    fig = px.line(zone_month, x="month", y="rain_mm", color="climate_zone", markers=True, height=330,
                  color_discrete_map={"Wet": "#1f77b4", "Intermediate": "#2ca02c", "Dry": "#ff7f0e"},
                  labels={"rain_mm": "mm / month", "month": "", "climate_zone": "Zone"},
                  title="Average monthly rainfall by climatic zone: May peak (SW-monsoon onset) in the Wet zone; all zones peak Oct–Dec")
    fig.add_vrect(x0=3.5, x1=8.5, fillcolor="grey", opacity=0.08, line_width=0, annotation_text="SW monsoon")
    plot(fig)

# ================================================================== PREDICT
if PAGE == "🔮 Predict":
    st.subheader("🔮 Will it rain heavily tomorrow?")
    st.write("Pick a town and a day. The inputs load the **real NASA POWER observations** for that day, and you can "
             "then change any value to run a *what-if* scenario. Predictions come from the trained, serialised (joblib) models.")
    left, right = st.columns([1.05, 1])
    with left:
        a, b = st.columns(2)
        city = a.selectbox("Town", sorted(towns.index), index=sorted(towns.index).index(DEFAULT_CITY))
        dmin, dmax = data.date.min().date(), data.date.max().date()
        day = b.date_input("Today's date", value=DEFAULT_DATE, min_value=dmin, max_value=dmax)
        rec = data[(data.city == city) & (data.date.dt.date == day)]
        real_day = not rec.empty
        if not real_day:
            st.warning("No record for that day; using the town's typical values for that month.")
            rec = data[(data.city == city) & (data.date.dt.month == day.month)].median(numeric_only=True).to_frame().T
        r = rec.iloc[0]
        t = towns.loc[city]
        st.caption(f"📍 {city}: {t.district} district, {t.province} Province · {t.climate_zone} zone · "
                   f"season: **{season_of(day.month)}**")
        K = f"{city}_{day}"  # widget keys change with town/date, so inputs reload the real values

        def num(label, col, lo, hi, step, val=None, fmt=None, container=st, help=None):
            v = float(r[col] if val is None else val)
            v = float(np.clip(v, lo, hi))
            return container.slider(label, lo, hi, round(v, 2), step, key=f"{col}_{K}", format=fmt, help=help)

        with st.expander("🌧️ Rainfall history", expanded=True):
            c = st.columns(2)
            rain_today = num("Rain today (mm)", "rain_mm", 0.0, 200.0, 0.5, container=c[0])
            rain_y = num("Rain yesterday (mm)", "rain_lag1", 0.0, 200.0, 0.5, container=c[1])
            rain_2 = num("Rain 2 days ago (mm)", "rain_3d_sum", 0.0, 200.0, 0.5,
                         val=max(r.rain_3d_sum - r.rain_mm - r.rain_lag1, 0), container=c[0])
            rain_e = num("Rain 3–6 days ago, total (mm)", "rain_7d_sum", 0.0, 400.0, 0.5,
                         val=max(r.rain_7d_sum - r.rain_3d_sum, 0), container=c[1])
            rain_days = num("Rainy days (≥1 mm) in last 7", "rain_days_7d", 0.0, 7.0, 1.0, container=c[0])
        with st.expander("💧 Atmosphere & humidity", expanded=True):
            c = st.columns(2)
            hum = num("Relative humidity (%)", "humidity_pct", 30.0, 100.0, 0.5, container=c[0])
            hum_ch = num("Humidity change since yesterday (%)", "humidity_change_1d", -30.0, 30.0, 0.5, container=c[1])
            cloud = num("Cloud cover (%)", "cloud_cover_pct", 0.0, 100.0, 1.0, container=c[0])
            pres = num("Surface pressure (hPa)", "pressure_hpa", 800.0, 1020.0, 0.1, container=c[1])
            pres_ch = num("Pressure change since yesterday (hPa)", "pressure_change_1d", -10.0, 10.0, 0.1, container=c[0])
            dew = num("Dew point (°C)", "dew_point_c", 5.0, 30.0, 0.1, container=c[1])
            qv = num("Specific humidity (g/kg)", "specific_humidity_gkg", 5.0, 25.0, 0.1, container=c[0])
        with st.expander("🌡️ Temperature, sun, wind & soil"):
            c = st.columns(2)
            tmax = num("Max temperature (°C)", "temp_max_c", 10.0, 42.0, 0.1, container=c[0])
            tmin = num("Min temperature (°C)", "temp_min_c", 5.0, 32.0, 0.1, container=c[1])
            tmean = num("Mean temperature (°C)", "temp_mean_c", 8.0, 36.0, 0.1, container=c[0])
            solar = num("Solar radiation (MJ/m²/day)", "solar_rad_mj", 0.0, 32.0, 0.1, container=c[1])
            wind = num("Mean wind speed (m/s)", "wind_speed_ms", 0.0, 15.0, 0.1, container=c[0])
            wmax = num("Max wind speed (m/s)", "wind_max_ms", 0.0, 20.0, 0.1, container=c[1])
            wdir = num("Wind direction (° from)", "wind_dir_deg", 0.0, 360.0, 1.0, container=c[0])
            soil_t = num("Topsoil wetness (0–1)", "soil_wetness_top", 0.0, 1.0, 0.01, container=c[1])
            soil_r = num("Root-zone soil wetness (0–1)", "soil_wetness_root", 0.0, 1.0, 0.01, container=c[0])

    doy = day.timetuple().tm_yday
    x = pd.DataFrame([{
        "latitude": t.latitude, "longitude": t.longitude, "elevation": t.elevation,
        "temp_max_c": tmax, "temp_min_c": tmin, "temp_mean_c": tmean, "temp_range": tmax - tmin,
        "rain_mm": rain_today, "rain_lag1": rain_y, "rain_3d_sum": rain_today + rain_y + rain_2,
        "rain_7d_sum": rain_today + rain_y + rain_2 + rain_e, "rain_days_7d": rain_days,
        "humidity_pct": hum, "humidity_change_1d": hum_ch, "dew_point_c": dew, "dewpoint_depression": tmean - dew,
        "specific_humidity_gkg": qv, "pressure_hpa": pres, "pressure_change_1d": pres_ch, "cloud_cover_pct": cloud,
        "wind_speed_ms": wind, "wind_max_ms": wmax, "wind_dir_sin": np.sin(np.deg2rad(wdir)), "wind_dir_cos": np.cos(np.deg2rad(wdir)),
        "solar_rad_mj": solar, "soil_wetness_top": soil_t, "soil_wetness_root": soil_r,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "city": city, "province": t.province, "climate_zone": t.climate_zone, "monsoon_season": season_of(day.month),
    }])[FEATURES]

    probs = {name: float(m.predict_proba(x)[0, 1]) for name, m in models.items()}

    with right:
        model_names = list(metrics.model)
        chosen = st.selectbox("Ensemble model", model_names, index=model_names.index(BEST),
                              format_func=lambda n: f"{n}  ·  {FAMILY[n]}")
        p = probs[chosen]
        heavy = p >= 0.5
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 48}},
            title={"text": f"P(heavy rain ≥ {meta['heavy_rain_mm']:.0f} mm tomorrow)<br><sub>{chosen}</sub>"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#08519c" if heavy else "#6baed6"},
                   "steps": [{"range": [0, 50], "color": "#eef4fa"}, {"range": [50, 100], "color": "#c6dbef"}],
                   "threshold": {"line": {"color": "#d62728", "width": 4}, "value": 50}}))
        g.update_layout(height=290, margin=dict(l=30, r=30, t=70, b=10))
        plot(g)
        if heavy:
            st.error(f"🌧️ **Heavy rain likely in {city} tomorrow** ({p:.0%}). Consider flood and landslide precautions.")
        else:
            st.success(f"🌤️ **No heavy rain expected in {city} tomorrow** ({p:.0%}).")
        if real_day and pd.notna(r.get("rain_tomorrow_mm", np.nan)):
            actual = r.rain_tomorrow_mm
            st.info(f"📖 Ground truth for the loaded day: **{actual:.1f} mm** fell the next day "
                    f"({'heavy rain ✔' if actual >= meta['heavy_rain_mm'] else 'no heavy rain'}). "
                    "This applies only while the inputs are unchanged.")
        votes = sum(v >= 0.5 for v in probs.values())
        st.write(f"**Ensemble agreement:** {votes} of {len(probs)} models predict heavy rain.")
        pdf = pd.DataFrame({"model": list(probs), "prob": list(probs.values())})
        pdf["family"] = pdf.model.map(FAMILY)
        pdf = pdf.sort_values("prob")
        fig = px.bar(pdf, x="prob", y="model", color="family", orientation="h", color_discrete_map=FAM_COLORS,
                     range_x=[0, 1], height=400, labels={"prob": "probability of heavy rain", "model": ""},
                     title="Every model's prediction for this input")
        fig.update_yaxes(categoryorder="array", categoryarray=list(pdf.model))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#d62728")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="")
        plot(fig)

# ================================================================== COMPARE
if PAGE == "📊 Compare models":
    st.subheader("📊 Bagging vs Boosting vs Stacking vs Hybrid")
    st.caption(f"Chronological hold-out test set: **{meta['test_period'][0]} → {meta['test_period'][1]}** "
               f"({meta['n_test']:,} town-days, {meta['pos_rate_test']:.1%} heavy-rain). "
               f"Trained on {meta['train_period'][0]} → {meta['train_period'][1]} ({meta['n_train']:,} town-days).")
    show = metrics.copy()
    show.insert(0, "rank", range(1, len(show) + 1))
    table(show.style.format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
                                   | {"train_time_s": "{:.1f}"})
                 .background_gradient(subset=["f1", "roc_auc", "pr_auc", "recall"], cmap="Greens"))

    c1, c2 = st.columns([1.4, 1])
    with c1:
        metric = st.radio("Metric", ["f1", "roc_auc", "pr_auc", "recall", "precision", "accuracy"], horizontal=True,
                          format_func=lambda m: m.upper().replace("_", "-"))
        s = metrics.sort_values(metric)
        fig = px.bar(s, x=metric, y="model", color="family", orientation="h", color_discrete_map=FAM_COLORS, text_auto=".3f",
                     height=470, labels={"model": "", metric: metric.upper()})
        fig.update_xaxes(range=[max(0, s[metric].min() - 0.05), min(1, s[metric].max() + 0.03)])
        fig.update_yaxes(categoryorder="array", categoryarray=list(s.model))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title_text="",
                          legend=dict(orientation="h", y=-0.12, x=0))
        plot(fig, config={"displayModeBar": False})
    with c2:
        fam = metrics.groupby("family")[["f1", "roc_auc", "recall", "precision"]].mean().reset_index().melt("family")
        fig = px.bar(fam, x="variable", y="value", color="family", barmode="group", color_discrete_map=FAM_COLORS,
                     height=250, title="Average score per ensemble family", labels={"variable": "", "value": ""})
        fig.update_yaxes(range=[max(0, fam.value.min() - 0.1), 1])
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="")
        plot(fig)
        fig = px.scatter(metrics, x="train_time_s", y="f1", color="family", text="model", color_discrete_map=FAM_COLORS,
                         log_x=True, height=260, title="Accuracy vs cost: F1 against training time (log scale)",
                         labels={"train_time_s": "training time (s)", "f1": "F1"})
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        plot(fig)

    c3, c4 = st.columns(2)
    with c3:
        best_per_family = metrics.loc[metrics.groupby("family").f1.idxmax(), "model"].tolist()
        sel = st.multiselect("ROC curves", list(metrics.model), default=best_per_family)
        fig = go.Figure()
        for n in sel:
            auc = metrics.set_index("model").loc[n, "roc_auc"]
            fig.add_trace(go.Scatter(x=roc[n]["fpr"], y=roc[n]["tpr"], mode="lines", name=f"{n} ({auc:.3f})",
                                     line=dict(color=FAM_COLORS[FAMILY[n]], width=3 if n == BEST else 1.8)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
        fig.update_layout(height=420, xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
                          margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0.35, y=0.05, font=dict(size=10)))
        plot(fig)
    with c4:
        cm_model = st.selectbox("Confusion matrix for", list(metrics.model), index=0)
        cm = np.array(cms[cm_model])
        fig = px.imshow(cm, text_auto=",", color_continuous_scale="Blues", height=360,
                        x=["Predicted: no heavy rain", "Predicted: heavy rain"], y=["Actual: no heavy rain", "Actual: heavy rain"])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        plot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✅ caught **{tp:,}** of {tp + fn:,} heavy-rain days (recall {tp / (tp + fn):.1%}) · "
                   f"❌ missed {fn:,} · ⚠️ {fp:,} false alarms (precision {tp / (tp + fp):.1%})")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"**{BEST}: performance by climatic zone**")
        table(pd.DataFrame(meta["by_zone"]).rename(columns={"climate_zone": "zone"}))
    with c6:
        st.markdown(f"**{BEST}: performance by monsoon season**")
        table(pd.DataFrame(meta["by_season"]).rename(columns={"monsoon_season": "season"}))
    st.caption(f"Stacking meta-learner weights: {meta['stacking_meta_weights']} · Random Forest out-of-bag accuracy: {meta['rf_oob_score']:.3f}")
    if interpretation:
        with st.expander("🧠 Why did the best model win? (interpretation)", expanded=True):
            st.markdown(interpretation)

# ================================================================== FEATURE IMPORTANCE
if PAGE == "🌲 Feature importance":
    st.subheader("🌲 What drives tomorrow's heavy rain?")
    c1, c2 = st.columns([1, 1])
    with c1:
        fm = st.selectbox("Model", list(metrics.model), index=0, key="fi_model")
        topn = st.slider("Number of features", 5, len(FEATURES), 15)
        d = pd.DataFrame({"feature": fi[fm]["features"], "importance": fi[fm]["importance"]}).head(topn)[::-1]
        fig = px.bar(d, x="importance", y="feature", orientation="h", height=max(350, 26 * topn),
                     color_discrete_sequence=[FAM_COLORS[FAMILY[fm]]], labels={"feature": "", "importance": "relative importance"},
                     title=f"{fm}: {fi[fm]['method']}")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        plot(fig)
    with c2:
        top_feats = pd.Series(fi[BEST]["importance"], index=fi[BEST]["features"]).head(12).index
        comp = pd.DataFrame({n: pd.Series(fi[n]["importance"], index=fi[n]["features"]).reindex(top_feats)
                             for n in metrics.model}).T
        fig = px.imshow(comp, color_continuous_scale="Blues", aspect="auto", height=520,
                        title=f"Do all models agree? Importance of {BEST}'s top-12 features across every model",
                        labels=dict(color="importance"))
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        plot(fig)
    st.info("**Reading the chart:** every model relies most on **rainfall memory** (rain today, yesterday, and the 3- and 7-day totals), "
            "then **season** (day-of-year and monsoon), **cloud cover / solar radiation** and the **1-day pressure change**. "
            "This matches meteorology: once a monsoon wet spell or a low-pressure system arrives, heavy rain tends to persist for "
            "several days. Boosting models also use the town and monsoon season to learn regional corrections.")

# ================================================================== ABOUT
if PAGE == "ℹ️ Method & AI disclosure":
    st.subheader("ℹ️ Method")
    st.markdown(f"""
* **Data (local, Sri Lankan):** NASA POWER daily weather (MERRA-2 reanalysis with bias-corrected rainfall) for **20 towns across all 9
  provinces** and the Wet, Intermediate and Dry zones. Each town lies in a different ~50 km grid cell. Period: {meta['train_period'][0]} → {meta['test_period'][1]}.
* **Target:** heavy rain tomorrow (≥ {meta['heavy_rain_mm']:.0f} mm) in the same town. Class balance ≈ {meta['pos_rate_train']:.0%} positive.
* **Preprocessing:** −999 values set to missing; short gaps interpolated; longer gaps filled with the town × month climatology; pressure kPa → hPa;
  lag and rolling rainfall features, pressure/humidity change, dew-point depression, cyclical day-of-year, monsoon season;
  one-hot encoding of town/province/zone/season; standard scaling; **balanced sample weights** for class imbalance.
* **Split:** chronological. Train {meta['train_period'][0]} → {meta['train_period'][1]}; test {meta['test_period'][0]} → {meta['test_period'][1]}.
* **Models:** baselines (Logistic Regression, Decision Tree, ANN) · **Bagging** (Random Forest, Extra Trees) ·
  **Boosting** (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost) · **Stacking** (RF + LightGBM + ANN → Logistic Regression,
  5-fold out-of-fold) · **Hybrid** (Random Forest + ANN soft voting).
* **Limitations:** reanalysis rainfall is a ~50 km grid average, so it smooths extreme local downpours. Thresholds and inputs are daily totals,
  not radar or satellite nowcasts. Use as a decision-support demo, not an official warning.
""")
    st.subheader("🤖 Generative-AI disclosure")
    st.markdown("""
This dashboard and the accompanying notebook were developed with help from **Claude (Anthropic), used through Claude Code**.
It generated the data-collection, preprocessing, modelling and dashboard code and drafted the explanations and video script.
All code was run and the results checked by the student. See Section 13 of the notebook for the full statement.
""")
