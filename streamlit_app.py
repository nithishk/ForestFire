from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from live_pipeline import effis_map_url, fetch_hurtgenwald_history, fetch_hurtgenwald_weather, risk_label
from sensor_simulation import generate_sensor_demo


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "weather_analysis"


st.set_page_config(page_title="Weather Pattern Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    [data-testid="stMetric"] {
        background: #f6f8fa;
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #263746 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #10212f !important;
    }
    .insight-box {
        background: #eef7f3;
        border: 1px solid #c7e4d8;
        color: #17382c;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
    }
    .insight-box strong { color: #0d2f23; }
    .note-box {
        background: #fff8e8;
        border: 1px solid #f0d391;
        color: #4b3410;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
    }
    .note-box strong { color: #3a2609; }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 10px 0 18px 0;
    }
    .metric-card {
        background: #f6f8fa;
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .metric-label {
        color: #263746;
        font-size: 0.86rem;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #10212f;
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .svg-chart {
        background: #f8fafc;
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 8px;
        margin: 8px 0 16px 0;
    }
    .simple-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .simple-table th {
        background: #e8eef3;
        color: #152536;
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #cbd5df;
    }
    .simple-table td {
        padding: 7px 8px;
        border-bottom: 1px solid #e3e8ee;
    }
    .risk-banner {
        border-radius: 10px;
        padding: 18px 20px;
        margin: 12px 0 18px 0;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .risk-banner h3 {
        margin: 0 0 6px 0;
        font-size: 1.35rem;
    }
    .risk-banner p {
        margin: 0;
        font-size: 0.95rem;
    }
    .risk-critical { background: #7f1d1d; color: #fff7ed; }
    .risk-high { background: #9a3412; color: #fff7ed; }
    .risk-watch { background: #854d0e; color: #fffbeb; }
    .risk-low { background: #14532d; color: #f0fdf4; }
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
        margin: 10px 0 18px 0;
    }
    .signal-card {
        background: #f8fafc;
        color: #132333;
        border: 1px solid #dbe3ea;
        border-radius: 8px;
        padding: 12px;
    }
    .signal-card strong {
        display: block;
        margin-bottom: 5px;
    }
    h1, h2, h3 { letter-spacing: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    paris = pd.read_csv(DATA_DIR / "paris_hourly_area_mean.csv", parse_dates=["time_utc"])
    paris_daily = pd.read_csv(DATA_DIR / "paris_daily_summary.csv", parse_dates=["date"])
    germany = pd.read_csv(DATA_DIR / "germany_hourly_area_mean.csv", parse_dates=["time_utc"])
    germany_daily = pd.read_csv(DATA_DIR / "germany_daily_summary.csv", parse_dates=["date"])
    germany_monthly = pd.read_csv(DATA_DIR / "germany_monthly_summary.csv")
    germany_grid = pd.read_csv(DATA_DIR / "germany_grid_cell_summary.csv")

    paris["hour"] = paris["time_utc"].dt.hour
    paris["date"] = paris["time_utc"].dt.date
    paris["wind_direction"] = paris["wind10_direction_deg"].apply(degrees_to_compass)
    paris = add_fire_weather_columns(paris)
    germany["hour"] = germany["time_utc"].dt.hour
    germany["month"] = germany["time_utc"].dt.to_period("M").astype(str)
    germany["wind_direction"] = germany["wind10_direction_deg_mean"].apply(degrees_to_compass)
    germany_grid["wind_direction"] = germany_grid["wind10_direction_deg_mean"].apply(degrees_to_compass)
    germany["season"] = germany["time_utc"].dt.month.map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Autumn",
            10: "Autumn",
            11: "Autumn",
        }
    )
    return {
        "paris": paris,
        "paris_daily": paris_daily,
        "germany": germany,
        "germany_daily": germany_daily,
        "germany_monthly": germany_monthly,
        "germany_grid": germany_grid,
    }


def date_filter(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    min_date = frame["time_utc"].dt.date.min()
    max_date = frame["time_utc"].dt.date.max()
    selected = st.date_input(
        "Date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=key,
    )
    if isinstance(selected, tuple) and len(selected) == 2:
        start, end = selected
    else:
        start, end = min_date, max_date
    return frame[(frame["time_utc"].dt.date >= start) & (frame["time_utc"].dt.date <= end)].copy()


def relative_humidity(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    actual_vapor = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    saturation_vapor = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return (100 * actual_vapor / saturation_vapor).clip(0, 100)


def degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((degrees + 22.5) // 45) % 8
    return directions[index]


def main_wind_direction(frame: pd.DataFrame, u_col: str, v_col: str) -> tuple[float, str]:
    mean_u = frame[u_col].mean()
    mean_v = frame[v_col].mean()
    degrees = (270 - np.degrees(np.arctan2(mean_v, mean_u))) % 360
    return degrees, degrees_to_compass(degrees)


def direction_counts(frame: pd.DataFrame, direction_col: str) -> pd.Series:
    order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return frame[direction_col].value_counts().reindex(order, fill_value=0)


def add_fire_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["relative_humidity_pct"] = relative_humidity(enriched["t2m_c"], enriched["d2m_c"])
    enriched["wind10_kmh"] = enriched["wind10_mps"] * 3.6
    enriched["fire_30_30_30"] = (
        (enriched["t2m_c"] >= 30)
        & (enriched["relative_humidity_pct"] <= 30)
        & (enriched["wind10_kmh"] >= 30)
    )
    enriched["fire_weather_score"] = (
        (enriched["t2m_c"] >= 30).astype(int)
        + (enriched["relative_humidity_pct"] <= 30).astype(int)
        + (enriched["wind10_kmh"] >= 30).astype(int)
    )
    return enriched


@st.cache_data(ttl=1800)
def load_hurtgenwald_weather() -> pd.DataFrame:
    return fetch_hurtgenwald_weather()


@st.cache_data(ttl=3600)
def load_hurtgenwald_history(start_date, end_date):
    return fetch_hurtgenwald_history(start_date, end_date)


@st.cache_data(ttl=300)
def load_sensor_demo(seed: int) -> pd.DataFrame:
    return generate_sensor_demo(seed=seed)


def metric_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        "<div class='metric-card'>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)


def _numeric_frame(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "Value")
    else:
        frame = data.copy()
    return frame.apply(pd.to_numeric, errors="coerce")


def svg_line_chart(data: pd.DataFrame | pd.Series, height: int = 300) -> None:
    frame = _numeric_frame(data).dropna(how="all")
    if frame.empty:
        st.info("No chart data available.")
        return
    width = 900
    pad = 42
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    ymin, ymax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"]
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        "<rect width='100%' height='100%' fill='#f8fafc'/>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#cbd5df'/>",
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#cbd5df'/>",
    ]
    n = max(len(frame) - 1, 1)
    for idx, col in enumerate(frame.columns):
        points = []
        for i, value in enumerate(frame[col].to_numpy(dtype=float)):
            if np.isfinite(value):
                x = pad + (i / n) * plot_w
                y = pad + (1 - ((value - ymin) / (ymax - ymin))) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
        if len(points) > 1:
            parts.append(
                f"<polyline fill='none' stroke='{colors[idx % len(colors)]}' stroke-width='2.5' points='{' '.join(points)}'/>"
            )
    legend_x = pad
    for idx, col in enumerate(frame.columns):
        y = 18 + idx * 18
        parts.append(f"<circle cx='{legend_x}' cy='{y}' r='5' fill='{colors[idx % len(colors)]}'/>")
        parts.append(f"<text x='{legend_x + 10}' y='{y + 4}' fill='#152536' font-size='13'>{escape(str(col))}</text>")
        legend_x += 170
    parts.append(f"<text x='{pad}' y='{height - 10}' fill='#475569' font-size='12'>min {ymin:.1f}</text>")
    parts.append(f"<text x='{width - pad - 80}' y='{height - 10}' fill='#475569' font-size='12'>max {ymax:.1f}</text>")
    parts.append("</svg>")
    st.markdown(f"<div class='svg-chart'>{''.join(parts)}</div>", unsafe_allow_html=True)


def svg_bar_chart(data: pd.Series | dict, height: int = 300) -> None:
    series = pd.Series(data).apply(pd.to_numeric, errors="coerce").fillna(0)
    if series.empty:
        st.info("No chart data available.")
        return
    width = 900
    pad = 42
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    ymax = max(float(series.max()), 1.0)
    bar_w = plot_w / len(series)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        "<rect width='100%' height='100%' fill='#f8fafc'/>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#cbd5df'/>",
    ]
    for i, (label, value) in enumerate(series.items()):
        h = (float(value) / ymax) * plot_h
        x = pad + i * bar_w + bar_w * 0.15
        y = height - pad - h
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w*0.7:.1f}' height='{h:.1f}' fill='#2563eb'/>")
        parts.append(f"<text x='{x + bar_w*0.35:.1f}' y='{height - 16}' fill='#334155' font-size='11' text-anchor='middle'>{escape(str(label))[:12]}</text>")
        parts.append(f"<text x='{x + bar_w*0.35:.1f}' y='{max(y - 6, 14):.1f}' fill='#152536' font-size='11' text-anchor='middle'>{float(value):.1f}</text>")
    parts.append("</svg>")
    st.markdown(f"<div class='svg-chart'>{''.join(parts)}</div>", unsafe_allow_html=True)


def svg_scatter_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    height: int = 420,
    **_: object,
) -> None:
    data = frame[[x, y, size]].dropna()
    if data.empty:
        st.info("No chart data available.")
        return
    width = 900
    pad = 44
    plot_w = width - pad * 2
    plot_h = height - pad * 2
    xmin, xmax = data[x].min(), data[x].max()
    ymin, ymax = data[y].min(), data[y].max()
    smin, smax = data[size].min(), data[size].max()
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        "<rect width='100%' height='100%' fill='#f8fafc'/>",
        f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#cbd5df'/>",
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#cbd5df'/>",
    ]
    for row in data.itertuples(index=False):
        xv, yv, sv = row
        px = pad + ((xv - xmin) / max(xmax - xmin, 1e-9)) * plot_w
        py = pad + (1 - ((yv - ymin) / max(ymax - ymin, 1e-9))) * plot_h
        radius = 3 + ((sv - smin) / max(smax - smin, 1e-9)) * 8
        parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='{radius:.1f}' fill='#2563eb' opacity='0.55'/>")
    parts.append(f"<text x='{pad}' y='{height - 10}' fill='#475569' font-size='12'>{escape(x)}</text>")
    parts.append(f"<text x='12' y='{pad}' fill='#475569' font-size='12'>{escape(y)}</text>")
    parts.append("</svg>")
    st.markdown(f"<div class='svg-chart'>{''.join(parts)}</div>", unsafe_allow_html=True)


def html_table(frame: pd.DataFrame, max_rows: int | None = None, **_: object) -> None:
    table = frame.head(max_rows) if max_rows else frame
    st.markdown(table.to_html(index=False, escape=True, classes="simple-table"), unsafe_allow_html=True)


def risk_class(prediction: str) -> str:
    return {
        "Critical": "risk-critical",
        "High": "risk-high",
        "Watch": "risk-watch",
        "Low": "risk-low",
    }.get(prediction, "risk-low")


def risk_color(prediction: str) -> str:
    return {
        "Critical": "#dc2626",
        "High": "#f97316",
        "Watch": "#facc15",
        "Low": "#22c55e",
    }.get(prediction, "#22c55e")


def risk_banner(sensor_id: str, zone: str, prediction: str, probability: float) -> None:
    message = (
        f"{sensor_id} at {zone} is showing the strongest fire signal. "
        "This is a demo prediction from simulated sensor readings."
    )
    st.markdown(
        "<div class='risk-banner "
        f"{risk_class(prediction)}'>"
        f"<h3>{escape(prediction)} risk · {probability:.0f}% probability</h3>"
        f"<p>{escape(message)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def sensor_network_view(latest: pd.DataFrame) -> None:
    width = 900
    height = 360
    pad = 70
    xmin, xmax = latest["lon"].min(), latest["lon"].max()
    ymin, ymax = latest["lat"].min(), latest["lat"].max()
    parts = [
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        "<rect width='100%' height='100%' rx='8' fill='#f8fafc'/>",
        "<rect x='35' y='35' width='830' height='290' rx='8' fill='none' stroke='#94a3b8' stroke-dasharray='10 8'/>",
        "<text x='50' y='62' fill='#334155' font-size='14' font-weight='700'>Huertgenwald MVP sensor grid</text>",
    ]
    for row in latest.itertuples():
        x = pad + ((row.lon - xmin) / max(xmax - xmin, 1e-9)) * (width - pad * 2)
        y = pad + (1 - ((row.lat - ymin) / max(ymax - ymin, 1e-9))) * (height - pad * 2)
        radius = 10 + (row.fire_probability_pct / 100) * 18
        color = risk_color(row.prediction)
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.1f}' fill='{color}' opacity='0.82'/>")
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius + 5:.1f}' fill='none' stroke='{color}' opacity='0.25' stroke-width='4'/>")
        parts.append(
            f"<text x='{x + radius + 8:.1f}' y='{y - 3:.1f}' fill='#0f172a' font-size='13' font-weight='700'>{escape(row.sensor_id)}</text>"
        )
        parts.append(
            f"<text x='{x + radius + 8:.1f}' y='{y + 13:.1f}' fill='#475569' font-size='12'>{escape(row.prediction)} · {row.fire_probability_pct:.0f}%</text>"
        )
    parts.append("<text x='50' y='342' fill='#475569' font-size='12'>Circle size and color represent predicted fire probability.</text>")
    parts.append("</svg>")
    st.markdown(f"<div class='svg-chart'>{''.join(parts)}</div>", unsafe_allow_html=True)


def signal_cards() -> None:
    items = [
        ("Heat", "Higher temperature raises stress."),
        ("Humidity", "Dry air increases ignition risk."),
        ("Wind", "Wind supports faster spread."),
        ("Smoke", "Smoke is the early warning signal."),
        ("CO", "CO supports combustion detection."),
        ("IR", "Infrared can indicate hot material."),
    ]
    cards = "".join(
        f"<div class='signal-card'><strong>{escape(title)}</strong>{escape(body)}</div>"
        for title, body in items
    )
    st.markdown(f"<div class='signal-grid'>{cards}</div>", unsafe_allow_html=True)


def round_numeric(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)
    return rounded


def insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'><strong>Insight:</strong> {text}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note-box'><strong>Note:</strong> {text}</div>", unsafe_allow_html=True)


def friendly_daily_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.rename(
            columns={
                "date": "Date",
                "weather_type": "Weather type",
                "temp_c_mean": "Average temp (C)",
                "temp_c_max": "Highest temp (C)",
                "temp_c_min": "Lowest temp (C)",
                "precipitation_mm_total": "Rain total (mm)",
                "wind10_mps_mean": "Average wind (m/s)",
            }
        )
        .pipe(round_numeric)
    )


def daily_weather_labels(paris_daily: pd.DataFrame) -> pd.DataFrame:
    labeled = paris_daily.copy()
    labels = []
    for row in labeled.itertuples():
        day_labels = []
        if row.temp_c_max >= 30:
            day_labels.append("Hot")
        if row.precipitation_mm_total >= 1:
            day_labels.append("Rainy")
        if row.wind10_mps_mean >= 5:
            day_labels.append("Windy")
        if not day_labels:
            day_labels.append("Mild/dry")
        labels.append(", ".join(day_labels))
    labeled["weather_type"] = labels
    return labeled


data = load_data()

st.title("Weather Dashboard")

overview_tab, live_nrw_tab, sensor_demo_tab, historical_nrw_tab, paris_tab, germany_tab, wildfire_tab, patterns_tab, simulation_tab = st.tabs(
    ["Overview", "Live NRW", "Sensor Demo", "Historical NRW", "Paris", "Germany", "Wildfire", "Patterns", "Simulation"]
)

with overview_tab:
    st.subheader("Overview")
    metric_cards(
        [
            ("Paris records", f"{len(data['paris']):,} hours"),
            ("Germany records", f"{len(data['germany']):,} hours"),
            ("Paris avg temp", f"{data['paris']['t2m_c'].mean():.1f} C"),
            ("Germany avg wind", f"{data['germany']['wind10_mps_mean'].mean():.1f} m/s"),
        ]
    )
    insight(
        "Paris shows a short summer period, while Germany gives us a full-year view of wind patterns."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Paris")
        st.write(
            "Warm overall. Rain is concentrated at the start of the period."
        )
        st.caption("Temperature and wind over time.")
        svg_line_chart(
            data["paris"].set_index("time_utc")[["t2m_c", "wind10_mps"]],
            height=300,
        )
    with right:
        st.markdown("#### Germany")
        st.write("Full-year wind data. October is the windiest month.")
        st.caption("Average wind by month.")
        svg_bar_chart(
            data["germany_monthly"].set_index("month")["wind10_mps_mean"],
            height=300,
        )

    note("Paris has 10 complete days, but 2026-07-19 is missing from the source data.")

with live_nrw_tab:
    st.subheader("Live NRW: Hürtgenwald")
    st.write("Current forecast conditions and Copernicus EFFIS fire-weather layer.")
    st.caption("Weather feed: Open-Meteo forecast. Fire danger map: Copernicus EFFIS WMS.")

    try:
        live_weather = load_hurtgenwald_weather()
        now_local = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None)
        future_weather = live_weather[live_weather["time"] >= now_local]
        latest = future_weather.iloc[0] if not future_weather.empty else live_weather.iloc[-1]
        worst = live_weather.sort_values(["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False).iloc[0]
        latest_risk = risk_label(int(latest["fire_weather_score"]), bool(latest["fire_30_30_30"]))
        worst_risk = risk_label(int(worst["fire_weather_score"]), bool(worst["fire_30_30_30"]))

        metric_cards(
            [
                ("Now", latest_risk),
                ("Temp", f"{latest['temperature_c']:.1f} C"),
                ("Humidity", f"{latest['humidity_pct']:.0f}%"),
                ("Wind", f"{latest['wind_kmh']:.1f} km/h"),
                ("Direction", f"{latest['wind_direction']} ({latest['wind_direction_deg']:.0f} deg)"),
            ]
        )

        insight(
            f"Highest forecast risk in the next 3 days: {worst_risk} on {worst['time']:%d %b, %H:%M}."
        )

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Forecast Inputs")
            st.caption("Temperature, humidity, wind, and rain for Hürtgenwald.")
            svg_line_chart(
                live_weather.set_index("time")[["temperature_c", "humidity_pct", "wind_kmh", "rain_mm"]],
                height=360,
            )
        with right:
            st.markdown("#### 30-30-30 Checks")
            rule_counts = pd.Series(
                {
                    "Temp >= 30 C": int((live_weather["temperature_c"] >= 30).sum()),
                    "Humidity <= 30%": int((live_weather["humidity_pct"] <= 30).sum()),
                    "Wind >= 30 km/h": int((live_weather["wind_kmh"] >= 30).sum()),
                    "All 3 together": int(live_weather["fire_30_30_30"].sum()),
                }
            )
            svg_bar_chart(rule_counts, height=360)

        st.markdown("#### Wind Direction")
        st.caption("Wind direction matters for possible fire spread.")
        svg_bar_chart(direction_counts(live_weather, "wind_direction"), height=260)

        st.markdown("#### Copernicus EFFIS Fire Weather Index")
        st.caption("EFFIS WMS layer for the Hürtgenwald/NRW area.")
        st.image(effis_map_url("mf010.fwi"), use_container_width=True)

        with st.expander("Show live forecast table"):
            table = live_weather.copy()
            table["risk"] = [
                risk_label(int(row.fire_weather_score), bool(row.fire_30_30_30))
                for row in table.itertuples()
            ]
            html_table(
                table.rename(
                    columns={
                        "time": "Time",
                        "temperature_c": "Temp (C)",
                        "humidity_pct": "Humidity (%)",
                        "rain_mm": "Rain (mm)",
                        "wind_kmh": "Wind (km/h)",
                        "wind_direction": "Direction",
                        "risk": "Risk",
                    }
                )[
                    [
                        "Time",
                        "Risk",
                        "Temp (C)",
                        "Humidity (%)",
                        "Rain (mm)",
                        "Wind (km/h)",
                        "Direction",
                    ]
                ].pipe(round_numeric),
                use_container_width=True,
                hide_index=True,
            )
    except Exception as exc:
        st.error("Live data is temporarily unavailable.")
        st.caption(str(exc))
        st.markdown("#### Copernicus EFFIS Fire Weather Index")
        st.image(effis_map_url("mf010.fwi"), use_container_width=True)

with sensor_demo_tab:
    st.subheader("Fire Prediction Demo")
    st.write("Simulated ground-sensor feed for the CTRL-F MVP.")
    note("Demo data only. This shows how live sensor readings would become a fire-risk prediction.")

    scenario = st.selectbox("Demo scenario", ["Normal with one hotspot", "Regenerate sample"], key="sensor_scenario")
    seed = 42 if scenario == "Normal with one hotspot" else int(pd.Timestamp.now().timestamp()) % 100000
    sensor_data = load_sensor_demo(seed)
    latest = sensor_data.sort_values("time").groupby("sensor_id", as_index=False).tail(1)
    highest = latest.sort_values("fire_probability_pct", ascending=False).iloc[0]

    risk_banner(
        str(highest["sensor_id"]),
        str(highest["zone"]),
        str(highest["prediction"]),
        float(highest["fire_probability_pct"]),
    )

    metric_cards(
        [
            ("Sensor", f"{highest['sensor_id']}"),
            ("Zone", str(highest["zone"])),
            ("Smoke", f"{highest['smoke_ppm']:.1f} ppm"),
            ("CO", f"{highest['co_ppm']:.1f} ppm"),
            ("Battery", f"{highest['battery_pct']:.0f}%"),
        ]
    )

    insight(
        "The prediction rises when multiple signals move together: smoke, CO, heat, low humidity, wind, and IR."
    )

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Sensor Network")
        st.caption("Risk level by sensor location.")
        sensor_network_view(latest)
    with right:
        st.markdown("#### Latest Readings")
        latest_table = latest.sort_values("fire_probability_pct", ascending=False)[
            [
                "sensor_id",
                "zone",
                "prediction",
                "fire_probability_pct",
                "temperature_c",
                "humidity_pct",
                "smoke_ppm",
                "co_ppm",
                "battery_pct",
            ]
        ].rename(
            columns={
                "sensor_id": "Sensor",
                "zone": "Zone",
                "prediction": "Prediction",
                "fire_probability_pct": "Fire probability (%)",
                "temperature_c": "Temp (C)",
                "humidity_pct": "Humidity (%)",
                "smoke_ppm": "Smoke (ppm)",
                "co_ppm": "CO (ppm)",
                "battery_pct": "Battery (%)",
            }
        )
        html_table(round_numeric(latest_table))

    st.markdown("#### Hotspot Timeline")
    st.caption("The demo hotspot escalates as smoke, CO, and IR rise together.")
    hotspot_history = sensor_data[sensor_data["sensor_id"] == highest["sensor_id"]].set_index("time")
    svg_line_chart(
        hotspot_history[["fire_probability_pct", "temperature_c", "humidity_pct", "smoke_ppm", "co_ppm"]],
        height=360,
    )

    st.markdown("#### Prediction Signals")
    signal_cards()

with historical_nrw_tab:
    st.subheader("Historical NRW: Hürtgenwald")
    st.write("Review past weather conditions around an incident window.")

    default_end = pd.Timestamp.today().date()
    default_start = default_end - pd.Timedelta(days=14)
    selected = st.date_input(
        "Incident window",
        (default_start, default_end),
        max_value=default_end,
        key="historical_nrw_window",
    )
    if isinstance(selected, tuple) and len(selected) == 2:
        hist_start, hist_end = selected
    else:
        hist_start, hist_end = default_start, default_end

    if hist_start > hist_end:
        st.error("Start date must be before end date.")
    else:
        try:
            history, source_name = load_hurtgenwald_history(hist_start, hist_end)
            if history.empty:
                st.warning("No historical rows returned for this date range.")
            else:
                full_rule_hours = int(history["fire_30_30_30"].sum())
                near_risk_hours = int((history["fire_weather_score"] >= 2).sum())
                worst = history.sort_values(
                    ["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False
                ).iloc[0]
                worst_risk = risk_label(int(worst["fire_weather_score"]), bool(worst["fire_30_30_30"]))

                metric_cards(
                    [
                        ("Data source", source_name),
                        ("30-30-30 hours", f"{full_rule_hours}"),
                        ("Near-risk hours", f"{near_risk_hours}"),
                        ("Max temp", f"{history['temperature_c'].max():.1f} C"),
                        ("Max wind", f"{history['wind_kmh'].max():.1f} km/h"),
                    ]
                )

                insight(
                    f"Highest risk: {worst_risk} on {worst['time']:%d %b, %H:%M}. "
                    "This shows conditions that could support spread; it does not identify the ignition cause."
                )

                left, right = st.columns([2, 1])
                with left:
                    st.markdown("#### Weather Before And During The Incident")
                    svg_line_chart(
                        history.set_index("time")[["temperature_c", "humidity_pct", "wind_kmh", "rain_mm"]],
                        height=360,
                    )
                with right:
                    st.markdown("#### Rule Checks")
                    rule_counts = pd.Series(
                        {
                            "Temp >= 30 C": int((history["temperature_c"] >= 30).sum()),
                            "Humidity <= 30%": int((history["humidity_pct"] <= 30).sum()),
                            "Wind >= 30 km/h": int((history["wind_kmh"] >= 30).sum()),
                            "All 3 together": full_rule_hours,
                        }
                    )
                    svg_bar_chart(rule_counts, height=360)

                st.markdown("#### Wind Direction")
                st.caption("Useful for reconstructing likely spread direction.")
                svg_bar_chart(direction_counts(history, "wind_direction"), height=260)

                daily_history = (
                    history.assign(Date=history["time"].dt.date)
                    .groupby("Date", as_index=False)
                    .agg(
                        Max_Temp_C=("temperature_c", "max"),
                        Lowest_Humidity_Pct=("humidity_pct", "min"),
                        Rain_Total_Mm=("rain_mm", "sum"),
                        Max_Wind_Kmh=("wind_kmh", "max"),
                        Main_Direction=("wind_direction", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
                        Rule_Hours=("fire_30_30_30", "sum"),
                        Near_Risk_Hours=("fire_weather_score", lambda s: int((s >= 2).sum())),
                    )
                    .rename(
                        columns={
                            "Max_Temp_C": "Max temp (C)",
                            "Lowest_Humidity_Pct": "Lowest humidity (%)",
                            "Rain_Total_Mm": "Rain total (mm)",
                            "Max_Wind_Kmh": "Max wind (km/h)",
                            "Main_Direction": "Main direction",
                            "Rule_Hours": "30-30-30 hours",
                            "Near_Risk_Hours": "Near-risk hours",
                        }
                    )
                )
                st.markdown("#### Daily Summary")
                html_table(round_numeric(daily_history))

                with st.expander("Show hourly historical data"):
                    table = history.copy()
                    table["Risk"] = [
                        risk_label(int(row.fire_weather_score), bool(row.fire_30_30_30))
                        for row in table.itertuples()
                    ]
                    html_table(
                        table.rename(
                            columns={
                                "time": "Time",
                                "temperature_c": "Temp (C)",
                                "humidity_pct": "Humidity (%)",
                                "rain_mm": "Rain (mm)",
                                "wind_kmh": "Wind (km/h)",
                                "wind_direction": "Direction",
                            }
                        )[
                            [
                                "Time",
                                "Risk",
                                "Temp (C)",
                                "Humidity (%)",
                                "Rain (mm)",
                                "Wind (km/h)",
                                "Direction",
                            ]
                        ].pipe(round_numeric),
                        max_rows=500,
                    )
        except Exception as exc:
            st.error("Historical data is temporarily unavailable.")
            st.caption(str(exc))

with paris_tab:
    st.subheader("Paris Weather")
    st.write("Temperature, rain, and wind for the available Paris dates.")
    paris = date_filter(data["paris"], "paris_date_filter")
    daily = daily_weather_labels(data["paris_daily"])
    paris_direction_deg, paris_direction = main_wind_direction(paris, "u10", "v10")

    metric_cards(
        [
            ("Mean temp", f"{paris['t2m_c'].mean():.1f} C"),
            ("Max temp", f"{paris['t2m_c'].max():.1f} C"),
            ("Total rain", f"{paris['tp_mm'].sum():.1f} mm"),
            ("Mean wind", f"{paris['wind10_mps'].mean():.1f} m/s"),
            ("Main direction", f"{paris_direction} ({paris_direction_deg:.0f} deg)"),
        ]
    )
    insight(
        "Warm days, cooler early mornings, hottest in the afternoon."
    )

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Hourly Trend")
        st.caption("Temperature, dew point, wind, and rain.")
        svg_line_chart(paris.set_index("time_utc")[["t2m_c", "d2m_c", "wind10_mps", "tp_mm"]], height=360)
    with right:
        st.markdown("#### Daily Labels")
        st.caption("Hot, rainy, windy, or mild/dry.")
        html_table(
            friendly_daily_table(daily[
                [
                    "date",
                    "weather_type",
                    "temp_c_mean",
                    "temp_c_max",
                    "precipitation_mm_total",
                    "wind10_mps_mean",
                ]
            ]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Wind Direction")
    st.caption("How often the wind came from each compass direction.")
    svg_bar_chart(direction_counts(paris, "wind_direction"), height=260)

    st.markdown("#### Daily Cycle")
    st.caption("Average by hour.")
    cycle = paris.groupby("hour", as_index=False).agg(
        temp_c=("t2m_c", "mean"),
        wind_mps=("wind10_mps", "mean"),
        rain_mm=("tp_mm", "mean"),
    )
    svg_line_chart(
        cycle.rename(columns={"temp_c": "Temperature (C)", "wind_mps": "Wind (m/s)", "rain_mm": "Rain (mm)"})
        .set_index("hour"),
        height=300,
    )

    with st.expander("Show detailed Paris rows"):
        html_table(round_numeric(paris), use_container_width=True, hide_index=True)

with germany_tab:
    st.subheader("Germany Wind")
    st.write("Wind patterns across Germany during 2025.")
    germany = date_filter(data["germany"], "germany_date_filter")
    germany_direction_deg, germany_direction = main_wind_direction(germany, "u10_mean", "v10_mean")

    metric_cards(
        [
            ("Mean wind", f"{germany['wind10_mps_mean'].mean():.1f} m/s"),
            ("Max wind", f"{germany['wind10_mps_max'].max():.1f} m/s"),
            ("P90 wind", f"{germany['wind10_mps_p90'].mean():.1f} m/s"),
            ("Filtered hours", f"{len(germany):,}"),
            ("Main direction", f"{germany_direction} ({germany_direction_deg:.0f} deg)"),
        ]
    )
    insight(
        "Autumn and winter are windier. The northwest grid cells stand out."
    )

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Wind Over Time")
        st.caption("Average wind and high-wind level.")
        svg_line_chart(germany.set_index("time_utc")[["wind10_mps_mean", "wind10_mps_p90"]], height=360)
    with right:
        st.markdown("#### Monthly Wind")
        st.caption("Seasonal pattern.")
        svg_bar_chart(data["germany_monthly"].set_index("month")["wind10_mps_mean"], height=360)

    st.markdown("#### Wind Direction")
    st.caption("How often the average wind came from each compass direction.")
    svg_bar_chart(direction_counts(germany, "wind_direction"), height=260)

    st.markdown("#### Wind By Location")
    st.caption("Bigger and darker points are windier.")
    map_data = data["germany_grid"].rename(columns={"latitude": "lat", "longitude": "lon"})
    svg_scatter_chart(
        map_data,
        x="lon",
        y="lat",
        size="wind10_mps_mean",
        color="wind10_mps_mean",
        height=420,
    )

    with st.expander("Show strongest wind locations"):
        strongest = data["germany_grid"].sort_values("wind10_mps_mean", ascending=False).head(20)
        html_table(
            strongest.rename(
                columns={
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "wind10_mps_mean": "Average wind (m/s)",
                    "wind10_mps_max": "Highest wind (m/s)",
                    "wind10_mps_p90": "High-wind level (m/s)",
                    "wind_direction": "Direction",
                }
            )[
                [
                    "Latitude",
                    "Longitude",
                    "Average wind (m/s)",
                    "Highest wind (m/s)",
                    "High-wind level (m/s)",
                    "Direction",
                ]
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )

with wildfire_tab:
    st.subheader("Wildfire Weather")
    st.write("30-30-30 rule: 30 C or hotter, relative humidity at or below 30%, and wind at least 30 km/h.")

    fire = date_filter(data["paris"], "wildfire_date_filter")
    fire_direction_deg, fire_direction = main_wind_direction(fire, "u10", "v10")
    fire_hours = int(fire["fire_30_30_30"].sum())
    near_fire_hours = int((fire["fire_weather_score"] >= 2).sum())

    metric_cards(
        [
            ("30-30-30 hours", f"{fire_hours}"),
            ("Near-risk hours", f"{near_fire_hours}"),
            ("Lowest humidity", f"{fire['relative_humidity_pct'].min():.1f}%"),
            ("Max wind", f"{fire['wind10_kmh'].max():.1f} km/h"),
            ("Main direction", f"{fire_direction} ({fire_direction_deg:.0f} deg)"),
        ]
    )

    if fire_hours:
        insight("The selected period includes hours that match the 30-30-30 fire-weather rule.")
    else:
        insight("No full 30-30-30 events appear in the selected Paris data.")

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Fire Weather Inputs")
        st.caption("Temperature, relative humidity, and wind speed in km/h.")
        svg_line_chart(
            fire.set_index("time_utc")[["t2m_c", "relative_humidity_pct", "wind10_kmh"]],
            height=360,
        )
    with right:
        st.markdown("#### Rule Checks")
        rule_counts = pd.Series(
            {
                "Temp >= 30 C": int((fire["t2m_c"] >= 30).sum()),
                "Humidity <= 30%": int((fire["relative_humidity_pct"] <= 30).sum()),
                "Wind >= 30 km/h": int((fire["wind10_kmh"] >= 30).sum()),
                "All 3 together": fire_hours,
            }
        )
        svg_bar_chart(rule_counts, height=360)

    st.markdown("#### Wind Direction")
    st.caption("Important for understanding possible spread direction.")
    svg_bar_chart(direction_counts(fire, "wind_direction"), height=260)

    st.markdown("#### Daily Fire Weather Summary")
    fire_daily = (
        fire.assign(Date=fire["time_utc"].dt.date)
        .groupby("Date", as_index=False)
        .agg(
            Max_Temp_C=("t2m_c", "max"),
            Lowest_Humidity_Pct=("relative_humidity_pct", "min"),
            Max_Wind_Kmh=("wind10_kmh", "max"),
            Main_Direction=("wind_direction", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
            Rule_Hours=("fire_30_30_30", "sum"),
            Near_Risk_Hours=("fire_weather_score", lambda s: int((s >= 2).sum())),
        )
        .rename(
            columns={
                "Max_Temp_C": "Max temp (C)",
                "Lowest_Humidity_Pct": "Lowest humidity (%)",
                "Max_Wind_Kmh": "Max wind (km/h)",
                "Main_Direction": "Main direction",
                "Rule_Hours": "30-30-30 hours",
                "Near_Risk_Hours": "Near-risk hours",
            }
        )
    )
    html_table(round_numeric(fire_daily), use_container_width=True, hide_index=True)

    with st.expander("Show hourly fire-weather data"):
        html_table(
            fire[
                [
                    "time_utc",
                    "t2m_c",
                    "relative_humidity_pct",
                    "wind10_kmh",
                    "wind_direction",
                    "fire_weather_score",
                    "fire_30_30_30",
                ]
            ]
            .rename(
                columns={
                    "time_utc": "Time",
                    "t2m_c": "Temp (C)",
                    "relative_humidity_pct": "Humidity (%)",
                    "wind10_kmh": "Wind (km/h)",
                    "wind_direction": "Direction",
                    "fire_weather_score": "Rule score",
                    "fire_30_30_30": "30-30-30",
                }
            )
            .pipe(round_numeric),
            use_container_width=True,
            hide_index=True,
        )

with patterns_tab:
    st.subheader("Pattern Finder")
    st.write("Key dates, months, and daily cycles.")

    paris_daily = daily_weather_labels(data["paris_daily"])
    germany_daily = data["germany_daily"]
    germany_monthly = data["germany_monthly"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Paris standouts")
        hottest = paris_daily.loc[paris_daily["temp_c_max"].idxmax()]
        wettest = paris_daily.loc[paris_daily["precipitation_mm_total"].idxmax()]
        windiest = paris_daily.loc[paris_daily["wind10_mps_mean"].idxmax()]
        st.write(f"Hottest day: **{hottest['date'].date()}**, {hottest['temp_c_max']:.1f} C")
        st.write(f"Wettest day: **{wettest['date'].date()}**, {wettest['precipitation_mm_total']:.1f} mm")
        st.write(f"Windiest day: **{windiest['date'].date()}**, {windiest['wind10_mps_mean']:.1f} m/s")
        html_table(friendly_daily_table(paris_daily), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Germany standouts")
        windiest_month = germany_monthly.loc[germany_monthly["wind10_mps_mean"].idxmax()]
        calmest_month = germany_monthly.loc[germany_monthly["wind10_mps_mean"].idxmin()]
        windiest_day = germany_daily.loc[germany_daily["wind10_mps_mean"].idxmax()]
        st.write(f"Windiest month: **{windiest_month['month']}**, {windiest_month['wind10_mps_mean']:.1f} m/s")
        st.write(f"Calmest month: **{calmest_month['month']}**, {calmest_month['wind10_mps_mean']:.1f} m/s")
        st.write(f"Windiest day: **{windiest_day['date'].date()}**, {windiest_day['wind10_mps_mean']:.1f} m/s")
        html_table(
            germany_monthly.rename(
                columns={
                    "month": "Month",
                    "wind10_mps_mean": "Average wind (m/s)",
                    "wind10_mps_min": "Lowest wind (m/s)",
                    "wind10_mps_max": "Highest wind (m/s)",
                    "wind10_mps_p90": "High-wind level (m/s)",
                }
            ).pipe(round_numeric),
            use_container_width=True,
            hide_index=True,
        )

    insight(
        "Paris patterns center on heat and rain. Germany patterns center on season and location."
    )

    st.markdown("#### Average Daily Cycles")
    st.caption("Paris temperature and Germany wind by hour.")
    p_cycle = data["paris"].groupby("hour", as_index=False)["t2m_c"].mean().rename(columns={"t2m_c": "Paris temp C"})
    g_cycle = (
        data["germany"]
        .groupby("hour", as_index=False)["wind10_mps_mean"]
        .mean()
        .rename(columns={"wind10_mps_mean": "Germany wind m/s"})
    )
    cycles = p_cycle.merge(g_cycle, on="hour")
    svg_line_chart(cycles.set_index("hour"), height=320)

with simulation_tab:
    st.subheader("Scenario Simulation")
    st.write("Adjust the sliders to test simple weather scenarios.")
    dataset = st.radio("Scenario dataset", ["Paris", "Germany"], horizontal=True)

    if dataset == "Paris":
        base = date_filter(data["paris"], "sim_paris_filter")
        sim_direction_deg, sim_direction = main_wind_direction(base, "u10", "v10")
        temp_shift = st.slider("Temperature shift (C)", -10.0, 10.0, 0.0, 0.5)
        wind_factor = st.slider("Wind multiplier", 0.0, 3.0, 1.0, 0.05)
        precip_factor = st.slider("Precipitation multiplier", 0.0, 5.0, 1.0, 0.1)
        base["sim_temp_c"] = base["t2m_c"] + temp_shift
        base["sim_wind_mps"] = base["wind10_mps"] * wind_factor
        base["sim_precip_mm"] = base["tp_mm"] * precip_factor

        metric_cards(
            [
                ("Sim mean temp", f"{base['sim_temp_c'].mean():.1f} C"),
                ("Sim max temp", f"{base['sim_temp_c'].max():.1f} C"),
                ("Sim rain", f"{base['sim_precip_mm'].sum():.1f} mm"),
                ("Sim wind", f"{base['sim_wind_mps'].mean():.1f} m/s"),
                ("Direction", f"{sim_direction} ({sim_direction_deg:.0f} deg)"),
            ]
        )
        insight(
            "Temperature shift changes heat. Wind and rain multipliers scale the original pattern."
        )
        svg_line_chart(base.set_index("time_utc")[["sim_temp_c", "sim_wind_mps", "sim_precip_mm"]], height=360)
        st.download_button(
            "Download simulated Paris CSV",
            base.to_csv(index=False).encode("utf-8"),
            "simulated_paris_weather.csv",
            "text/csv",
        )
    else:
        base = date_filter(data["germany"], "sim_germany_filter")
        sim_direction_deg, sim_direction = main_wind_direction(base, "u10_mean", "v10_mean")
        wind_factor = st.slider("Wind multiplier", 0.0, 3.0, 1.0, 0.05)
        high_wind_boost = st.slider("High-wind boost", 0.0, 2.0, 1.0, 0.05)
        base["sim_wind_mps"] = base["wind10_mps_mean"] * wind_factor
        base["sim_p90_wind_mps"] = base["wind10_mps_p90"] * wind_factor * high_wind_boost

        metric_cards(
            [
                ("Sim mean wind", f"{base['sim_wind_mps'].mean():.1f} m/s"),
                ("Sim max wind", f"{base['sim_wind_mps'].max():.1f} m/s"),
                ("Sim P90 wind", f"{base['sim_p90_wind_mps'].mean():.1f} m/s"),
                ("Hours", f"{len(base):,}"),
                ("Direction", f"{sim_direction} ({sim_direction_deg:.0f} deg)"),
            ]
        )
        insight(
            "Wind multiplier changes the whole year. High-wind boost raises stronger wind periods."
        )
        svg_line_chart(base.set_index("time_utc")[["sim_wind_mps", "sim_p90_wind_mps"]], height=360)
        st.download_button(
            "Download simulated Germany CSV",
            base.to_csv(index=False).encode("utf-8"),
            "simulated_germany_wind.csv",
            "text/csv",
        )

with st.expander("Files created"):
    st.write(
        {
            "Excel workbook": str(DATA_DIR / "weather_analysis_workbook.xlsx"),
            "Pattern report": str(DATA_DIR / "weather_pattern_analysis.md"),
            "Paris CSV": str(DATA_DIR / "paris_hourly_area_mean.csv"),
            "Germany CSV": str(DATA_DIR / "germany_hourly_area_mean.csv"),
        }
    )
