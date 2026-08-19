from __future__ import annotations

from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from live_pipeline import (
    GLOBAL_FIRE_LOCATIONS,
    effis_map_url,
    fetch_global_fire_weather,
    fetch_location_history,
    fetch_location_weather,
    risk_label,
)
from sensor_simulation import generate_sensor_demo


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "weather_analysis"
GLOBAL_MONITOR_CACHE_VERSION = "global-areas-v2"


st.set_page_config(page_title="CTRL-F Fire Prediction Alerts", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(34,197,94,0.10), transparent 28%),
            radial-gradient(circle at 88% 8%, rgba(220,38,38,0.12), transparent 30%),
            #090d13;
    }
    .block-container { padding-top: 1.1rem; }
    [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #26313d;
    }
    [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 10px;
    }
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
        background: linear-gradient(135deg, #ecfdf5, #eff6ff);
        border: 1px solid #a7f3d0;
        color: #12382a;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
    }
    .insight-box strong { color: #0d2f23; }
    .note-box {
        background: linear-gradient(135deg, #fff7ed, #fef3c7);
        border: 1px solid #fbbf24;
        color: #4b3410;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
    }
    .note-box strong { color: #3a2609; }
    .temp-alert {
        background: linear-gradient(135deg, #991b1b, #ef4444);
        border: 1px solid rgba(254,202,202,0.45);
        border-radius: 12px;
        color: #fff7ed;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
        box-shadow: 0 16px 34px rgba(127, 29, 29, 0.28);
    }
    .temp-alert h4 {
        margin: 0 0 6px 0;
        font-size: 1.08rem;
    }
    .temp-alert p {
        margin: 0;
        color: #ffedd5;
        font-size: 0.94rem;
    }
    .temp-ok {
        background: linear-gradient(135deg, #064e3b, #15803d);
        border: 1px solid rgba(187,247,208,0.34);
        border-radius: 12px;
        color: #f0fdf4;
        padding: 12px 14px;
        margin: 10px 0 16px 0;
    }
    .temp-ok h4 {
        margin: 0 0 4px 0;
        font-size: 1rem;
    }
    .temp-ok p {
        margin: 0;
        color: #dcfce7;
        font-size: 0.9rem;
    }
    .country-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
        margin: 12px 0 18px 0;
    }
    .country-card {
        background: linear-gradient(180deg, #ffffff, #f5f8fb);
        border: 1px solid #dbe3ea;
        border-top: 5px solid #22c55e;
        border-radius: 12px;
        color: #10212f;
        padding: 15px;
        box-shadow: 0 12px 26px rgba(2, 6, 23, 0.12);
    }
    .country-card.high { border-top-color: #f97316; }
    .country-card.critical { border-top-color: #dc2626; }
    .country-card.elevated { border-top-color: #eab308; }
    .country-card h4 {
        margin: 0 0 4px 0;
        font-size: 1.05rem;
    }
    .country-card .site {
        color: #475569;
        font-size: 0.86rem;
        margin-bottom: 10px;
    }
    .country-card .risk {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 9px;
        margin-bottom: 10px;
        font-size: 0.78rem;
        font-weight: 800;
        background: #dcfce7;
        color: #166534;
    }
    .country-card.high .risk { background: #ffedd5; color: #9a3412; }
    .country-card.critical .risk { background: #fee2e2; color: #991b1b; }
    .country-card.elevated .risk { background: #fef9c3; color: #854d0e; }
    .country-card dl {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        margin: 0;
    }
    .country-card dt {
        color: #64748b;
        font-size: 0.76rem;
    }
    .country-card dd {
        margin: 2px 0 0 0;
        font-weight: 800;
    }
    .bi-panel {
        background:
            linear-gradient(180deg, rgba(15,23,42,0.86), rgba(15,23,42,0.72)),
            radial-gradient(circle at 16% 0%, rgba(34,197,94,0.16), transparent 32%);
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0 14px 0;
        box-shadow: 0 16px 34px rgba(2,6,23,0.28);
    }
    .bi-title {
        color: #f8fafc;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .bi-subtitle {
        color: #b6c3d1;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    .bi-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin: 10px 0 16px 0;
    }
    .bi-kpi {
        background: linear-gradient(180deg, #ffffff, #eef4f8);
        border: 1px solid #d7e1ea;
        border-radius: 10px;
        padding: 12px 14px;
        color: #0f2233;
        box-shadow: 0 12px 26px rgba(2,6,23,0.16);
        min-height: 94px;
    }
    .bi-kpi span {
        display: block;
        color: #53677a;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .bi-kpi strong {
        display: block;
        font-size: 1.62rem;
        line-height: 1.05;
    }
    .bi-kpi em {
        display: block;
        color: #64748b;
        font-style: normal;
        font-size: 0.82rem;
        margin-top: 7px;
    }
    .bi-kpi.alert {
        background: linear-gradient(135deg, #991b1b, #ef4444);
        border-color: rgba(254,202,202,0.45);
        color: #fff7ed;
    }
    .bi-kpi.alert span,
    .bi-kpi.alert em { color: #fee2e2; }
    .area-board {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 10px;
        margin: 8px 0 16px 0;
    }
    .area-tile {
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, #ffffff, #f6f9fb);
        color: #10212f;
        border: 1px solid #d8e2ec;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 13px 14px 12px 14px;
        box-shadow: 0 10px 22px rgba(2,6,23,0.12);
    }
    .area-tile.high { border-left-color: #f97316; }
    .area-tile.critical { border-left-color: #dc2626; }
    .area-tile.elevated { border-left-color: #eab308; }
    .area-tile h4 {
        margin: 0 0 3px 0;
        font-size: 1rem;
    }
    .area-tile .area-meta {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 10px;
    }
    .area-tile .score-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
        margin-bottom: 8px;
    }
    .area-tile .score {
        font-size: 1.28rem;
        font-weight: 900;
    }
    .area-tile .badge {
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 0.75rem;
        font-weight: 800;
        background: #dcfce7;
        color: #166534;
    }
    .area-tile.high .badge { background: #ffedd5; color: #9a3412; }
    .area-tile.critical .badge { background: #fee2e2; color: #991b1b; }
    .area-tile.elevated .badge { background: #fef9c3; color: #854d0e; }
    .area-bar {
        height: 8px;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .area-bar span {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
    }
    .area-facts {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
        color: #334155;
        font-size: 0.78rem;
    }
    .area-facts strong {
        display: block;
        color: #10212f;
        font-size: 0.9rem;
        margin-top: 2px;
    }
    .bi-layout {
        display: grid;
        grid-template-columns: minmax(420px, 1.55fr) minmax(320px, 1fr);
        gap: 14px;
        align-items: start;
    }
    @media (max-width: 1000px) {
        .bi-layout { grid-template-columns: 1fr; }
    }
    .control-panel {
        background:
            linear-gradient(135deg, rgba(8,47,73,0.86), rgba(20,83,45,0.72)),
            radial-gradient(circle at 92% 10%, rgba(248,113,113,0.22), transparent 28%);
        border: 1px solid rgba(125,211,252,0.22);
        border-left: 5px solid #22c55e;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0 14px 0;
        color: #f8fafc;
        box-shadow: 0 18px 38px rgba(2,6,23,0.30);
    }
    .control-panel strong {
        display: block;
        font-size: 1.02rem;
        margin-bottom: 4px;
    }
    .control-panel span {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .action-panel {
        background: linear-gradient(180deg, #ffffff, #f5f8fb);
        border: 1px solid #d8e2ec;
        border-left: 5px solid #0ea5e9;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
        color: #10212f;
        box-shadow: 0 14px 30px rgba(2,6,23,0.13);
    }
    .action-panel.critical { border-left-color: #dc2626; }
    .action-panel.high { border-left-color: #f97316; }
    .action-panel.elevated { border-left-color: #eab308; }
    .action-panel h4 {
        margin: 0 0 8px 0;
        font-size: 1.03rem;
    }
    .action-list {
        display: grid;
        gap: 8px;
        margin: 0;
        padding: 0;
        list-style: none;
    }
    .action-list li {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 9px 10px;
        color: #263746;
    }
    .product-hero {
        background:
            linear-gradient(135deg, rgba(5,46,22,0.92), rgba(15,23,42,0.78) 52%, rgba(127,29,29,0.76)),
            radial-gradient(circle at 75% 10%, rgba(248,113,113,0.20), transparent 30%);
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 18px;
        padding: 28px;
        color: #f8fafc;
        margin: 8px 0 18px 0;
        box-shadow: 0 22px 48px rgba(2,6,23,0.34);
    }
    .product-hero-grid {
        display: grid;
        grid-template-columns: minmax(320px, 1.15fr) minmax(300px, 0.85fr);
        gap: 20px;
        align-items: center;
    }
    @media (max-width: 980px) {
        .product-hero-grid { grid-template-columns: 1fr; }
    }
    .product-kicker {
        color: #86efac;
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .product-hero h2 {
        margin: 0;
        max-width: 820px;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1.02;
    }
    .product-hero p {
        max-width: 760px;
        color: #dbeafe;
        font-size: 1.02rem;
        margin: 14px 0 0 0;
    }
    .product-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
        margin-top: 22px;
    }
    .product-stat {
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(15,23,42,0.45);
        border-radius: 12px;
        padding: 12px 14px;
    }
    .product-stat span {
        display: block;
        color: #93c5fd;
        font-size: 0.76rem;
        margin-bottom: 6px;
    }
    .product-stat strong {
        display: block;
        font-size: 1.15rem;
    }
    .demo-console {
        background: rgba(248,250,252,0.96);
        border: 1px solid rgba(226,232,240,0.84);
        border-radius: 16px;
        padding: 14px;
        color: #10212f;
        box-shadow: 0 22px 42px rgba(2,6,23,0.26);
    }
    .console-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 12px;
    }
    .console-title strong {
        display: block;
        font-size: 0.98rem;
    }
    .console-title span {
        display: block;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 2px;
    }
    .console-badge {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 999px;
        padding: 6px 9px;
        font-size: 0.76rem;
        font-weight: 900;
        white-space: nowrap;
    }
    .console-map {
        position: relative;
        height: 210px;
        border-radius: 12px;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(22,101,52,0.20), transparent 36%),
            linear-gradient(45deg, #dbeafe 0 18%, #dcfce7 18% 45%, #fef9c3 45% 62%, #fee2e2 62% 100%);
        border: 1px solid #d8e2ec;
        margin-bottom: 12px;
    }
    .console-road {
        position: absolute;
        height: 2px;
        background: rgba(71,85,105,0.40);
        transform-origin: left center;
    }
    .road-a { width: 72%; left: 12%; top: 42%; transform: rotate(-18deg); }
    .road-b { width: 62%; left: 25%; top: 62%; transform: rotate(20deg); }
    .road-c { width: 44%; left: 38%; top: 28%; transform: rotate(52deg); }
    .hotspot {
        position: absolute;
        width: 30px;
        height: 30px;
        left: 58%;
        top: 44%;
        border-radius: 999px;
        background: #dc2626;
        border: 6px solid rgba(254,202,202,0.92);
        box-shadow: 0 0 0 12px rgba(220,38,38,0.12);
    }
    .wind-arrow {
        position: absolute;
        left: 62%;
        top: 52%;
        width: 84px;
        height: 3px;
        background: #2563eb;
        transform: rotate(-35deg);
    }
    .wind-arrow:after {
        content: "";
        position: absolute;
        right: -2px;
        top: -5px;
        border-left: 10px solid #2563eb;
        border-top: 6px solid transparent;
        border-bottom: 6px solid transparent;
    }
    .console-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
    }
    .console-metric {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 9px;
    }
    .console-metric span {
        display: block;
        color: #64748b;
        font-size: 0.72rem;
        margin-bottom: 4px;
    }
    .console-metric strong {
        display: block;
        font-size: 0.95rem;
    }
    .phase-grid,
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 12px;
        margin: 10px 0 18px 0;
    }
    .phase-card,
    .feature-card {
        background: linear-gradient(180deg, #ffffff, #f4f8fb);
        border: 1px solid #d8e2ec;
        border-radius: 12px;
        padding: 15px;
        color: #10212f;
        box-shadow: 0 12px 28px rgba(2,6,23,0.12);
    }
    .phase-card {
        border-top: 5px solid #22c55e;
    }
    .phase-card:nth-child(2) { border-top-color: #f97316; }
    .phase-card:nth-child(3) { border-top-color: #2563eb; }
    .phase-card span,
    .feature-card span {
        display: inline-block;
        color: #0f766e;
        font-size: 0.74rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }
    .phase-card h4,
    .feature-card h4 {
        margin: 0 0 7px 0;
        font-size: 1.05rem;
    }
    .phase-card p,
    .feature-card p {
        margin: 0;
        color: #475569;
        font-size: 0.92rem;
    }
    .demo-cta {
        background: linear-gradient(135deg, #ecfdf5, #eff6ff);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 16px;
        color: #12382a;
        margin: 10px 0 18px 0;
    }
    .demo-step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
        margin: 10px 0 18px 0;
    }
    .demo-step {
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 14px 30px rgba(2,6,23,0.20);
    }
    .demo-step span {
        display: inline-flex;
        width: 26px;
        height: 26px;
        align-items: center;
        justify-content: center;
        background: #22c55e;
        color: #052e16;
        border-radius: 999px;
        font-weight: 900;
        margin-bottom: 9px;
    }
    .demo-step h4 {
        margin: 0 0 6px 0;
        font-size: 1rem;
    }
    .demo-step p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 10px 0 18px 0;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff, #f1f5f9);
        border: 1px solid #d8e1ea;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.10);
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
        background: linear-gradient(180deg, #ffffff, #f5f8fb);
        border: 1px solid #d7e1ea;
        border-radius: 8px;
        padding: 8px;
        margin: 8px 0 16px 0;
        box-shadow: 0 12px 28px rgba(2, 6, 23, 0.12);
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
        padding: 16px 18px;
        margin: 12px 0 18px 0;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }
    .risk-banner h3 {
        margin: 0 0 6px 0;
        font-size: 1.2rem;
    }
    .risk-banner p {
        margin: 0;
        font-size: 0.95rem;
    }
    .risk-critical { background: linear-gradient(135deg, #991b1b, #dc2626); color: #fff7ed; }
    .risk-high { background: linear-gradient(135deg, #9a3412, #f97316); color: #fff7ed; }
    .risk-watch { background: linear-gradient(135deg, #a16207, #eab308); color: #111827; }
    .risk-low { background: linear-gradient(135deg, #166534, #22c55e); color: #f0fdf4; }
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 10px;
        margin: 10px 0 18px 0;
    }
    .signal-card {
        background: linear-gradient(180deg, #ffffff, #f3f7fb);
        color: #132333;
        border: 1px solid #d8e2ec;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.08);
    }
    .signal-card strong {
        display: block;
        margin-bottom: 4px;
        font-size: 1rem;
    }
    .signal-card p {
        margin: 0;
        color: #475569;
        font-size: 0.9rem;
    }
    .signal-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 8px;
        margin-bottom: 8px;
        font-size: 0.74rem;
        font-weight: 700;
        background: #dbeafe;
        color: #1e293b;
    }
    .topbar {
        background: linear-gradient(135deg, rgba(15,23,42,0.78), rgba(6,78,59,0.52));
        border: 1px solid rgba(148,163,184,0.18);
        border-left: 4px solid #22c55e;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 10px 0 18px 0;
        color: #f8fafc;
        display: grid;
        grid-template-columns: minmax(280px, 1fr) auto;
        gap: 18px;
        align-items: center;
        box-shadow: 0 14px 34px rgba(2, 6, 23, 0.22);
    }
    .topbar-title strong {
        display: block;
        font-size: 1.55rem;
        line-height: 1.25;
    }
    .topbar-title span {
        display: block;
        color: #dbeafe;
        font-size: 0.92rem;
        margin-top: 6px;
    }
    .topbar-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
    }
    .status-pill {
        background: rgba(15,23,42,0.48);
        border: 1px solid rgba(125,211,252,0.22);
        border-radius: 999px;
        padding: 7px 10px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        white-space: nowrap;
    }
    .status-pill span {
        color: #7dd3fc;
        font-size: 0.74rem;
        margin-right: 5px;
    }
    .status-pill strong {
        color: #ffffff;
        font-size: 0.86rem;
    }
    @media (max-width: 900px) {
        .topbar { grid-template-columns: 1fr; }
        .topbar-chips { justify-content: flex-start; }
    }
    .decision-panel {
        background: linear-gradient(180deg, #ffffff, #f5f7fb);
        border: 1px solid #dbe3ea;
        border-left: 5px solid #dc2626;
        border-radius: 10px;
        padding: 16px 18px;
        margin: 12px 0 18px 0;
        color: #132333;
    }
    .decision-panel h4 {
        margin: 0 0 10px 0;
        font-size: 1.05rem;
    }
    .decision-panel p {
        margin: 4px 0;
        color: #334155;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(15,23,42,0.12);
        background: #ffffff;
        color: #1e293b;
    }
    .chip-critical { background: #fee2e2; color: #991b1b; }
    .chip-high { background: #ffedd5; color: #9a3412; }
    .chip-watch { background: #fef9c3; color: #854d0e; }
    .chip-low { background: #dcfce7; color: #166534; }
    .spread-box {
        background: linear-gradient(135deg, #eff6ff, #e0f2fe);
        border: 1px solid #bfdbfe;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0 16px 0;
        color: #10233f;
    }
    .spread-box h4 {
        margin: 0 0 6px 0;
        font-size: 1.02rem;
    }
    .spread-box p {
        margin: 0;
        color: #263746;
    }
    .fire-console {
        display: grid;
        grid-template-columns: minmax(280px, 1.15fr) minmax(280px, 1fr);
        gap: 12px;
        margin: 12px 0 16px 0;
    }
    .fire-alert-card {
        border-radius: 12px;
        padding: 18px;
        color: #fff7ed;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 14px 32px rgba(0,0,0,0.18);
    }
    .fire-alert-card h3 {
        margin: 0 0 8px 0;
        font-size: 1.45rem;
    }
    .fire-alert-card p {
        margin: 4px 0;
        font-size: 0.95rem;
    }
    .evidence-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(130px, 1fr));
        gap: 10px;
    }
    .evidence-tile {
        background: linear-gradient(180deg, #ffffff, #f5f8fb);
        color: #10212f;
        border: 1px solid #dbe3ea;
        border-radius: 10px;
        padding: 13px 14px;
        min-height: 76px;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.08);
    }
    .evidence-tile span {
        display: block;
        color: #475569;
        font-size: 0.82rem;
        margin-bottom: 7px;
    }
    .evidence-tile strong {
        display: block;
        font-size: 1.12rem;
        line-height: 1.15;
    }
    .spread-tile {
        border-left: 5px solid #2563eb;
        background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    }
    .rule-panel {
        background: linear-gradient(180deg, #ffffff, #f5f8fb);
        border: 1px solid #dbe3ea;
        border-radius: 10px;
        padding: 14px;
        color: #10212f;
        box-shadow: 0 10px 22px rgba(2, 6, 23, 0.10);
    }
    .rule-verdict {
        border-radius: 8px;
        padding: 11px 12px;
        margin-bottom: 10px;
        font-weight: 800;
    }
    .rule-clear { background: #dcfce7; color: #166534; }
    .rule-watch { background: #fef9c3; color: #854d0e; }
    .rule-alert { background: #fee2e2; color: #991b1b; }
    .rule-list {
        display: grid;
        gap: 8px;
    }
    .rule-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 7px;
        font-size: 0.92rem;
    }
    .rule-row:last-child { border-bottom: 0; padding-bottom: 0; }
    .rule-row span { color: #475569; }
    .rule-row strong { color: #0f172a; }
    @media (max-width: 900px) {
        .fire-console { grid-template-columns: 1fr; }
    }
    .color-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 6px 0 14px 0;
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        display: inline-block;
        border-radius: 999px;
        margin-right: 6px;
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


def add_sensor_wind_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "wind_direction_deg" not in enriched.columns:
        enriched["wind_direction_deg"] = 235.0
    if "wind_direction" not in enriched.columns:
        enriched["wind_direction"] = enriched["wind_direction_deg"].apply(degrees_to_compass)
    if "spread_direction_deg" not in enriched.columns:
        enriched["spread_direction_deg"] = (enriched["wind_direction_deg"] + 180) % 360
    if "spread_direction" not in enriched.columns:
        enriched["spread_direction"] = enriched["spread_direction_deg"].apply(degrees_to_compass)
    return enriched


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


def germany_live_locations() -> list[dict]:
    return [location for location in GLOBAL_FIRE_LOCATIONS if location["country"] == "Germany"]


def germany_location_options() -> dict[str, dict]:
    return {
        f"{location['area']} · {location['site']}": location
        for location in germany_live_locations()
    }


@st.cache_data(ttl=1800)
def load_selected_location_weather(area: str, site: str) -> pd.DataFrame:
    location = next(
        item for item in GLOBAL_FIRE_LOCATIONS
        if item["country"] == "Germany" and item["area"] == area and item["site"] == site
    )
    return fetch_location_weather(location)


@st.cache_data(ttl=3600)
def load_selected_location_history(area: str, site: str, start_date, end_date) -> tuple[pd.DataFrame, str]:
    location = next(
        item for item in GLOBAL_FIRE_LOCATIONS
        if item["country"] == "Germany" and item["area"] == area and item["site"] == site
    )
    return fetch_location_history(location, start_date, end_date)


@st.cache_data(ttl=1800)
def load_global_fire_weather() -> pd.DataFrame:
    _ = GLOBAL_MONITOR_CACHE_VERSION
    return prepare_global_weather(fetch_global_fire_weather())


def prepare_global_weather(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "area" not in prepared.columns:
        prepared["area"] = prepared.get("site", prepared.get("country", "Monitored area"))
    prepared["area"] = prepared["area"].fillna(prepared.get("site", "Monitored area"))
    if "timezone" not in prepared.columns:
        prepared["timezone"] = "UTC"
    if "wind_direction" not in prepared.columns and "wind_direction_deg" in prepared.columns:
        prepared["wind_direction"] = prepared["wind_direction_deg"].apply(degrees_to_compass)
    if "spread_direction" not in prepared.columns and "wind_direction_deg" in prepared.columns:
        prepared["spread_direction"] = prepared["wind_direction_deg"].apply(lambda value: degrees_to_compass((float(value) + 180) % 360))
    return prepared


@st.cache_data(ttl=300)
def load_sensor_demo(seed: int) -> pd.DataFrame:
    return add_sensor_wind_columns(generate_sensor_demo(seed=seed))


def format_signal_time(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%d %b %Y, %H:%M")


def format_signal_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%d %b")


def format_signal_clock(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%H:%M")


def metric_cards(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        "<div class='metric-card'>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='metric-grid'>{cards}</div>", unsafe_allow_html=True)


def bi_kpi_cards(items: list[tuple[str, str, str, bool]]) -> None:
    cards = "".join(
        f"<div class='bi-kpi{' alert' if is_alert else ''}'>"
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<em>{escape(note)}</em>"
        "</div>"
        for label, value, note, is_alert in items
    )
    st.markdown(f"<div class='bi-kpi-grid'>{cards}</div>", unsafe_allow_html=True)


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
        "Watch": "#eab308",
        "Low": "#22c55e",
    }.get(prediction, "#22c55e")


def country_card_class(risk: str) -> str:
    return {
        "Critical": "critical",
        "High": "high",
        "Elevated": "elevated",
        "Low": "low",
    }.get(risk, "low")


def latest_by_country(global_weather: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (_, area, site), frame in global_weather.groupby(["country", "area", "site"]):
        timezone = str(frame["timezone"].iat[0]) if "timezone" in frame.columns else "UTC"
        now = pd.Timestamp.now(tz=ZoneInfo(timezone)).tz_localize(None)
        future = frame[frame["time"] >= now]
        latest = future.iloc[0] if not future.empty else frame.iloc[-1]
        worst = frame.sort_values(["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False).iloc[0]
        row = latest.copy()
        row["risk"] = risk_label(int(latest["fire_weather_score"]), bool(latest["fire_30_30_30"]))
        row["worst_risk"] = risk_label(int(worst["fire_weather_score"]), bool(worst["fire_30_30_30"]))
        row["worst_time"] = worst["time"]
        row["spread_direction"] = degrees_to_compass((float(latest["wind_direction_deg"]) + 180) % 360)
        rows.append(row)
    return pd.DataFrame(rows)


def global_country_cards(latest: pd.DataFrame) -> None:
    cards = []
    for row in latest.sort_values(["fire_weather_score", "temperature_c"], ascending=False).itertuples():
        risk = str(row.risk)
        temp_note = "Temp alert" if row.temperature_c > 22 else "Temp normal"
        cards.append(
            f"<div class='country-card {country_card_class(risk)}'>"
            f"<h4>{escape(row.country)}</h4>"
            f"<div class='site'>{escape(row.area)} · {escape(row.site)}</div>"
            f"<div class='risk'>{escape(risk)} risk</div>"
            "<dl>"
            f"<div><dt>Temp</dt><dd>{row.temperature_c:.1f} C</dd></div>"
            f"<div><dt>Humidity</dt><dd>{row.humidity_pct:.0f}%</dd></div>"
            f"<div><dt>Wind</dt><dd>{row.wind_kmh:.1f} km/h</dd></div>"
            f"<div><dt>Spread</dt><dd>{escape(row.wind_direction)} -> {escape(row.spread_direction)}</dd></div>"
            f"<div><dt>Alert</dt><dd>{escape(temp_note)}</dd></div>"
            f"<div><dt>Worst window</dt><dd>{escape(row.worst_risk)}</dd></div>"
            "</dl>"
            "</div>"
        )
    st.markdown(f"<div class='country-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def global_area_board(latest: pd.DataFrame, limit: int | None = None) -> None:
    tiles = []
    board = latest.sort_values(["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False)
    if limit is not None:
        board = board.head(limit)
    for row in board.itertuples():
        risk = str(row.risk)
        score = float(row.fire_weather_score)
        score_width = max(4.0, min(score, 100.0))
        tiles.append(
            f"<div class='area-tile {country_card_class(risk)}'>"
            f"<h4>{escape(row.area)}</h4>"
            f"<div class='area-meta'>{escape(row.country)} · {escape(row.site)}</div>"
            "<div class='score-row'>"
            f"<div class='score'>{score:.0f}</div>"
            f"<div class='badge'>{escape(risk)}</div>"
            "</div>"
            f"<div class='area-bar'><span style='width:{score_width:.0f}%'></span></div>"
            "<div class='area-facts'>"
            f"<div>Temp<strong>{row.temperature_c:.1f} C</strong></div>"
            f"<div>Wind<strong>{row.wind_kmh:.1f} km/h</strong></div>"
            f"<div>Spread<strong>{escape(row.spread_direction)}</strong></div>"
            "</div>"
            "</div>"
        )
    st.markdown(f"<div class='area-board'>{''.join(tiles)}</div>", unsafe_allow_html=True)


def effis_interactive_map(site: str, area: str, latitude: float, longitude: float) -> None:
    layer_url = effis_map_url("mf010.fwi", latitude=latitude, longitude=longitude)
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        body {{
          margin: 0;
          font-family: Inter, Segoe UI, Arial, sans-serif;
          background: #0b1118;
          color: #0f172a;
        }}
        .map-shell {{
          border: 1px solid #d7e1ea;
          border-radius: 14px;
          overflow: hidden;
          background: #ffffff;
          box-shadow: 0 18px 36px rgba(2,6,23,0.20);
        }}
        .map-head {{
          display: flex;
          justify-content: space-between;
          gap: 16px;
          align-items: center;
          padding: 14px 16px;
          background: linear-gradient(135deg, #f8fafc, #eaf3ee);
          border-bottom: 1px solid #dbe3ea;
        }}
        .map-head strong {{
          display: block;
          font-size: 16px;
          margin-bottom: 3px;
        }}
        .map-head span {{
          color: #526678;
          font-size: 13px;
        }}
        .map-head a {{
          color: #0f766e;
          border: 1px solid #99f6e4;
          background: #ecfdf5;
          border-radius: 999px;
          padding: 7px 10px;
          text-decoration: none;
          font-weight: 700;
          font-size: 12px;
          white-space: nowrap;
        }}
        #map {{
          height: 430px;
          width: 100%;
        }}
        .legend {{
          position: absolute;
          z-index: 500;
          bottom: 18px;
          right: 14px;
          background: rgba(255,255,255,0.94);
          border: 1px solid #cbd5e1;
          border-radius: 10px;
          padding: 10px 12px;
          box-shadow: 0 10px 24px rgba(15,23,42,0.18);
          font-size: 12px;
          line-height: 1.35;
        }}
        .legend strong {{
          display: block;
          margin-bottom: 6px;
          font-size: 12px;
        }}
        .legend-row {{
          display: flex;
          align-items: center;
          gap: 7px;
          margin: 4px 0;
          color: #334155;
        }}
        .swatch {{
          width: 14px;
          height: 10px;
          border-radius: 3px;
          display: inline-block;
        }}
      </style>
    </head>
    <body>
      <div class="map-shell">
        <div class="map-head">
          <div>
            <strong>{escape(site)}</strong>
            <span>{escape(area)} - roads, terrain, selected point, and fire-weather intensity overlay</span>
          </div>
          <a href="{escape(layer_url)}" target="_blank" rel="noopener noreferrer">Open source layer</a>
        </div>
        <div id="map"></div>
      </div>
      <script>
        const map = L.map('map', {{ zoomControl: true }}).setView([{latitude:.5f}, {longitude:.5f}], 9);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          maxZoom: 18,
          attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        L.tileLayer.wms('https://maps.effis.emergency.copernicus.eu/effis', {{
          layers: 'mf010.fwi',
          format: 'image/png',
          transparent: true,
          opacity: 0.58,
          version: '1.1.1'
        }}).addTo(map);

        L.marker([{latitude:.5f}, {longitude:.5f}]).addTo(map)
          .bindPopup('{escape(site)}<br>{escape(area)}')
          .openPopup();

        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function() {{
          const div = L.DomUtil.create('div', 'legend');
          div.innerHTML = `
            <strong>Fire-weather intensity</strong>
            <div class="legend-row"><span class="swatch" style="background:#9af7c2"></span>Low</div>
            <div class="legend-row"><span class="swatch" style="background:#d6ea42"></span>Moderate</div>
            <div class="legend-row"><span class="swatch" style="background:#f59e0b"></span>High</div>
            <div class="legend-row"><span class="swatch" style="background:#dc2626"></span>Very high</div>
          `;
          return div;
        }};
        legend.addTo(map);
      </script>
    </body>
    </html>
    """
    components.html(html, height=520, scrolling=False)


def risk_banner(sensor_id: str, zone: str, prediction: str, probability: float, signal_time: object) -> None:
    signal_label = format_signal_time(signal_time)
    message = (
        f"{sensor_id} · {zone} is the strongest current signal in the sensor network."
    )
    st.markdown(
        "<div class='risk-banner "
        f"{risk_class(prediction)}'>"
        f"<h3>{escape(prediction)} · {probability:.0f}% fire probability</h3>"
        f"<p>{escape(message)}</p>"
        f"<p><strong>Estimated fire-risk time:</strong> {escape(signal_label)}</p>"
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
        "<defs><marker id='arrowhead' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='#2563eb'/></marker></defs>",
        "<rect width='100%' height='100%' rx='8' fill='#f8fafc'/>",
        "<rect x='35' y='35' width='830' height='290' rx='8' fill='none' stroke='#94a3b8' stroke-dasharray='10 8'/>",
        "<text x='50' y='62' fill='#334155' font-size='14' font-weight='700'>Field sensor grid</text>",
    ]
    top_sensor = latest.sort_values("fire_probability_pct", ascending=False).iloc[0]
    for row in latest.itertuples():
        x = pad + ((row.lon - xmin) / max(xmax - xmin, 1e-9)) * (width - pad * 2)
        y = pad + (1 - ((row.lat - ymin) / max(ymax - ymin, 1e-9))) * (height - pad * 2)
        radius = 10 + (row.fire_probability_pct / 100) * 18
        color = risk_color(row.prediction)
        if row.sensor_id == top_sensor["sensor_id"] and row.fire_probability_pct > 0:
            angle = np.deg2rad(row.spread_direction_deg)
            arrow_length = 72
            x2 = x + np.sin(angle) * arrow_length
            y2 = y - np.cos(angle) * arrow_length
            parts.append(
                f"<line x1='{x:.1f}' y1='{y:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#2563eb' stroke-width='4' marker-end='url(#arrowhead)' opacity='0.9'/>"
            )
            parts.append(
                f"<text x='{x2 + 8:.1f}' y='{y2:.1f}' fill='#1d4ed8' font-size='12' font-weight='700'>Spread toward {escape(row.spread_direction)}</text>"
            )
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
        ("Early signal", "Smoke", "Smoke rising near one sensor is the first warning sign."),
        ("Confirmation", "CO", "CO helps confirm that the signal may be combustion, not only dust or fog."),
        ("Heat stress", "Temperature", "Higher temperature makes dry vegetation easier to ignite."),
        ("Dry fuel", "Humidity", "Low humidity means leaves and ground fuel dry faster."),
        ("Spread", "Wind", "Wind direction shows where flames or smoke may move next."),
        ("Hot material", "Infrared", "IR can indicate hot ground, flame, or heated material near the sensor."),
    ]
    cards = "".join(
        "<div class='signal-card'>"
        f"<span class='signal-badge'>{escape(badge)}</span>"
        f"<strong>{escape(title)}</strong>"
        f"<p>{escape(body)}</p>"
        "</div>"
        for badge, title, body in items
    )
    st.markdown(f"<div class='signal-grid'>{cards}</div>", unsafe_allow_html=True)


def hero_section() -> None:
    st.markdown(
        """
        <div class='hero'>
          <div class='hero-kicker'>CTRL-F · Fire Intelligence</div>
          <div class='hero-title'>Field fire-risk operations view.</div>
          <div class='hero-subtitle'>
            Sensor signals, live weather, wind direction, and historical fire-weather context in one operational view.
          </div>
          <div class='status-row'>
            <div class='status-pill'><span>Coverage</span><strong>Multi-region</strong></div>
            <div class='status-pill'><span>Mode</span><strong>Sensor + Weather</strong></div>
            <div class='status-pill'><span>Prediction</span><strong>Fire probability</strong></div>
            <div class='status-pill'><span>Data Inputs</span><strong>Weather + Sensors</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_header() -> None:
    st.markdown(
        """
        <div class='topbar'>
          <div class='topbar-title'>
            <strong>CTRL-F FireWatch</strong>
            <span>Operational view for sensor signals, weather, and wind-driven spread.</span>
          </div>
          <div class='topbar-chips'>
            <div class='status-pill'><span>Coverage</span><strong>Germany · USA · Canada</strong></div>
            <div class='status-pill'><span>Signals</span><strong>Sensors + Weather</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_panel(sensor_id: str, zone: str, prediction: str, probability: float, signal_time: object) -> None:
    status = "Fire-risk signal detected" if prediction in {"Critical", "High"} else "No strong fire-risk signal"
    priority = "Critical" if prediction == "Critical" else ("High" if prediction == "High" else "Normal")
    chip_class = {
        "Critical": "chip-critical",
        "High": "chip-high",
        "Watch": "chip-watch",
        "Low": "chip-low",
    }.get(prediction, "chip-low")
    st.markdown(
        "<div class='decision-panel'>"
        "<h4>Risk assessment</h4>"
        f"<p>{escape(sensor_id)} at {escape(zone)} is the lead signal. The current model combines sensor readings with weather context.</p>"
        "<div class='chip-row'>"
        f"<span class='chip {chip_class}'>{escape(priority)} risk</span>"
        f"<span class='chip'>{probability:.0f}% probability</span>"
        f"<span class='chip'>Estimated time: {escape(format_signal_time(signal_time))}</span>"
        f"<span class='chip'>{escape(status)}</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def wind_spread_panel(row: pd.Series) -> None:
    wind_direction = str(row.get("wind_direction", "SW"))
    spread_direction = str(row.get("spread_direction", "NE"))
    wind_kmh = float(row.get("wind_kmh", 0.0))
    st.markdown(
        "<div class='spread-box'>"
        "<h4>Wind impact</h4>"
        f"<p>Wind is coming from <strong>{escape(wind_direction)}</strong> "
        f"at <strong>{wind_kmh:.1f} km/h</strong>. "
        f"If ignition starts, likely spread is toward <strong>{escape(spread_direction)}</strong>.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def fire_summary_panel(
    row: pd.Series,
    probability: float,
    estimated_fire_time: object,
    spread_direction: str,
    wind_kmh: float,
) -> None:
    prediction = str(row.get("prediction", "Low"))
    sensor_id = str(row.get("sensor_id", "Sensor"))
    zone = str(row.get("zone", "Field zone"))
    wind_direction = str(row.get("wind_direction", "SW"))
    smoke_ppm = float(row.get("smoke_ppm", 0.0))
    card_class = risk_class(prediction)
    status = "Fire-risk signal detected" if prediction in {"Critical", "High"} else "Monitoring"
    st.markdown(
        "<div class='fire-console'>"
        f"<div class='fire-alert-card {card_class}'>"
        f"<h3>{escape(prediction)} risk - {probability:.0f}% probability</h3>"
        f"<p><strong>{escape(sensor_id)}</strong> at <strong>{escape(zone)}</strong></p>"
        f"<p>Estimated fire-risk time: <strong>{escape(format_signal_time(estimated_fire_time))}</strong></p>"
        f"<p>{escape(status)}</p>"
        "</div>"
        "<div class='evidence-grid'>"
        f"<div class='evidence-tile'><span>Sensor</span><strong>{escape(sensor_id)}</strong></div>"
        f"<div class='evidence-tile'><span>Smoke</span><strong>{smoke_ppm:.1f} ppm</strong></div>"
        f"<div class='evidence-tile'><span>Wind from</span><strong>{escape(wind_direction)} at {wind_kmh:.1f} km/h</strong></div>"
        f"<div class='evidence-tile spread-tile'><span>Likely spread</span><strong>{escape(wind_direction)} -> {escape(spread_direction)}</strong></div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def rule_check_tiles(weather: pd.DataFrame, latest: pd.Series) -> None:
    hot_hours = int((weather["temperature_c"] >= 30).sum())
    dry_hours = int((weather["humidity_pct"] <= 30).sum())
    windy_hours = int((weather["wind_kmh"] >= 30).sum())
    rule_hours = int(weather["fire_30_30_30"].sum())
    near_hours = int((weather["fire_weather_score"] >= 2).sum())
    if rule_hours:
        verdict = f"30-30-30 window forecast: {rule_hours} hours"
        verdict_class = "rule-alert"
    elif near_hours:
        verdict = f"Near-risk weather: {near_hours} hours"
        verdict_class = "rule-watch"
    else:
        verdict = "No 30-30-30 window forecast"
        verdict_class = "rule-clear"
    rows = [
        ("Temperature", f"{float(latest['temperature_c']):.1f} C now", "needs >= 30 C", hot_hours),
        ("Humidity", f"{float(latest['humidity_pct']):.0f}% now", "needs <= 30%", dry_hours),
        ("Wind", f"{float(latest['wind_kmh']):.1f} km/h now", "needs >= 30 km/h", windy_hours),
        ("Full rule", f"{rule_hours} forecast hours", "all three together", rule_hours),
    ]
    row_html = "".join(
        f"<div class='rule-row'><span>{escape(label)}<br>{escape(threshold)}</span><strong>{escape(value)}</strong></div>"
        for label, value, threshold, hours in rows
    )
    st.markdown(
        "<div class='rule-panel'>"
        f"<div class='rule-verdict {verdict_class}'>{escape(verdict)}</div>"
        f"<div class='rule-list'>{row_html}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def risk_legend() -> None:
    items = [
        ("Low", "#22c55e"),
        ("Watch", "#eab308"),
        ("High", "#f97316"),
        ("Critical", "#dc2626"),
    ]
    html = "".join(
        f"<span><span class='legend-dot' style='background:{color}'></span>{label}</span>"
        for label, color in items
    )
    st.markdown(f"<div class='color-legend'>{html}</div>", unsafe_allow_html=True)


def round_numeric(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_cols = rounded.select_dtypes(include="number").columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)
    return rounded


def insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'><strong>Insight:</strong> {text}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note-box'><strong>Note:</strong> {text}</div>", unsafe_allow_html=True)


def temperature_alert(temp_c: float, source: str, threshold: float = 22.0) -> None:
    if temp_c > threshold:
        st.markdown(
            "<div class='temp-alert'>"
            "<h4>Temperature alert</h4>"
            f"<p>{escape(source)} is at <strong>{temp_c:.1f} C</strong>, above the configured alert threshold of "
            f"<strong>{threshold:.0f} C</strong>. Heat stress is increasing.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='temp-ok'>"
            "<h4>Temperature normal</h4>"
            f"<p>{escape(source)} is at <strong>{temp_c:.1f} C</strong>, below the configured alert threshold.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


def product_demo_page() -> None:
    st.markdown(
        """
        <div class='product-hero'>
          <div class='product-hero-grid'>
            <div>
              <div class='product-kicker'>CTRL-F Fire Prediction Alerts</div>
              <h2>Detect risk early. Alert field teams. Coordinate response.</h2>
              <p>
                A product demo for wildfire operations: sensor signals, live weather, wind direction,
                regional fire-weather maps, and historical review in one command-ready workspace.
              </p>
              <div class='product-stats'>
                <div class='product-stat'><span>Inputs</span><strong>Sensors + weather</strong></div>
                <div class='product-stat'><span>Output</span><strong>Risk + response brief</strong></div>
                <div class='product-stat'><span>Coverage</span><strong>Germany · USA · Canada</strong></div>
                <div class='product-stat'><span>Use</span><strong>Before · During · After</strong></div>
              </div>
            </div>
            <div class='demo-console'>
              <div class='console-top'>
                <div class='console-title'><strong>Live operations snapshot</strong><span>Selected monitoring area</span></div>
                <div class='console-badge'>Alert ready</div>
              </div>
              <div class='console-map'>
                <div class='console-road road-a'></div>
                <div class='console-road road-b'></div>
                <div class='console-road road-c'></div>
                <div class='hotspot'></div>
                <div class='wind-arrow'></div>
              </div>
              <div class='console-grid'>
                <div class='console-metric'><span>Risk</span><strong>High</strong></div>
                <div class='console-metric'><span>Wind</span><strong>SW -> NE</strong></div>
                <div class='console-metric'><span>Action</span><strong>Brief sent</strong></div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Operational Workflow")
    phases = [
        ("Before", "Risk readiness", "Compare monitored areas, review heat/wind/humidity, and identify locations needing extra attention."),
        ("During", "Alert and response", "Use sensor probability, wind direction, and fire-weather intensity to decide where teams should look first."),
        ("After", "Incident review", "Review historical weather windows and export a concise brief for reporting or stakeholder updates."),
    ]
    phase_html = "".join(
        "<div class='phase-card'>"
        f"<span>{escape(phase)}</span>"
        f"<h4>{escape(title)}</h4>"
        f"<p>{escape(body)}</p>"
        "</div>"
        for phase, title, body in phases
    )
    st.markdown(f"<div class='phase-grid'>{phase_html}</div>", unsafe_allow_html=True)

    st.markdown("#### Live Demo Flow")
    steps = [
        ("1", "Predict", "Open Fire Prediction to see how sensor readings become a fire-probability signal."),
        ("2", "Verify", "Open Live Weather to check temperature, humidity, wind, 30-30-30 rules, and EFFIS map context."),
        ("3", "Prioritize", "Open Regional Monitor to compare monitored areas and focus the response queue."),
        ("4", "Report", "Open Historical Risk to review an incident window and download a summary brief."),
    ]
    step_html = "".join(
        "<div class='demo-step'>"
        f"<span>{escape(number)}</span>"
        f"<h4>{escape(title)}</h4>"
        f"<p>{escape(body)}</p>"
        "</div>"
        for number, title, body in steps
    )
    st.markdown(f"<div class='demo-step-grid'>{step_html}</div>", unsafe_allow_html=True)

    st.markdown("#### Platform Capabilities")
    features = [
        ("Live weather alerts", "Select Germany regions and track current temperature, humidity, rain, wind, and spread direction."),
        ("Regional monitor", "Compare Germany, USA, and Canada monitoring areas with a prioritized risk queue."),
        ("Sensor prediction", "Turn smoke, CO, heat, humidity, wind, and IR signals into a fire-probability score."),
        ("EFFIS map context", "View Copernicus fire-weather intensity on top of a street map with a selected-area marker."),
        ("Historical risk review", "Reconstruct incident windows for multiple Germany regions and export a review brief."),
        ("Brief exports", "Download alert and historical summaries for field communication or client demos."),
    ]
    feature_html = "".join(
        "<div class='feature-card'>"
        f"<span>Capability</span>"
        f"<h4>{escape(title)}</h4>"
        f"<p>{escape(body)}</p>"
        "</div>"
        for title, body in features
    )
    st.markdown(f"<div class='feature-grid'>{feature_html}</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='demo-cta'><strong>Client takeaway:</strong> CTRL-F is not just a weather dashboard. It is a response workflow: detect the signal, verify conditions, prioritize regions, and export a clear field brief.</div>",
        unsafe_allow_html=True,
    )


def operational_actions(risk: str, temp_c: float, wind_kmh: float, wind_direction: str, spread_direction: str) -> list[str]:
    actions = []
    if risk in {"Critical", "High"}:
        actions.append("Escalate to field lead and keep the area under active watch.")
        actions.append(f"Position observers downwind. Wind is from {wind_direction}, likely spread is toward {spread_direction}.")
    elif risk == "Elevated":
        actions.append("Keep the area on watch and review the next forecast window.")
    else:
        actions.append("Continue routine monitoring; no strong fire-weather signal right now.")
    if temp_c > 22:
        actions.append(f"Temperature is above the configured alert threshold at {temp_c:.1f} C.")
    if wind_kmh >= 20:
        actions.append(f"Wind is notable at {wind_kmh:.1f} km/h; spread direction should be reviewed.")
    return actions[:4]


def action_panel(title: str, risk: str, actions: list[str]) -> None:
    panel_class = country_card_class(risk)
    items = "".join(f"<li>{escape(action)}</li>" for action in actions)
    st.markdown(
        f"<div class='action-panel {panel_class}'>"
        f"<h4>{escape(title)}</h4>"
        f"<ul class='action-list'>{items}</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def incident_brief_text(
    location: dict,
    latest: pd.Series,
    latest_risk: str,
    worst: pd.Series,
    worst_risk: str,
    actions: list[str],
) -> str:
    action_lines = "\n".join(f"- {action}" for action in actions)
    return (
        "CTRL-F Fire Prediction Alert Brief\n"
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}\n\n"
        f"Area: {location['area']}\n"
        f"Site: {location['site']}\n"
        f"Current risk: {latest_risk}\n"
        f"Temperature: {float(latest['temperature_c']):.1f} C\n"
        f"Humidity: {float(latest['humidity_pct']):.0f}%\n"
        f"Wind: {float(latest['wind_kmh']):.1f} km/h from {latest['wind_direction']}\n"
        f"Worst forecast window: {worst_risk} on {worst['time']:%d %b %Y, %H:%M}\n\n"
        "Recommended actions:\n"
        f"{action_lines}\n"
    )


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

compact_header()

product_tab, overview_tab, sensor_demo_tab, live_nrw_tab, global_tab, historical_nrw_tab, weather_tab = st.tabs(
    ["Product Demo", "Overview", "Fire Prediction", "Live Weather", "Regional Monitor", "Historical Risk", "Weather Analysis"]
)

with product_tab:
    product_demo_page()

with overview_tab:
    st.subheader("Mission View")
    metric_cards(
        [
            ("System focus", "Sensors + weather"),
            ("Coverage", "Multi-region"),
            ("Core signal", "Fire probability"),
            ("Data inputs", "Weather + sensors"),
        ]
    )
    insight(
        "The current view combines ground-sensor signals, live weather, wind direction, and regional fire-weather context."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### What CTRL-F Detects")
        st.write(
            "Smoke, CO, heat, humidity, wind, and IR signals are combined into one fire-probability score."
        )
        signal_cards()
    with right:
        st.markdown("#### Data Flow")
        flow = pd.DataFrame(
            [
                ["1", "Sensor board", "Reads smoke, CO, IR, temperature, humidity"],
                ["2", "Data tunnel", "LoRa/WiFi sends readings to server"],
                ["3", "Prediction", "Risk score detects early fire signal"],
                ["4", "Dashboard", "Shows prediction and supporting evidence"],
            ],
            columns=["Step", "Layer", "Role"],
        )
        html_table(flow)

    note("Fire Prediction currently uses sample sensor readings. Live Weather uses weather API calls and Copernicus EFFIS map layers.")

with live_nrw_tab:
    st.subheader("Live Weather Alert")
    st.write("Current forecast conditions and Copernicus EFFIS fire-weather layer for selected Germany regions.")
    st.caption("Weather feed: Open-Meteo forecast. Fire danger map: Copernicus EFFIS WMS.")

    try:
        st.markdown(
            "<div class='control-panel'><strong>Live region selection</strong><span>Choose a Germany monitoring area to update forecast, wind, rule checks, and EFFIS map.</span></div>",
            unsafe_allow_html=True,
        )
        location_labels = germany_location_options()
        selected_location_label = st.selectbox(
            "Germany region",
            list(location_labels.keys()),
            key="live_germany_region",
        )
        selected_location = location_labels[selected_location_label]
        live_weather = load_selected_location_weather(selected_location["area"], selected_location["site"])
        now_local = pd.Timestamp.now(tz=selected_location["timezone"]).tz_localize(None)
        future_weather = live_weather[live_weather["time"] >= now_local]
        latest = future_weather.iloc[0] if not future_weather.empty else live_weather.iloc[-1]
        worst = live_weather.sort_values(["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False).iloc[0]
        latest_risk = risk_label(int(latest["fire_weather_score"]), bool(latest["fire_30_30_30"]))
        worst_risk = risk_label(int(worst["fire_weather_score"]), bool(worst["fire_30_30_30"]))

        metric_cards(
            [
                ("Region", selected_location["area"]),
                ("Site", selected_location["site"]),
                ("Now", latest_risk),
                ("Temp", f"{latest['temperature_c']:.1f} C"),
                ("Humidity", f"{latest['humidity_pct']:.0f}%"),
                ("Wind", f"{latest['wind_kmh']:.1f} km/h"),
                ("Direction", f"{latest['wind_direction']} ({latest['wind_direction_deg']:.0f} deg)"),
            ]
        )
        temperature_alert(float(latest["temperature_c"]), f"{selected_location['area']} weather")

        insight(
            f"Highest forecast risk in the next 3 days: {worst_risk} on {worst['time']:%d %b, %H:%M}."
        )
        spread_direction = degrees_to_compass((float(latest["wind_direction_deg"]) + 180) % 360)
        actions = operational_actions(
            latest_risk,
            float(latest["temperature_c"]),
            float(latest["wind_kmh"]),
            str(latest["wind_direction"]),
            spread_direction,
        )
        action_panel("Recommended response", latest_risk, actions)
        st.download_button(
            "Download alert brief",
            data=incident_brief_text(selected_location, latest, latest_risk, worst, worst_risk, actions),
            file_name=f"ctrlf_alert_brief_{selected_location['area'].lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Forecast Inputs")
            st.caption(f"Temperature, humidity, wind, and rain for {selected_location['site']}.")
            svg_line_chart(
                live_weather.set_index("time")[["temperature_c", "humidity_pct", "wind_kmh", "rain_mm"]],
                height=360,
            )
        with right:
            st.markdown("#### 30-30-30 Checks")
            st.caption("Current values compared with the fire-weather rule.")
            rule_check_tiles(live_weather, latest)

        st.markdown("#### Wind Direction")
        st.caption("Wind direction matters for possible fire spread.")
        svg_bar_chart(direction_counts(live_weather, "wind_direction"), height=260)

        st.markdown("#### Copernicus EFFIS Fire Weather Index")
        st.caption("Street map context with the Copernicus EFFIS Fire Weather Index overlay.")
        effis_interactive_map(
            selected_location["site"],
            selected_location["area"],
            float(selected_location["latitude"]),
            float(selected_location["longitude"]),
        )

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
        effis_interactive_map("Huertgenwald, NRW", "North Rhine-Westphalia", 50.716, 6.375)

with global_tab:
    st.subheader("Regional Fire-Weather Monitor")
    st.write("Operational comparison across monitored areas in Germany, USA, and Canada.")

    try:
        global_weather = load_global_fire_weather()
        latest_global = latest_by_country(global_weather)

        st.markdown(
            "<div class='bi-panel'>"
            "<div class='bi-title'>Filters</div>"
            "<div class='bi-subtitle'>Focus the view by country, state or province, and monitoring location.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        filter_left, filter_right, filter_third, filter_fourth = st.columns([1, 1, 1.15, 0.85])
        with filter_left:
            country_options = ["All countries"] + sorted(latest_global["country"].unique().tolist())
            selected_country = st.selectbox("Country", country_options, key="global_country_filter")

        country_filtered = latest_global.copy()
        weather_filtered = global_weather.copy()
        if selected_country != "All countries":
            country_filtered = country_filtered[country_filtered["country"] == selected_country]
            weather_filtered = weather_filtered[weather_filtered["country"] == selected_country]

        with filter_right:
            area_options = ["All areas"] + sorted(country_filtered["area"].unique().tolist())
            selected_area = st.selectbox("State / province / area", area_options, key="global_area_filter")

        filtered_latest = country_filtered.copy()
        if selected_area != "All areas":
            filtered_latest = filtered_latest[filtered_latest["area"] == selected_area]
            weather_filtered = weather_filtered[weather_filtered["area"] == selected_area]

        with filter_third:
            detail_options = (
                filtered_latest.assign(label=lambda d: d["country"] + " · " + d["area"] + " · " + d["site"])
                .sort_values(["country", "area", "site"])["label"]
                .tolist()
            )
            selected_detail = st.selectbox("Trend detail", detail_options, key="global_area_detail")

        with filter_fourth:
            card_limit_label = st.selectbox("Area board", ["Top 8 areas", "Top 12 areas", "All areas"], key="global_card_limit")
            card_limit = {"Top 8 areas": 8, "Top 12 areas": 12, "All areas": None}[card_limit_label]

        high_count = int(filtered_latest["risk"].isin(["High", "Critical"]).sum())
        temp_alerts = int((filtered_latest["temperature_c"] > 22).sum())
        windiest = filtered_latest.sort_values("wind_kmh", ascending=False).iloc[0]
        hottest = filtered_latest.sort_values("temperature_c", ascending=False).iloc[0]
        highest_risk = filtered_latest.sort_values(["fire_weather_score", "temperature_c"], ascending=False).iloc[0]

        bi_kpi_cards(
            [
                ("Highest risk", f"{highest_risk['risk']}", f"{highest_risk['area']} · {highest_risk['fire_weather_score']:.0f} score", highest_risk["risk"] in ["High", "Critical"]),
                ("Monitored areas", f"{len(filtered_latest)}", f"{filtered_latest['country'].nunique()} countries in view", False),
                ("Temp alerts > 22 C", f"{temp_alerts}", f"Hottest: {hottest['area']} {hottest['temperature_c']:.1f} C", temp_alerts > 0),
                ("High/Critical now", f"{high_count}", "Current forecast snapshot", high_count > 0),
                ("Windiest area", f"{windiest['wind_kmh']:.1f} km/h", f"{windiest['area']} · from {windiest['wind_direction']}", False),
            ]
        )
        if temp_alerts:
            st.markdown(
                "<div class='temp-alert'>"
                "<h4>Temperature alert active</h4>"
                f"<p>{temp_alerts} monitored area(s) are above 22 C under the current alert configuration.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='temp-ok'><h4>No temperature alert</h4><p>All monitored areas are at or below 22 C right now.</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Area Risk Board")
        st.caption("Cards are sorted by highest fire-weather score first.")
        global_area_board(filtered_latest, limit=card_limit)
        insight("Use the filters to focus the operational view. The board highlights priority areas, while the matrix keeps the full selected dataset.")

        st.markdown("#### Priority Queue")
        priority_queue = (
            filtered_latest.sort_values(["fire_weather_score", "temperature_c", "wind_kmh"], ascending=False)
            .head(6)[
                [
                    "country",
                    "area",
                    "site",
                    "risk",
                    "temperature_c",
                    "humidity_pct",
                    "wind_kmh",
                    "wind_direction",
                    "spread_direction",
                ]
            ]
            .rename(
                columns={
                    "country": "Country",
                    "area": "Area",
                    "site": "Site",
                    "risk": "Risk",
                    "temperature_c": "Temp (C)",
                    "humidity_pct": "Humidity (%)",
                    "wind_kmh": "Wind (km/h)",
                    "wind_direction": "Wind from",
                    "spread_direction": "Spread toward",
                }
            )
        )
        html_table(round_numeric(priority_queue), use_container_width=True, hide_index=True)

        detail_country, detail_area, selected_site = [part.strip() for part in selected_detail.split("·", 2)]
        country_weather = global_weather[
            (global_weather["country"] == detail_country)
            & (global_weather["area"] == detail_area)
            & (global_weather["site"] == selected_site)
        ]
        left_panel, right_panel = st.columns([1.35, 1])
        with left_panel:
            st.markdown(f"#### Trend: {detail_area}")
            st.caption(f"{detail_country} · {selected_site}")
            svg_line_chart(
                country_weather.set_index("time")[["temperature_c", "humidity_pct", "wind_kmh", "rain_mm"]],
                height=300,
            )
        with right_panel:
            st.markdown("#### Portfolio Summary")
            country_summary = (
                filtered_latest.groupby("country", as_index=False)
                .agg(
                    Areas=("area", "nunique"),
                    Alerts=("temperature_c", lambda s: int((s > 22).sum())),
                    Avg_Temp_C=("temperature_c", "mean"),
                    Avg_Wind_kmh=("wind_kmh", "mean"),
                    Max_Score=("fire_weather_score", "max"),
                )
                .rename(columns={"country": "Country", "Avg_Temp_C": "Avg temp (C)", "Avg_Wind_kmh": "Avg wind (km/h)", "Max_Score": "Max score"})
            )
            html_table(round_numeric(country_summary), use_container_width=True, hide_index=True)
            st.markdown("#### Coverage")
            coverage = (
                latest_global.groupby("country", as_index=False)
                .agg(Areas=("area", "nunique"))
                .sort_values("Areas", ascending=False)
                .rename(columns={"country": "Country"})
            )
            html_table(coverage, use_container_width=True, hide_index=True)

        with st.expander("Detailed Area Matrix", expanded=True):
            comparison = filtered_latest[
                [
                    "country",
                    "area",
                    "site",
                    "risk",
                    "temperature_c",
                    "humidity_pct",
                    "wind_kmh",
                    "wind_direction",
                    "spread_direction",
                    "worst_risk",
                    "worst_time",
                ]
            ].rename(
                columns={
                    "country": "Country",
                    "area": "State / province / area",
                    "site": "Site",
                    "risk": "Risk now",
                    "temperature_c": "Temp (C)",
                    "humidity_pct": "Humidity (%)",
                    "wind_kmh": "Wind (km/h)",
                    "wind_direction": "Wind from",
                    "spread_direction": "Spread toward",
                    "worst_risk": "Worst 3-day risk",
                    "worst_time": "Worst time",
                }
            )
            html_table(round_numeric(comparison), use_container_width=True, hide_index=True)
        note("Current coverage uses selected monitoring locations. National-scale deployment should use scheduled gridded weather and fire-weather processing.")
    except Exception as exc:
        st.error("Global monitor data is temporarily unavailable.")
        st.caption(str(exc))

with sensor_demo_tab:
    st.subheader("Fire Prediction")
    st.write("Sensor and weather signals for early fire-risk detection.")

    with st.expander("Scenario settings"):
        note("Sample sensor feed for presentation purposes. The same view can connect to live hardware readings.")
        scenario = st.selectbox("Sensor feed", ["Fixed sample", "Refresh sample"], key="sensor_scenario")
    seed = 42 if scenario == "Fixed sample" else int(pd.Timestamp.now().timestamp()) % 100000
    sensor_data = add_sensor_wind_columns(load_sensor_demo(seed))
    latest = sensor_data.sort_values("time").groupby("sensor_id", as_index=False).tail(1)
    highest = latest.sort_values("fire_probability_pct", ascending=False).iloc[0]
    hotspot_rows = sensor_data[sensor_data["sensor_id"] == highest["sensor_id"]].copy()
    critical_rows = hotspot_rows[hotspot_rows["prediction"] == "Critical"]
    first_critical = None if critical_rows.empty else critical_rows.sort_values("time").iloc[0]
    estimated_fire_time = highest["time"] if first_critical is None else first_critical["time"]
    highest_wind_kmh = float(highest.get("wind_kmh", 0.0))
    highest_spread_direction = str(highest.get("spread_direction", "NE"))

    fire_summary_panel(
        highest,
        float(highest["fire_probability_pct"]),
        estimated_fire_time,
        highest_spread_direction,
        highest_wind_kmh,
    )
    temperature_alert(float(highest["temperature_c"]), f"{highest['sensor_id']} sensor")

    insight(
        "Why this matters: the sensor flags the hotspot, and wind direction shows where spread may move next."
    )

    st.markdown("#### Sensor Network")
    st.caption("Risk level by sensor location.")
    risk_legend()
    sensor_network_view(latest)

    with st.expander("Latest sensor readings"):
        st.markdown("#### Latest Readings")
        latest_table = latest.sort_values("fire_probability_pct", ascending=False)[
            [
                "time",
                "sensor_id",
                "zone",
                "prediction",
                "fire_probability_pct",
                "temperature_c",
                "humidity_pct",
                "wind_kmh",
                "wind_direction",
                "spread_direction",
                "smoke_ppm",
                "co_ppm",
                "battery_pct",
            ]
        ].rename(
            columns={
                "time": "Date",
                "sensor_id": "Sensor",
                "zone": "Zone",
                "prediction": "Prediction",
                "fire_probability_pct": "Fire probability (%)",
                "temperature_c": "Temp (C)",
                "humidity_pct": "Humidity (%)",
                "wind_kmh": "Wind (km/h)",
                "wind_direction": "Wind from",
                "spread_direction": "Spread toward",
                "smoke_ppm": "Smoke (ppm)",
                "co_ppm": "CO (ppm)",
                "battery_pct": "Battery (%)",
            }
        )
        latest_table.insert(1, "Time", latest_table["Date"].map(format_signal_clock))
        latest_table["Date"] = latest_table["Date"].map(format_signal_date)
        html_table(round_numeric(latest_table))

    with st.expander("Hotspot timeline"):
        st.markdown("#### Hotspot Timeline")
        if first_critical is None:
            st.caption("The hotspot escalates as smoke, CO, and IR rise together.")
        else:
            st.caption(
                f"Estimated fire-risk time: {format_signal_time(first_critical['time'])}. This is when the sensor first reaches critical risk in the sample feed."
            )
        hotspot_history = hotspot_rows.set_index("time")
        svg_line_chart(
            hotspot_history[["fire_probability_pct", "temperature_c", "humidity_pct", "smoke_ppm", "co_ppm"]],
            height=320,
        )

    with st.expander("Prediction signals"):
        st.markdown("#### Prediction Signals")
        signal_cards()

with historical_nrw_tab:
    st.subheader("Historical Risk Review")
    st.write("Review past weather conditions around an incident window for selected Germany regions.")
    st.caption("Use this to understand whether heat, low humidity, wind, and rain patterns supported fire spread.")

    default_end = pd.Timestamp.today().date()
    default_start = default_end - pd.Timedelta(days=14)
    st.markdown(
        "<div class='control-panel'><strong>Incident review controls</strong><span>Select a region and date window to reconstruct fire-weather conditions.</span></div>",
        unsafe_allow_html=True,
    )
    history_left, history_right = st.columns([1.15, 1])
    with history_left:
        history_location_labels = germany_location_options()
        selected_history_label = st.selectbox(
            "Germany region",
            list(history_location_labels.keys()),
            key="historical_germany_region",
        )
        selected_history_location = history_location_labels[selected_history_label]
    with history_right:
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
            history, source_name = load_selected_location_history(
                selected_history_location["area"],
                selected_history_location["site"],
                hist_start,
                hist_end,
            )
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
                        ("Region", selected_history_location["area"]),
                        ("Site", selected_history_location["site"]),
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
                historical_actions = operational_actions(
                    worst_risk,
                    float(worst["temperature_c"]),
                    float(worst["wind_kmh"]),
                    str(worst["wind_direction"]),
                    degrees_to_compass((float(worst["wind_direction_deg"]) + 180) % 360),
                )
                action_panel("Incident review notes", worst_risk, historical_actions)
                st.download_button(
                    "Download historical review",
                    data=incident_brief_text(selected_history_location, worst, worst_risk, worst, worst_risk, historical_actions),
                    file_name=f"ctrlf_historical_review_{selected_history_location['area'].lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                left, right = st.columns([2, 1])
                with left:
                    st.markdown("#### Incident-Window Weather")
                    st.caption(f"{selected_history_location['site']} · {hist_start:%d %b %Y} to {hist_end:%d %b %Y}")
                    svg_line_chart(
                        history.set_index("time")[["temperature_c", "humidity_pct", "wind_kmh", "rain_mm"]],
                        height=360,
                    )
                with right:
                    st.markdown("#### Fire-Weather Rule Checks")
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
                st.caption(f"Dominant wind directions for {selected_history_location['site']} during the selected window.")
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

with weather_tab:
    st.subheader("Weather Analysis")
    st.write("Explore supporting research datasets for weather and wind-pattern analysis.")
    selected_weather = st.selectbox("Dataset", ["Paris weather", "Germany wind"], key="weather_analysis_dataset")

    if selected_weather == "Paris weather":
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
        insight("Paris is a short summer weather sample. Use it for temperature, rain, and hourly pattern checks.")

        st.markdown("#### Hourly Trend")
        st.caption("Temperature, dew point, wind, and rain.")
        svg_line_chart(paris.set_index("time_utc")[["t2m_c", "d2m_c", "wind10_mps", "tp_mm"]], height=340)

        with st.expander("Daily labels"):
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
        with st.expander("Wind direction"):
            svg_bar_chart(direction_counts(paris, "wind_direction"), height=260)
        with st.expander("Detailed Paris rows"):
            html_table(round_numeric(paris), use_container_width=True, hide_index=True)
    else:
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
        insight("Germany gives the full-year wind pattern. Use it for seasonal wind and spread-direction context.")

        left, right = st.columns([2, 1])
        with left:
            st.markdown("#### Wind Over Time")
            svg_line_chart(germany.set_index("time_utc")[["wind10_mps_mean", "wind10_mps_p90"]], height=340)
        with right:
            st.markdown("#### Monthly Wind")
            svg_bar_chart(data["germany_monthly"].set_index("month")["wind10_mps_mean"], height=340)

        with st.expander("Wind direction"):
            svg_bar_chart(direction_counts(germany, "wind_direction"), height=260)
        with st.expander("Wind by location"):
            map_data = data["germany_grid"].rename(columns={"latitude": "lat", "longitude": "lon"})
            svg_scatter_chart(
                map_data,
                x="lon",
                y="lat",
                size="wind10_mps_mean",
                color="wind10_mps_mean",
                height=420,
            )
        with st.expander("Strongest wind locations"):
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
