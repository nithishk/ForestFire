from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "weather_analysis"
WORKING_DIR = ROOT / "working" / "extracted_paris"


PARIS_ZIP = ROOT / "Paris 15.07-25.07 2cf2906b8c36a5d3534c2a9ef843042b.zip"
GERMANY_NC = ROOT / "data_stream-oper_stepType-instant.nc"


def wind_speed(u: pd.Series, v: pd.Series) -> pd.Series:
    return np.sqrt(u**2 + v**2)


def wind_direction_deg(u: pd.Series, v: pd.Series) -> pd.Series:
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def prepare_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORKING_DIR.mkdir(parents=True, exist_ok=True)


def extract_paris_zip() -> tuple[Path, Path]:
    with zipfile.ZipFile(PARIS_ZIP) as archive:
        archive.extractall(WORKING_DIR)
    return (
        WORKING_DIR / "data_stream-oper_stepType-instant.nc",
        WORKING_DIR / "data_stream-oper_stepType-accum.nc",
    )


def dataset_to_frame(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index()
    drop_cols = [c for c in ["number", "expver"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def convert_paris() -> dict[str, str]:
    instant_path, accum_path = extract_paris_zip()
    instant = xr.open_dataset(instant_path)
    accum = xr.open_dataset(accum_path)

    paris = dataset_to_frame(instant).merge(
        dataset_to_frame(accum),
        on=["valid_time", "latitude", "longitude"],
        how="left",
    )
    instant.close()
    accum.close()

    paris = paris.rename(columns={"valid_time": "time_utc"})
    paris["t2m_c"] = paris["t2m"] - 273.15
    paris["d2m_c"] = paris["d2m"] - 273.15
    paris["skt_c"] = paris["skt"] - 273.15
    paris["stl1_c"] = paris["stl1"] - 273.15
    paris["msl_hpa"] = paris["msl"] / 100
    paris["sp_hpa"] = paris["sp"] / 100
    paris["tp_mm"] = paris["tp"] * 1000
    paris["wind10_mps"] = wind_speed(paris["u10"], paris["v10"])
    paris["wind100_mps"] = wind_speed(paris["u100"], paris["v100"])
    paris["wind10_direction_deg"] = wind_direction_deg(paris["u10"], paris["v10"])
    paris["wind100_direction_deg"] = wind_direction_deg(paris["u100"], paris["v100"])

    ordered = [
        "time_utc",
        "latitude",
        "longitude",
        "t2m_c",
        "d2m_c",
        "skt_c",
        "stl1_c",
        "msl_hpa",
        "sp_hpa",
        "tp_mm",
        "u10",
        "v10",
        "wind10_mps",
        "wind10_direction_deg",
        "u100",
        "v100",
        "wind100_mps",
        "wind100_direction_deg",
        "swvl1",
        "blh",
    ]
    paris = paris[ordered].sort_values(["time_utc", "latitude", "longitude"])
    paris_grid_csv = OUTPUT_DIR / "paris_hourly_grid.csv"
    paris.to_csv(paris_grid_csv, index=False)

    value_cols = [c for c in paris.columns if c not in ["time_utc", "latitude", "longitude"]]
    paris_hourly = paris.groupby("time_utc", as_index=False)[value_cols].mean()
    paris_hourly_csv = OUTPUT_DIR / "paris_hourly_area_mean.csv"
    paris_hourly.to_csv(paris_hourly_csv, index=False)

    paris_daily = (
        paris_hourly.assign(date=pd.to_datetime(paris_hourly["time_utc"]).dt.date)
        .groupby("date")
        .agg(
            temp_c_mean=("t2m_c", "mean"),
            temp_c_min=("t2m_c", "min"),
            temp_c_max=("t2m_c", "max"),
            dewpoint_c_mean=("d2m_c", "mean"),
            precipitation_mm_total=("tp_mm", "sum"),
            wind10_mps_mean=("wind10_mps", "mean"),
            wind10_mps_max=("wind10_mps", "max"),
            pressure_hpa_mean=("msl_hpa", "mean"),
            boundary_layer_height_m_mean=("blh", "mean"),
        )
        .reset_index()
    )
    paris_daily_csv = OUTPUT_DIR / "paris_daily_summary.csv"
    paris_daily.to_csv(paris_daily_csv, index=False)

    return {
        "paris_grid_csv": str(paris_grid_csv),
        "paris_hourly_csv": str(paris_hourly_csv),
        "paris_daily_csv": str(paris_daily_csv),
    }


def convert_germany() -> dict[str, str]:
    ds = xr.open_dataset(GERMANY_NC)
    df = dataset_to_frame(ds).rename(columns={"valid_time": "time_utc"})
    ds.close()

    df["wind10_mps"] = wind_speed(df["u10"], df["v10"])
    df["wind10_direction_deg"] = wind_direction_deg(df["u10"], df["v10"])
    df = df.sort_values(["time_utc", "latitude", "longitude"])

    germany_hourly = (
        df.groupby("time_utc", as_index=False)
        .agg(
            u10_mean=("u10", "mean"),
            v10_mean=("v10", "mean"),
            wind10_mps_mean=("wind10_mps", "mean"),
            wind10_mps_min=("wind10_mps", "min"),
            wind10_mps_max=("wind10_mps", "max"),
            wind10_mps_p10=("wind10_mps", lambda s: s.quantile(0.10)),
            wind10_mps_p90=("wind10_mps", lambda s: s.quantile(0.90)),
        )
    )
    germany_hourly["wind10_direction_deg_mean"] = wind_direction_deg(
        germany_hourly["u10_mean"], germany_hourly["v10_mean"]
    )
    germany_hourly_csv = OUTPUT_DIR / "germany_hourly_area_mean.csv"
    germany_hourly.to_csv(germany_hourly_csv, index=False)

    germany_daily = (
        germany_hourly.assign(date=pd.to_datetime(germany_hourly["time_utc"]).dt.date)
        .groupby("date")
        .agg(
            wind10_mps_mean=("wind10_mps_mean", "mean"),
            wind10_mps_min=("wind10_mps_min", "min"),
            wind10_mps_max=("wind10_mps_max", "max"),
            wind10_mps_p90=("wind10_mps_p90", "mean"),
        )
        .reset_index()
    )
    germany_daily_csv = OUTPUT_DIR / "germany_daily_summary.csv"
    germany_daily.to_csv(germany_daily_csv, index=False)

    germany_monthly = (
        germany_hourly.assign(month=pd.to_datetime(germany_hourly["time_utc"]).dt.to_period("M").astype(str))
        .groupby("month")
        .agg(
            wind10_mps_mean=("wind10_mps_mean", "mean"),
            wind10_mps_min=("wind10_mps_min", "min"),
            wind10_mps_max=("wind10_mps_max", "max"),
            wind10_mps_p90=("wind10_mps_p90", "mean"),
        )
        .reset_index()
    )
    germany_monthly_csv = OUTPUT_DIR / "germany_monthly_summary.csv"
    germany_monthly.to_csv(germany_monthly_csv, index=False)

    germany_grid = (
        df.groupby(["latitude", "longitude"], as_index=False)
        .agg(
            wind10_mps_mean=("wind10_mps", "mean"),
            wind10_mps_min=("wind10_mps", "min"),
            wind10_mps_max=("wind10_mps", "max"),
            wind10_mps_p90=("wind10_mps", lambda s: s.quantile(0.90)),
            u10_mean=("u10", "mean"),
            v10_mean=("v10", "mean"),
        )
    )
    germany_grid["wind10_direction_deg_mean"] = wind_direction_deg(
        germany_grid["u10_mean"], germany_grid["v10_mean"]
    )
    germany_grid_csv = OUTPUT_DIR / "germany_grid_cell_summary.csv"
    germany_grid.to_csv(germany_grid_csv, index=False)

    return {
        "germany_hourly_csv": str(germany_hourly_csv),
        "germany_daily_csv": str(germany_daily_csv),
        "germany_monthly_csv": str(germany_monthly_csv),
        "germany_grid_csv": str(germany_grid_csv),
    }


def build_analysis(paths: dict[str, str]) -> dict[str, object]:
    paris_daily = pd.read_csv(paths["paris_daily_csv"])
    paris_hourly = pd.read_csv(paths["paris_hourly_csv"], parse_dates=["time_utc"])
    germany_daily = pd.read_csv(paths["germany_daily_csv"])
    germany_monthly = pd.read_csv(paths["germany_monthly_csv"])
    germany_grid = pd.read_csv(paths["germany_grid_csv"])

    analysis = {
        "paris": {
            "date_start": str(paris_daily["date"].min()),
            "date_end": str(paris_daily["date"].max()),
            "mean_temp_c": round(float(paris_hourly["t2m_c"].mean()), 2),
            "max_temp_c": round(float(paris_hourly["t2m_c"].max()), 2),
            "min_temp_c": round(float(paris_hourly["t2m_c"].min()), 2),
            "total_precip_mm": round(float(paris_hourly["tp_mm"].sum()), 2),
            "mean_wind10_mps": round(float(paris_hourly["wind10_mps"].mean()), 2),
            "windiest_hour_utc": str(paris_hourly.loc[paris_hourly["wind10_mps"].idxmax(), "time_utc"]),
            "hottest_hour_utc": str(paris_hourly.loc[paris_hourly["t2m_c"].idxmax(), "time_utc"]),
        },
        "germany": {
            "date_start": str(germany_daily["date"].min()),
            "date_end": str(germany_daily["date"].max()),
            "mean_wind10_mps": round(float(germany_daily["wind10_mps_mean"].mean()), 2),
            "max_area_wind10_mps": round(float(germany_daily["wind10_mps_max"].max()), 2),
            "windiest_day": str(germany_daily.loc[germany_daily["wind10_mps_mean"].idxmax(), "date"]),
            "windiest_month": str(germany_monthly.loc[germany_monthly["wind10_mps_mean"].idxmax(), "month"]),
            "calmest_month": str(germany_monthly.loc[germany_monthly["wind10_mps_mean"].idxmin(), "month"]),
            "highest_mean_grid_cell": {
                "latitude": round(float(germany_grid.loc[germany_grid["wind10_mps_mean"].idxmax(), "latitude"]), 4),
                "longitude": round(float(germany_grid.loc[germany_grid["wind10_mps_mean"].idxmax(), "longitude"]), 4),
                "mean_wind10_mps": round(float(germany_grid["wind10_mps_mean"].max()), 2),
            },
        },
    }
    analysis_path = OUTPUT_DIR / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def main() -> None:
    prepare_dirs()
    paths = {}
    paths.update(convert_paris())
    paths.update(convert_germany())
    analysis = build_analysis(paths)
    manifest = {"outputs": paths, "analysis": analysis}
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
