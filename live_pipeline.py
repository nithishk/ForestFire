from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlencode

import pandas as pd
import requests


HURTGENWALD_LAT = 50.716
HURTGENWALD_LON = 6.375
EFFIS_WMS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]


def degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((degrees + 22.5) // 45) % 8
    return directions[index]


def weather_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    weather = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "temperature_c": hourly["temperature_2m"],
            "humidity_pct": hourly["relative_humidity_2m"],
            "rain_mm": hourly["precipitation"],
            "wind_kmh": hourly["wind_speed_10m"],
            "wind_direction_deg": hourly["wind_direction_10m"],
        }
    )
    return add_fire_risk_columns(weather)


def add_fire_risk_columns(weather: pd.DataFrame) -> pd.DataFrame:
    enriched = weather.copy()
    enriched["wind_direction"] = enriched["wind_direction_deg"].apply(degrees_to_compass)
    enriched["fire_30_30_30"] = (
        (enriched["temperature_c"] >= 30)
        & (enriched["humidity_pct"] <= 30)
        & (enriched["wind_kmh"] >= 30)
    )
    enriched["fire_weather_score"] = (
        (enriched["temperature_c"] >= 30).astype(int)
        + (enriched["humidity_pct"] <= 30).astype(int)
        + (enriched["wind_kmh"] >= 30).astype(int)
    )
    return enriched


def fetch_hurtgenwald_weather() -> pd.DataFrame:
    """Fetch 3-day hourly weather forecast for Huertgenwald."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": HURTGENWALD_LAT,
        "longitude": HURTGENWALD_LON,
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_days": 3,
        "timezone": "Europe/Berlin",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return weather_frame(response.json())


def fetch_hurtgenwald_recent_weather(past_days: int = 14) -> pd.DataFrame:
    """Fetch recent hourly weather for incident review."""
    days = max(1, min(int(past_days), 92))
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": HURTGENWALD_LAT,
        "longitude": HURTGENWALD_LON,
        "hourly": ",".join(WEATHER_VARIABLES),
        "past_days": days,
        "forecast_days": 1,
        "timezone": "Europe/Berlin",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return weather_frame(response.json())


def fetch_hurtgenwald_archive_weather(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch older historical hourly weather from the archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": HURTGENWALD_LAT,
        "longitude": HURTGENWALD_LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "Europe/Berlin",
        "wind_speed_unit": "kmh",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return weather_frame(response.json())


def fetch_hurtgenwald_history(start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    """Choose recent or archive source based on requested dates."""
    today = date.today()
    if start_date >= today - timedelta(days=92):
        weather = fetch_hurtgenwald_recent_weather((today - start_date).days + 1)
        mask = (weather["time"].dt.date >= start_date) & (weather["time"].dt.date <= end_date)
        return weather.loc[mask].copy(), "recent forecast API"

    archive_end = min(end_date, today - timedelta(days=5))
    return fetch_hurtgenwald_archive_weather(start_date, archive_end), "historical archive API"


def effis_map_url(layer: str = "mf010.fwi", map_date: date | None = None) -> str:
    """Build a Copernicus EFFIS WMS map URL for the Huertgenwald/NRW area."""
    selected_date = map_date or date.today()
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": "5.95,50.35,6.85,51.05",
        "WIDTH": "900",
        "HEIGHT": "600",
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        "TIME": selected_date.isoformat(),
    }
    return EFFIS_WMS_URL + "?" + urlencode(params)


def risk_label(score: int, full_rule: bool) -> str:
    if full_rule:
        return "Critical"
    if score >= 2:
        return "High"
    if score == 1:
        return "Elevated"
    return "Low"
