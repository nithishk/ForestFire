from __future__ import annotations

from datetime import date
from urllib.parse import urlencode

import pandas as pd
import requests


HURTGENWALD_LAT = 50.716
HURTGENWALD_LON = 6.375
EFFIS_WMS_URL = "https://maps.effis.emergency.copernicus.eu/effis"


def degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((degrees + 22.5) // 45) % 8
    return directions[index]


def fetch_hurtgenwald_weather() -> pd.DataFrame:
    """Fetch 3-day hourly weather forecast for Hürtgenwald."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": HURTGENWALD_LAT,
        "longitude": HURTGENWALD_LON,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "forecast_days": 3,
        "timezone": "Europe/Berlin",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()["hourly"]
    weather = pd.DataFrame(
        {
            "time": pd.to_datetime(payload["time"]),
            "temperature_c": payload["temperature_2m"],
            "humidity_pct": payload["relative_humidity_2m"],
            "rain_mm": payload["precipitation"],
            "wind_kmh": payload["wind_speed_10m"],
            "wind_direction_deg": payload["wind_direction_10m"],
        }
    )
    weather["wind_direction"] = weather["wind_direction_deg"].apply(degrees_to_compass)
    weather["fire_30_30_30"] = (
        (weather["temperature_c"] >= 30)
        & (weather["humidity_pct"] <= 30)
        & (weather["wind_kmh"] >= 30)
    )
    weather["fire_weather_score"] = (
        (weather["temperature_c"] >= 30).astype(int)
        + (weather["humidity_pct"] <= 30).astype(int)
        + (weather["wind_kmh"] >= 30).astype(int)
    )
    return weather


def effis_map_url(layer: str = "mf010.fwi", map_date: date | None = None) -> str:
    """Build a Copernicus EFFIS WMS map URL for the Hürtgenwald/NRW area."""
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
