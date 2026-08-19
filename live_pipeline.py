from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import urlencode

import pandas as pd
import requests


HURTGENWALD_LAT = 50.716
HURTGENWALD_LON = 6.375
EFFIS_WMS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
GLOBAL_FIRE_LOCATIONS = [
    {
        "country": "Germany",
        "area": "North Rhine-Westphalia",
        "site": "Huertgenwald, NRW",
        "latitude": 50.716,
        "longitude": 6.375,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Brandenburg",
        "site": "Potsdam forest belt",
        "latitude": 52.390,
        "longitude": 13.064,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Bavaria",
        "site": "Bavarian Forest",
        "latitude": 49.050,
        "longitude": 13.250,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Saxony",
        "site": "Saxon Switzerland",
        "latitude": 50.920,
        "longitude": 14.155,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Saxony-Anhalt",
        "site": "Harz forest edge",
        "latitude": 51.790,
        "longitude": 10.955,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Lower Saxony",
        "site": "Lueneburg Heath",
        "latitude": 53.165,
        "longitude": 9.938,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Baden-Wuerttemberg",
        "site": "Black Forest",
        "latitude": 48.277,
        "longitude": 8.186,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "Germany",
        "area": "Mecklenburg-Vorpommern",
        "site": "Mueritz forest belt",
        "latitude": 53.423,
        "longitude": 12.678,
        "timezone": "Europe/Berlin",
    },
    {
        "country": "USA",
        "area": "California",
        "site": "Los Angeles foothills, CA",
        "latitude": 34.199,
        "longitude": -118.176,
        "timezone": "America/Los_Angeles",
    },
    {
        "country": "USA",
        "area": "Colorado",
        "site": "Boulder foothills, CO",
        "latitude": 40.015,
        "longitude": -105.270,
        "timezone": "America/Denver",
    },
    {
        "country": "USA",
        "area": "Oregon",
        "site": "Bend dry forest, OR",
        "latitude": 44.058,
        "longitude": -121.315,
        "timezone": "America/Los_Angeles",
    },
    {
        "country": "USA",
        "area": "Arizona",
        "site": "Flagstaff ponderosa belt, AZ",
        "latitude": 35.199,
        "longitude": -111.651,
        "timezone": "America/Phoenix",
    },
    {
        "country": "USA",
        "area": "New Mexico",
        "site": "Santa Fe wildland edge, NM",
        "latitude": 35.687,
        "longitude": -105.938,
        "timezone": "America/Denver",
    },
    {
        "country": "USA",
        "area": "Idaho",
        "site": "Boise foothills, ID",
        "latitude": 43.615,
        "longitude": -116.202,
        "timezone": "America/Boise",
    },
    {
        "country": "USA",
        "area": "Montana",
        "site": "Missoula wildland edge, MT",
        "latitude": 46.872,
        "longitude": -113.994,
        "timezone": "America/Denver",
    },
    {
        "country": "USA",
        "area": "Texas",
        "site": "Austin Hill Country, TX",
        "latitude": 30.267,
        "longitude": -97.743,
        "timezone": "America/Chicago",
    },
    {
        "country": "USA",
        "area": "Washington",
        "site": "Wenatchee dry forest, WA",
        "latitude": 47.423,
        "longitude": -120.310,
        "timezone": "America/Los_Angeles",
    },
    {
        "country": "Canada",
        "area": "Alberta",
        "site": "Fort McMurray, Alberta",
        "latitude": 56.726,
        "longitude": -111.380,
        "timezone": "America/Edmonton",
    },
    {
        "country": "Canada",
        "area": "British Columbia",
        "site": "Kelowna wildland edge, BC",
        "latitude": 49.888,
        "longitude": -119.496,
        "timezone": "America/Vancouver",
    },
    {
        "country": "Canada",
        "area": "Ontario",
        "site": "Thunder Bay forest, ON",
        "latitude": 48.380,
        "longitude": -89.247,
        "timezone": "America/Toronto",
    },
]
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


def fetch_location_weather(location: dict, forecast_days: int = 3) -> pd.DataFrame:
    """Fetch a live forecast for one demo wildfire-monitoring location."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "forecast_days": forecast_days,
        "timezone": location["timezone"],
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    weather = weather_frame(response.json())
    weather["country"] = location["country"]
    weather["area"] = location["area"]
    weather["site"] = location["site"]
    weather["latitude"] = location["latitude"]
    weather["longitude"] = location["longitude"]
    weather["timezone"] = location["timezone"]
    return weather


def fetch_global_fire_weather() -> pd.DataFrame:
    """Fetch live forecasts for representative Germany, USA, and Canada demo sites."""
    frames = [fetch_location_weather(location) for location in GLOBAL_FIRE_LOCATIONS]
    return pd.concat(frames, ignore_index=True)


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
