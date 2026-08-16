from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


SENSOR_LOCATIONS = [
    ("HF-01", 50.7174, 6.3732, "North ridge"),
    ("HF-02", 50.7138, 6.3807, "Trail edge"),
    ("HF-03", 50.7202, 6.3904, "Dry slope"),
    ("HF-04", 50.7096, 6.3658, "Forest road"),
    ("HF-05", 50.7245, 6.3711, "Lookout"),
]


def degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((degrees + 22.5) // 45) % 8
    return directions[index]


def generate_sensor_demo(hours: int = 24, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [end - timedelta(hours=hours - 1 - i) for i in range(hours)]
    rows = []

    for sensor_id, lat, lon, zone in SENSOR_LOCATIONS:
        hotspot = sensor_id == "HF-03"
        for i, timestamp in enumerate(times):
            hour_wave = np.sin((i / max(hours - 1, 1)) * np.pi)
            temperature = 22 + 8 * hour_wave + rng.normal(0, 0.7)
            humidity = 54 - 20 * hour_wave + rng.normal(0, 2.2)
            wind_kmh = 10 + 14 * hour_wave + rng.normal(0, 1.8)
            wind_direction_deg = (235 + rng.normal(0, 18)) % 360
            smoke_ppm = 4 + rng.normal(0, 0.8)
            co_ppm = 1.2 + rng.normal(0, 0.25)
            flame_ir = 0.05 + rng.normal(0, 0.02)

            if hotspot and i >= int(hours * 0.68):
                ramp = i - int(hours * 0.68) + 1
                temperature += ramp * 0.8
                humidity -= ramp * 1.7
                smoke_ppm += ramp * 4.5
                co_ppm += ramp * 0.9
                flame_ir += ramp * 0.06
                wind_kmh += ramp * 0.7

            rows.append(
                {
                    "time": timestamp,
                    "sensor_id": sensor_id,
                    "zone": zone,
                    "lat": lat,
                    "lon": lon,
                    "temperature_c": max(0, temperature),
                    "humidity_pct": np.clip(humidity, 5, 100),
                    "wind_kmh": max(0, wind_kmh),
                    "wind_direction_deg": wind_direction_deg,
                    "wind_direction": degrees_to_compass(wind_direction_deg),
                    "spread_direction_deg": (wind_direction_deg + 180) % 360,
                    "spread_direction": degrees_to_compass((wind_direction_deg + 180) % 360),
                    "smoke_ppm": max(0, smoke_ppm),
                    "co_ppm": max(0, co_ppm),
                    "flame_ir": np.clip(flame_ir, 0, 1),
                    "battery_pct": np.clip(96 - i * 0.08 + rng.normal(0, 0.3), 0, 100),
                }
            )

    data = pd.DataFrame(rows)
    return add_prediction(data)


def add_prediction(data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    scored["fire_probability_pct"] = (
        0.9 * (scored["temperature_c"] - 20)
        + 0.9 * (35 - scored["humidity_pct"])
        + 0.75 * scored["wind_kmh"]
        + 1.9 * scored["smoke_ppm"]
        + 6.5 * scored["co_ppm"]
        + 45 * scored["flame_ir"]
        - 35
    ).clip(0, 100)
    scored["prediction"] = pd.cut(
        scored["fire_probability_pct"],
        bins=[-1, 30, 60, 80, 101],
        labels=["Low", "Watch", "High", "Critical"],
    ).astype(str)
    return scored
