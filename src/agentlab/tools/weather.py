"""天气工具:Open-Meteo 免费、无需 Key,适合练手。

两步:城市名 -> 经纬度(geocoding),经纬度 -> 天气(forecast)。
"""

from __future__ import annotations

from typing import Any

import httpx

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气代码 -> 中文
WEATHER_CODES = {
    0: "晴",
    1: "晴间多云",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "小阵雨",
    81: "阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def get_weather(
    city: str,
    *,
    client: httpx.Client | None = None,
    forecast_days: int = 3,
) -> dict[str, Any]:
    owns_client = client is None
    client = client or httpx.Client(timeout=15.0)
    try:
        geo_resp = client.get(
            GEO_URL,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results") or []
        if not results:
            raise ValueError(f"找不到城市: {city}")
        place = results[0]

        forecast_resp = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,weather_code"
                ),
                "timezone": "auto",
                "forecast_days": max(1, min(int(forecast_days), 7)),
            },
        )
        forecast_resp.raise_for_status()
        data = forecast_resp.json()
    finally:
        if owns_client:
            client.close()

    current = data.get("current") or {}
    daily = data.get("daily") or {}
    days = []
    for index, date in enumerate(daily.get("time") or []):
        code = (daily.get("weather_code") or [None])[index]
        days.append(
            {
                "date": date,
                "weather": WEATHER_CODES.get(code, f"未知({code})"),
                "temp_max": (daily.get("temperature_2m_max") or [None])[index],
                "temp_min": (daily.get("temperature_2m_min") or [None])[index],
                "precipitation_probability": (daily.get("precipitation_probability_max") or [None])[
                    index
                ],
            }
        )

    return {
        "city": place.get("name"),
        "region": place.get("admin1"),
        "country": place.get("country"),
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather": WEATHER_CODES.get(
                current.get("weather_code"), f"未知({current.get('weather_code')})"
            ),
        },
        "daily": days,
        "source": "open-meteo.com",
    }
