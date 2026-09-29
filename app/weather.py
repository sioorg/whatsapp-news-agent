"""Live weather via Open-Meteo — free, keyless, no signup, no rate limit for
this scale. Used by the get_weather tool in app/tools.py.

Deliberately not cached into the local knowledge base (app.rag): unlike a
news article, current conditions are perishable. A stale cached reading
served up later as "current" would be actively wrong, not just outdated
context — so every call here is a fresh live lookup, always.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

# WMO weather codes, as used by Open-Meteo's `weather_code` field.
_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def describe(weather_code: int) -> str:
    return _WEATHER_CODES.get(weather_code, f"weather code {weather_code}")


def geocode(location: str) -> dict | None:
    """Resolve a place name to coordinates. None if nothing matches.

    When several places share a name (e.g. "Springfield" matches five US
    cities), picks the most populous — Open-Meteo returns population for
    most entries.

    This does NOT fix every ambiguity: a well-known alias can resolve to an
    obscure, wrong place rather than the famous city meant, because
    Open-Meteo only matches the literal name given, not alternate/former
    names. Concretely verified: searching "Bangalore" returns only
    "Bangalore Town, Sindh, Pakistan" (population not even listed) — the
    actual 8.5-million-person city is indexed solely as "Bengaluru". There
    is no complete fix short of a hand-maintained alias table, which isn't
    attempted here; get_weather's docstring instead asks the calling model
    to prefer current official names, leaning on its own general knowledge
    of common aliases.
    """

    response = requests.get(
        GEOCODE_URL,
        params={"name": location, "count": 5, "language": "en", "format": "json"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    return max(results, key=lambda r: r.get("population") or 0)


def current_and_today(latitude: float, longitude: float) -> dict:
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m"
            ),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def format_report(place: dict, forecast: dict) -> str:
    current = forecast["current"]
    daily = forecast["daily"]

    where = ", ".join(
        part for part in [place.get("name"), place.get("admin1"), place.get("country")] if part
    )

    return (
        f"Weather for {where}:\n"
        f"Now: {current['temperature_2m']}°C "
        f"(feels like {current['apparent_temperature']}°C), "
        f"{describe(current['weather_code'])}, "
        f"{current['relative_humidity_2m']}% humidity, "
        f"wind {current['wind_speed_10m']} km/h.\n"
        f"Today: high {daily['temperature_2m_max'][0]}°C, "
        f"low {daily['temperature_2m_min'][0]}°C, "
        f"{daily['precipitation_probability_max'][0]}% chance of precipitation."
    )


def get_report(location: str) -> str:
    place = geocode(location)
    if place is None:
        return f"Couldn't find a place called '{location}'."

    forecast = current_and_today(place["latitude"], place["longitude"])
    return format_report(place, forecast)
