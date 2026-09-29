"""app.weather: geocoding disambiguation, formatting, and the "not found"
path — all against a mocked requests.get, never the real Open-Meteo API
(that's tests/test_integration.py's job, since it's free/keyless and safe
to hit for real, but still shouldn't be in the fast/mocked default layer).
"""

import app.weather as weather


class _FakeResponse:
    def __init__(self, json_body, status=200):
        self._json = json_body
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _forecast(days=1, unit_symbol="°C"):
    """A minimal but shape-accurate multi-day-capable forecast body,
    matching what Open-Meteo actually returns (verified directly against
    the real API before writing this test data)."""

    return {
        "current_units": {"temperature_2m": unit_symbol},
        "current": {
            "temperature_2m": 27.6,
            "apparent_temperature": 31.5,
            "weather_code": 3,
            "relative_humidity_2m": 66,
            "wind_speed_10m": 1.5,
        },
        "daily": {
            # format_report never parses these, just echoes them — plain
            # placeholders are fine, no need for real calendar dates.
            "time": [f"2026-day-{i}" for i in range(days)],
            "temperature_2m_max": [27.8 + i for i in range(days)],
            "temperature_2m_min": [19.8 + i for i in range(days)],
            "precipitation_probability_max": [10 + i for i in range(days)],
        },
    }


def test_describe_known_code():
    assert weather.describe(0) == "clear sky"
    assert weather.describe(95) == "thunderstorm"


def test_describe_unknown_code_falls_back():
    assert weather.describe(12345) == "weather code 12345"


def test_geocode_returns_none_for_no_results(monkeypatch):
    monkeypatch.setattr(
        weather.requests, "get", lambda *a, **k: _FakeResponse({"results": []})
    )
    assert weather.geocode("Nowhereville") is None


def test_geocode_returns_none_when_results_key_is_missing(monkeypatch):
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: _FakeResponse({}))
    assert weather.geocode("Nowhereville") is None


def test_geocode_picks_the_most_populous_match(monkeypatch):
    """Reproduces the real "Springfield" ambiguity found during
    development: Open-Meteo returns several same-named US cities, and the
    right default is the most populous one, not the first in the list."""

    results = [
        {"name": "Springfield", "country": "United States", "admin1": "Ohio", "population": 59680},
        {"name": "Springfield", "country": "United States", "admin1": "Missouri", "population": 170188},
        {"name": "Springfield", "country": "United States", "admin1": "Illinois", "population": 114394},
    ]
    monkeypatch.setattr(
        weather.requests, "get", lambda *a, **k: _FakeResponse({"results": results})
    )

    place = weather.geocode("Springfield")

    assert place["admin1"] == "Missouri"


def test_geocode_treats_missing_population_as_zero(monkeypatch):
    """A real response seen during development: an obscure place with no
    population field at all, alongside one that has it — must not crash
    comparing None to an int, and must prefer the one with real data."""

    results = [
        {"name": "A", "population": None},
        {"name": "B", "population": 500},
    ]
    monkeypatch.setattr(
        weather.requests, "get", lambda *a, **k: _FakeResponse({"results": results})
    )

    assert weather.geocode("Somewhere")["name"] == "B"


def test_format_report_includes_location_and_conditions():
    place = {"name": "Bengaluru", "admin1": "Karnataka", "country": "India"}

    report = weather.format_report(place, _forecast())

    assert "Bengaluru, Karnataka, India" in report
    assert "27.6°C" in report
    assert "overcast" in report
    assert "19.8°C" in report
    assert "Today:" in report  # single day still uses the original wording


def test_format_report_omits_missing_location_parts():
    place = {"name": "Somewhere", "admin1": None, "country": None}

    report = weather.format_report(place, _forecast())

    assert report.startswith("Weather for Somewhere:")


def test_format_report_uses_whatever_unit_symbol_the_api_returned():
    """Never hardcode °C — Open-Meteo echoes back the actual unit used,
    which is what temperature_unit=fahrenheit changes."""

    place = {"name": "Miami", "admin1": None, "country": "United States"}

    report = weather.format_report(place, _forecast(unit_symbol="°F"))

    assert "°F" in report
    assert "°C" not in report


def test_format_report_lists_every_day_for_a_multi_day_forecast():
    place = {"name": "Paris", "admin1": None, "country": "France"}

    report = weather.format_report(place, _forecast(days=3))

    assert "Forecast:" in report
    assert report.count("chance of precipitation") == 3
    assert "2026-day-0" in report
    assert "2026-day-2" in report


def test_get_report_returns_a_clear_message_when_the_place_is_not_found(monkeypatch):
    monkeypatch.setattr(
        weather.requests, "get", lambda *a, **k: _FakeResponse({"results": []})
    )

    assert weather.get_report("Nowhereville") == "Couldn't find a place called 'Nowhereville'."


def test_get_report_builds_a_full_report_end_to_end(monkeypatch):
    geocode_response = _FakeResponse(
        {"results": [{"name": "Paris", "country": "France", "latitude": 48.85, "longitude": 2.35}]}
    )
    forecast_body = _forecast()
    forecast_body["current"]["weather_code"] = 61  # "slight rain"
    forecast_response = _FakeResponse(forecast_body)

    def fake_get(url, params=None, timeout=None):
        return geocode_response if url == weather.GEOCODE_URL else forecast_response

    monkeypatch.setattr(weather.requests, "get", fake_get)

    report = weather.get_report("Paris")

    assert "Paris, France" in report
    assert "slight rain" in report


def test_normalize_unit_accepts_common_spellings():
    assert weather._normalize_unit("fahrenheit") == "fahrenheit"
    assert weather._normalize_unit("Fahrenheit") == "fahrenheit"
    assert weather._normalize_unit("F") == "fahrenheit"
    assert weather._normalize_unit("celsius") == "celsius"
    assert weather._normalize_unit("anything else") == "celsius"


def test_fetch_forecast_requests_the_normalized_unit(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse(_forecast())

    monkeypatch.setattr(weather.requests, "get", fake_get)

    weather.fetch_forecast(1.0, 2.0, unit="Fahrenheit")

    assert captured["temperature_unit"] == "fahrenheit"


def test_fetch_forecast_clamps_days_to_the_max(monkeypatch):
    """A malformed or over-eager tool call (e.g. the model asking for 100
    days) must not turn into an oversized or rejected request."""

    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse(_forecast())

    monkeypatch.setattr(weather.requests, "get", fake_get)

    weather.fetch_forecast(1.0, 2.0, days=100)

    assert captured["forecast_days"] == weather.MAX_FORECAST_DAYS


def test_fetch_forecast_clamps_days_to_at_least_one(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured.update(params)
        return _FakeResponse(_forecast())

    monkeypatch.setattr(weather.requests, "get", fake_get)

    weather.fetch_forecast(1.0, 2.0, days=0)

    assert captured["forecast_days"] == 1
