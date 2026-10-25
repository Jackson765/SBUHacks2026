"""Existing Open-Meteo client with explicit units, provenance, and caching."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from time import monotonic

import requests
from config import (WEATHER_CACHE_SECONDS, WEATHER_EXPECTED_INTERVAL_SECONDS,
                    WEATHER_LOCATION, WEATHER_LOCATION_TOLERANCE_DEGREES, WEATHER_TIMEOUT)

URL = 'https://api.open-meteo.com/v1/forecast'
LIVE_SCENARIO = 'LIVE WEATHER SCENARIO - not weather recorded with the video'
FIELDS = {
    'temperature_2m': ('temperature_f', '°F'),
    'relative_humidity_2m': ('humidity_percent', '%'),
    'apparent_temperature': ('feels_like_f', '°F'),
    'precipitation': ('precipitation_mm', 'mm'), 'rain': ('rain_mm', 'mm'),
    'snowfall': ('snowfall_cm', 'cm'), 'weather_code': ('weather_code', 'wmo code'),
    'wind_speed_10m': ('wind_speed_mph', 'mph'), 'visibility': ('visibility_m', 'm'),
}


def utc_now():
    return datetime.now(timezone.utc)


def request_params(location):
    return dict(latitude=location['latitude'], longitude=location['longitude'],
                current=list(FIELDS), temperature_unit='fahrenheit', wind_speed_unit='mph',
                precipitation_unit='mm', timezone='America/New_York', timeformat='unixtime')


def unavailable_weather(reason, location=None):
    return dict(available=False, source='Open-Meteo weather model', scenario='live_weather_scenario',
                scenario_label=LIVE_SCENARIO, location=deepcopy(location or WEATHER_LOCATION),
                observed_at=None, checked_at=utc_now().isoformat(), errors=[str(reason)])


def parse_weather(payload, location, received_at):
    """Retain the actual interval and validate the units returned by Open-Meteo."""
    if not isinstance(payload, dict) or not isinstance(payload.get('current'), dict) or not isinstance(payload.get('current_units'), dict):
        raise ValueError('Weather response must contain current values and units.')
    current, units = payload['current'], payload['current_units']
    if units.get('time') != 'unixtime':
        raise ValueError('Weather timestamp units must be unixtime.')
    timestamp = current['time']
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError('Weather timestamp is missing or invalid.')
    observed = datetime.fromtimestamp(timestamp, timezone.utc)
    interval = current['interval']
    if isinstance(interval, bool) or interval != WEATHER_EXPECTED_INTERVAL_SECONDS:
        raise ValueError(f'Expected {WEATHER_EXPECTED_INTERVAL_SECONDS}-second current-weather interval.')
    grid = {key: float(payload[key]) for key in ('latitude', 'longitude')}
    for key in grid:
        if not math.isfinite(grid[key]) or abs(grid[key] - location[key]) > WEATHER_LOCATION_TOLERANCE_DEGREES:
            raise ValueError('Weather response location does not match the requested area.')
    result = dict(available=True, source='Open-Meteo weather model', scenario='live_weather_scenario',
                  scenario_label=LIVE_SCENARIO, location=deepcopy(location), grid_location=grid,
                  observed_at=observed.isoformat(), fetched_at=received_at.isoformat(),
                  checked_at=received_at.isoformat(), interval_seconds=interval, units={}, errors=[])
    result['reported_units'] = deepcopy(units)
    for api_field, (name, expected_unit) in FIELDS.items():
        value = current.get(api_field)
        actual_unit = str(units.get(api_field, '')).casefold()
        if api_field == 'wind_speed_10m' and actual_unit == 'mp/h':
            actual_unit = 'mph'  # Open-Meteo's spelling for the requested miles/hour unit.
        if value is not None and actual_unit != expected_unit.casefold():
            raise ValueError(f'Unexpected units for {api_field}; expected {expected_unit}.')
        result[name] = value
        result['units'][name] = expected_unit
    return result


class OpenMeteoClient:
    def __init__(self, location=None, *, timeout=WEATHER_TIMEOUT, cache_seconds=WEATHER_CACHE_SECONDS):
        self.location = deepcopy(location or WEATHER_LOCATION)
        self.timeout = timeout
        self.cache_seconds = cache_seconds
        self._snapshot = None
        self._cached_at = None

    def get_current_weather(self, *, force=False):
        now, tick = utc_now(), monotonic()
        if force or self._snapshot is None or tick - self._cached_at >= self.cache_seconds:
            try:
                response = requests.get(URL, params=request_params(self.location), timeout=self.timeout)
                response.raise_for_status()
                snapshot = parse_weather(response.json(), self.location, now)
            except (requests.RequestException, KeyError, TypeError, ValueError, OverflowError, OSError) as error:
                # A failed refresh must never masquerade as fresh clear weather.
                snapshot = unavailable_weather(error, self.location)
            self._snapshot, self._cached_at = snapshot, tick
        snapshot = deepcopy(self._snapshot)
        snapshot['checked_at'] = now.isoformat()  # Re-evaluate freshness on every cache hit.
        return snapshot


_client = OpenMeteoClient()


def get_current_weather():
    return _client.get_current_weather()


if __name__ == '__main__':
    print(json.dumps(get_current_weather(), indent=2))
