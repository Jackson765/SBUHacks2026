"""Open-Meteo schema, bounded failures, and cache tests; always offline."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_code
from decision_engine import evaluate_advisory
from test_decision_engine import NOW


def api_payload(**changes):
    current = dict(time=int(NOW.timestamp()), interval=900, temperature_2m=60.,
                   relative_humidity_2m=50., apparent_temperature=60., precipitation=0.,
                   rain=0., snowfall=0., weather_code=0, wind_speed_10m=5., visibility=10000.)
    current.update(changes)
    units = {field: unit for field, (_, unit) in api_code.FIELDS.items()}
    units['time'] = 'unixtime'
    return dict(latitude=40.91, longitude=-73.14, current=current, current_units=units)


class WeatherApiTests(unittest.TestCase):
    def setUp(self):
        network = patch('api_code.requests.get')
        self.get = network.start()
        self.addCleanup(network.stop)
        clock = patch('api_code.utc_now', return_value=NOW)
        clock.start()
        self.addCleanup(clock.stop)
        self.get.return_value = Mock(json=Mock(return_value=api_payload()))
        self.client = api_code.OpenMeteoClient()

    def test_requests_explicit_units_visibility_and_utc_time_once_with_bounded_timeout(self):
        weather = self.client.get_current_weather()
        self.assertTrue(weather['available'])
        self.assertEqual(weather['observed_at'], NOW.isoformat())
        self.assertEqual(weather['interval_seconds'], 900)
        self.assertEqual(weather['units']['snowfall_cm'], 'cm')
        self.assertEqual(weather['scenario'], 'live_weather_scenario')
        args = self.get.call_args.kwargs
        self.assertEqual(args['timeout'], (2., 3.))
        self.assertEqual(args['params']['precipitation_unit'], 'mm')
        self.assertEqual(args['params']['timeformat'], 'unixtime')
        self.assertIn('visibility', args['params']['current'])

    def test_open_meteo_miles_per_hour_spelling_is_normalized_without_changing_values(self):
        payload = api_payload()
        payload['current_units']['wind_speed_10m'] = 'mp/h'
        self.get.return_value.json.return_value = payload
        weather = self.client.get_current_weather()
        self.assertTrue(weather['available'])
        self.assertEqual(weather['wind_speed_mph'], 5)
        self.assertEqual(weather['units']['wind_speed_mph'], 'mph')
        self.assertEqual(weather['reported_units']['wind_speed_10m'], 'mp/h')

    def test_cache_reuses_snapshot_but_rechecks_freshness_and_protects_cached_values(self):
        with patch('api_code.monotonic', side_effect=[0, 1, 2]):
            first = self.client.get_current_weather()
            first['temperature_f'] = -99
            for _ in range(2):
                with patch('api_code.utc_now', return_value=NOW+timedelta(hours=1)):
                    cached = self.client.get_current_weather()
                decision = evaluate_advisory(60, 65, 65, cached, 'freeway')
                self.assertEqual(decision['status'], 'partial_data')
                self.assertEqual(cached['temperature_f'], 60)
        self.get.assert_called_once()

    def test_timeout_http_and_bad_json_are_cached_unavailable_not_clear(self):
        for error in [requests.Timeout('timeout'), requests.HTTPError('HTTP error'), ValueError('bad JSON')]:
            with self.subTest(error=error):
                client = api_code.OpenMeteoClient()
                self.get.side_effect = error
                self.assertFalse(client.get_current_weather()['available'])
                self.assertFalse(client.get_current_weather()['available'])
        self.assertEqual(self.get.call_count, 3)

    def test_wrong_units_interval_location_and_shape_are_unavailable(self):
        payloads = []
        for field, value in [('rain', 'inch'), ('temperature_2m', '°C')]:
            payload = api_payload()
            payload['current_units'][field] = value
            payloads.append(payload)
        payloads += [api_payload(interval=3600), api_payload(time=None), {'current': None}, []]
        mismatch = api_payload()
        mismatch['latitude'] = 34.05
        payloads.append(mismatch)
        for payload in payloads:
            self.get.return_value.json.return_value = payload
            self.assertFalse(api_code.OpenMeteoClient().get_current_weather()['available'])

    def test_expired_good_cache_is_not_silently_used_after_failed_refresh(self):
        with patch('api_code.monotonic', side_effect=[0, 601]):
            self.assertTrue(self.client.get_current_weather()['available'])
            self.get.side_effect = requests.Timeout('refresh timeout')
            failed = self.client.get_current_weather()
        self.assertFalse(failed['available'])
        self.assertIn('refresh timeout', failed['errors'][0])

    def test_missing_critical_fields_are_reported_unavailable_by_decision(self):
        self.get.return_value.json.return_value = api_payload(temperature_2m=None)
        result = evaluate_advisory(60, 65, 65, self.client.get_current_weather(), 'freeway')
        self.assertEqual(result['status'], 'partial_data')
        self.assertIsNone(result['advisory_speed_mph'])

if __name__ == '__main__':
    unittest.main()
