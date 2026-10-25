"""Offline acceptance cases for illustrative traffic/weather policy only."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision_engine import EXPECTED_UNITS, evaluate_advisory

NOW = datetime(2026, 10, 24, 20, tzinfo=timezone.utc)


def observation(**changes):
    weather = dict(available=True, source='test fixture', scenario='simulated',
                   location=dict(name='Simulated road location', latitude=40.91, longitude=-73.14),
                   observed_at=NOW.isoformat(), checked_at=NOW.isoformat(), interval_seconds=900,
                   units=dict(EXPECTED_UNITS), temperature_f=60., precipitation_mm=0.,
                   rain_mm=0., snowfall_cm=0., weather_code=0, visibility_m=10000., humidity_percent=95)
    weather.update(changes)
    return weather


class DecisionEngineTests(unittest.TestCase):
    def decide(self, prediction=35, weather=None, *, limit=65, free_flow=65, road='freeway', **kwargs):
        return evaluate_advisory(prediction, free_flow, limit,
                                 observation() if weather is None else weather, road, **kwargs)

    def test_four_numeric_worked_examples_and_both_controlling(self):
        cases = [(35, True, 45, 55, 45, 'traffic'), (60, False, 65, 65, 65, 'none'),
                 (45, False, 55, 65, 55, 'traffic'), (60, True, 65, 55, 55, 'weather'),
                 (45, True, 55, 55, 55, 'both')]
        for p, rain, traffic, weather_cap, advisory, factor in cases:
            with self.subTest(prediction=p, rain=rain):
                weather = observation(weather_code=61, rain_mm=.1, precipitation_mm=.1) if rain else observation()
                result = self.decide(p, weather)
                self.assertAlmostEqual(result['traffic_ratio'], p/65)
                self.assertEqual(result['traffic_cap_mph'], traffic)
                self.assertEqual(result['weather_cap_mph'], weather_cap)
                self.assertEqual(result['advisory_speed_mph'], advisory)
                self.assertEqual(result['controlling_factor'], factor)
                self.assertEqual(result['status'], 'demo_advisory')

    def test_severe_snow_freezing_thunderstorm_and_visibility_need_manual_review(self):
        for changes in [dict(snowfall_cm=2, weather_code=75),
                        dict(temperature_f=32, precipitation_mm=.1, rain_mm=.1),
                        dict(weather_code=66), dict(weather_code=95), dict(weather_code=96),
                        dict(weather_code=99), dict(visibility_m=499), dict(weather_code=45)]:
            with self.subTest(changes=changes):
                result = self.decide(weather=observation(**changes))
                self.assertEqual(result['status'], 'manual_review')
                self.assertEqual(result['weather_category'], 'severe')
                self.assertIsNone(result['advisory_speed_mph'])
                self.assertIsNone(result['candidate_speed_mph'])
                self.assertTrue(result['warnings'])

    def test_known_hazard_is_not_hidden_by_other_missing_fields(self):
        result = self.decide(weather=observation(temperature_f=None, weather_code=95))
        self.assertEqual(result['status'], 'manual_review')
        self.assertEqual(result['weather_category'], 'severe')
        self.assertIsNone(result['advisory_speed_mph'])

    def test_timeout_keeps_explicit_traffic_only_candidate(self):
        result = self.decide(weather=dict(available=False, errors=['API timeout']))
        self.assertEqual(result['traffic_cap_mph'], 45)
        self.assertEqual(result['candidate_speed_mph'], 45)
        self.assertEqual(result['status'], 'partial_data')
        self.assertIsNone(result['advisory_speed_mph'])
        self.assertEqual(result['weather_category'], 'unavailable')

    def test_missing_invalid_stale_or_wrong_unit_prediction_is_insufficient(self):
        for p in (None, float('nan'), float('inf'), -1, True):
            self.assertEqual(self.decide(p)['status'], 'insufficient_data')
        for f in (0, -1, None, float('inf')):
            self.assertEqual(self.decide(free_flow=f)['status'], 'insufficient_data')
        for kwargs in [dict(prediction_age_seconds=301), dict(prediction_units='km/h'),
                       dict(prediction_calibrated=False)]:
            result = self.decide(**kwargs)
            self.assertEqual(result['status'], 'insufficient_data')
            self.assertIsNone(result['advisory_speed_mph'])

    def test_low_signalized_and_unknown_roads_need_review_unless_explicit_demo_override(self):
        for limit, road in [(30, 'freeway'), (65, 'signalized_road'), (45, None), (65, 'unknown')]:
            result = self.decide(limit=limit, road=road)
            self.assertEqual(result['status'], 'manual_review')
            self.assertIsNone(result['advisory_speed_mph'])
        result = self.decide(limit=30, road='signalized_road', policy={'allow_unsupported_road_demo': True})
        self.assertEqual(result['status'], 'demo_advisory')
        self.assertTrue(any('OVERRIDE' in warning for warning in result['warnings']))
        self.assertEqual(self.decide(limit=10)['status'], 'manual_review')

    def test_threshold_boundaries_and_downward_quantization_never_exceed_caps(self):
        self.assertEqual(self.decide(52)['traffic_cap_mph'], 65)
        self.assertEqual(self.decide(39)['traffic_cap_mph'], 55)
        self.assertEqual(self.decide(38.999)['traffic_cap_mph'], 45)
        result = self.decide(100, limit=63)
        self.assertEqual(result['candidate_speed_mph'], 63)
        self.assertEqual(result['advisory_speed_mph'], 60)
        for p in range(0, 101, 5):
            for limit in (45, 47, 60, 63, 65):
                result = self.decide(p, limit=limit)
                self.assertLessEqual(result['advisory_speed_mph'], min(limit, result['traffic_cap_mph'], result['weather_cap_mph']))
                self.assertEqual(result['advisory_speed_mph'] % 5, 0)

    def test_verified_legal_max_allows_normal_traffic_advisory_above_base(self):
        result = self.decide(50, limit=30, free_flow=55, legal_max_speed_mph=70)
        self.assertEqual(result['traffic_cap_mph'], 55)
        self.assertEqual(result['weather_cap_mph'], 70)
        self.assertEqual(result['candidate_speed_mph'], 55)
        self.assertEqual(result['advisory_speed_mph'], 55)
        self.assertEqual(result['base_speed_limit_mph'], 30)
        self.assertEqual(result['legal_max_speed_mph'], 70)

        rain = observation(weather_code=61, rain_mm=.1, precipitation_mm=.1)
        result = self.decide(50, weather=rain, limit=30, free_flow=55, legal_max_speed_mph=70)
        self.assertEqual(result['advisory_speed_mph'], 55)
        self.assertLessEqual(result['advisory_speed_mph'], result['weather_cap_mph'])

    def test_legal_maximum_is_a_hard_cap_and_cannot_be_below_base(self):
        result = self.decide(100, limit=60, free_flow=80, legal_max_speed_mph=70)
        self.assertEqual(result['advisory_speed_mph'], 70)
        invalid = self.decide(50, limit=60, free_flow=80, legal_max_speed_mph=55)
        self.assertEqual(invalid['status'], 'insufficient_data')
        self.assertIsNone(invalid['advisory_speed_mph'])

    def test_missing_stale_incompatible_weather_never_becomes_clear(self):
        cases = [dict(temperature_f=None), dict(weather_code=999), dict(precipitation_mm=float('nan')),
                 dict(location=None), dict(units=None), dict(interval_seconds=3600),
                 dict(observed_at=(NOW-timedelta(hours=1)).isoformat()),
                 dict(observed_at=(NOW+timedelta(hours=1)).isoformat()),
                 dict(observed_at='2026-10-24T20:00:00'), dict(scenario='historically_matched')]
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.decide(weather=observation(**changes))
                self.assertEqual(result['weather_category'], 'unavailable')
                self.assertEqual(result['status'], 'partial_data')
                self.assertIsNone(result['advisory_speed_mph'])

    def test_live_weather_is_explicit_scenario_humidity_alone_is_not_fog(self):
        weather = observation(scenario='live_weather_scenario', humidity_percent=100, visibility_m=None)
        original = deepcopy(weather)
        result = self.decide(60, weather)
        self.assertEqual(result['weather_category'], 'normal')
        self.assertTrue(any('not historically matched' in w for w in result['warnings']))
        self.assertTrue(any('humidity alone' in w for w in result['warnings']))
        self.assertEqual(weather, original)

    def test_invalid_policy_fails_closed(self):
        for changes in [dict(normal_flow_ratio=.5), dict(rain_reduction_mph=-10),
                        dict(increment_mph=10), dict(low_visibility_m=float('nan')),
                        dict(allow_unsupported_road_demo='yes')]:
            self.assertEqual(self.decide(policy=changes)['status'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
