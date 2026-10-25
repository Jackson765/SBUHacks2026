"""Pure experimental advisory rules; no networking, clocks, or hardware control."""
from datetime import datetime
import math

from config import ADVISORY_POLICY, WEATHER_EXPECTED_INTERVAL_SECONDS

NORMAL_CODES = {0, 1, 2, 3}
RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}
SEVERE_CODES = {45, 48, 56, 57, 66, 67, 71, 73, 75, 77, 85, 86, 95, 96, 99}
EXPECTED_UNITS = {"temperature_f": "°F", "precipitation_mm": "mm", "rain_mm": "mm",
                  "snowfall_cm": "cm", "weather_code": "wmo code", "visibility_m": "m"}


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def timestamp(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Weather timestamps need an explicit UTC offset.")
    return parsed


def classify_weather(weather, limit, policy):
    """Evaluate a snapshot as of its explicitly supplied checked_at timestamp."""
    warnings, hazards = [], []
    if not isinstance(weather, dict) or not weather.get("available"):
        errors = weather.get("errors", []) if isinstance(weather, dict) else []
        if not isinstance(errors, (list, tuple)):
            errors = [errors]
        return "unavailable", None, [], ["Weather unavailable: " + ("; ".join(map(str, errors)) or "no observation")]
    if weather.get("scenario") not in {"live_weather_scenario", "simulated"}:
        return "unavailable", None, [], ["Unverified weather time/location association; use an explicitly labeled scenario."]
    location = weather.get("location", {})
    if not isinstance(location, dict):
        return "unavailable", None, [], ["Weather location is missing or invalid."]
    lat, lon = number(location.get("latitude")), number(location.get("longitude"))
    try:
        age = (timestamp(weather["checked_at"]) - timestamp(weather["observed_at"])).total_seconds()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return "unavailable", None, [], ["Weather observation timestamp is missing or invalid."]
    if (lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180
            or age > policy["weather_max_age_seconds"] or age < -policy["future_tolerance_seconds"]):
        return "unavailable", None, [], ["Weather is stale, future-dated, or missing a valid location."]
    if number(weather.get("interval_seconds")) != WEATHER_EXPECTED_INTERVAL_SECONDS:
        return "unavailable", None, [], ["Weather accumulation interval is missing or incompatible."]
    values, errors = {}, []
    units = weather.get("units")
    if not isinstance(units, dict):
        return "unavailable", None, [], ["Weather units are missing."]
    for field, unit in EXPECTED_UNITS.items():
        value = number(weather.get(field))
        if field == "visibility_m" and weather.get(field) is None:
            warnings.append("Visibility unavailable; humidity alone does not establish or rule out fog.")
            continue
        if value is None or units.get(field) != unit or (field != "temperature_f" and value < 0):
            errors.append(f"Weather field {field} is missing, invalid, or has incompatible units.")
            continue
        values[field] = value
    code = values.get("weather_code")
    if code is None or not code.is_integer() or code not in NORMAL_CODES | RAIN_CODES | SEVERE_CODES:
        errors.append("Weather code is unknown; clear conditions cannot be assumed.")
    if values.get("snowfall_cm", 0) > 0 or code in {71, 73, 75, 77, 85, 86}:
        hazards.append("Snow/snowfall is indicated.")
    if code in {56, 57, 66, 67} or (max(values.get("precipitation_mm", 0), values.get("rain_mm", 0)) > 0 and values.get("temperature_f", math.inf) <= policy["freezing_temperature_f"]):
        hazards.append("Possible freezing precipitation; air temperature does not prove road ice.")
    if code in {95, 96, 99}:
        hazards.append("Thunderstorm/hail is indicated.")
    if code in {45, 48} or values.get("visibility_m", math.inf) < policy["low_visibility_m"]:
        hazards.append("Fog or very low visibility is indicated.")
    if weather["scenario"] == "live_weather_scenario":
        warnings.append("LIVE WEATHER SCENARIO: current model weather is not historically matched to this prerecorded video.")
    else:
        warnings.append("SIMULATED WEATHER SCENARIO: supplied weather is not an observation of the video.")
    warnings.append("Weather-model data do not measure roadway friction.")
    if hazards:
        cap = limit - policy["severe_weather_reduction_mph"]
        return "severe", cap, hazards + [f"Provisional demo weather cap {cap:g} mph; manual review required."], warnings + errors
    if errors:
        return "unavailable", None, [], warnings + errors
    if code in RAIN_CODES or values["rain_mm"] > 0 or values["precipitation_mm"] > 0:
        cap = limit - policy["rain_reduction_mph"]
        return "adverse", cap, [f"Rain/precipitation rule caps the demo advisory at {cap:g} mph."], warnings
    return "normal", limit, ["Dry model conditions; no recognized weather hazard in the available fields."], warnings


def evaluate_advisory(predicted_speed_mph, free_flow_speed_mph, posted_limit_mph, weather,
                      road_type=None, *, policy=None, prediction_age_seconds=0,
                      prediction_units="mph", prediction_calibrated=True,
                      legal_max_speed_mph=None):
    """Consume final local mph without invoking Model B or consulting a clock.

    ``posted_limit_mph`` is the configured base reference. When supplied,
    ``legal_max_speed_mph`` is the hard recommendation ceiling. Weather includes
    explicit units/location/observed_at/checked_at; callers supply a fresh
    checked_at (the API client does so even on cache hits).
    """
    settings = dict(ADVISORY_POLICY)
    if policy is not None:
        settings.update(policy)
    result = dict(traffic_ratio=None, traffic_cap_mph=None, weather_category="unavailable",
                  weather_cap_mph=None, candidate_speed_mph=None, advisory_speed_mph=None,
                  status="insufficient_data", controlling_factor="unavailable", reasons=[],
                  warnings=[])
    p, f, base_limit = map(number, (predicted_speed_mph, free_flow_speed_mph, posted_limit_mph))
    legal_limit = number(legal_max_speed_mph) if legal_max_speed_mph is not None else base_limit
    age = number(prediction_age_seconds)
    for key in settings:
        if key != "allow_unsupported_road_demo":
            value = number(settings[key])
            if value is None:
                result["warnings"].append("Invalid demo policy configuration.")
                return result
            settings[key] = value
    if (not isinstance(settings["allow_unsupported_road_demo"], bool)
            or not 0 < settings["moderate_flow_ratio"] < settings["normal_flow_ratio"] <= 1
            or settings["increment_mph"] != 5
            or settings["minimum_supported_limit_mph"] <= 0 or settings["low_visibility_m"] <= 0
            or any(settings[k] < 0 for k in settings if k.endswith(("reduction_mph", "age_seconds", "tolerance_seconds")))):
        result["warnings"].append("Invalid demo policy configuration.")
        return result
    if (p is None or f is None or base_limit is None or legal_limit is None
            or p < 0 or f <= 0 or base_limit <= 0 or legal_limit < base_limit
            or age is None or age < 0 or age > settings["prediction_max_age_seconds"]
            or prediction_units != "mph" or not prediction_calibrated):
        result["warnings"].append("Awaiting a finite, nonnegative, fresh, calibrated local-mph prediction and positive road parameters.")
        return result
    result.update(base_speed_limit_mph=base_limit, legal_max_speed_mph=legal_limit)
    ratio = p / f
    if not math.isfinite(ratio):
        result["warnings"].append("Traffic ratio overflow; check road parameters.")
        return result
    if ratio >= settings["normal_flow_ratio"]:
        reduction, condition = settings["normal_reduction_mph"], "normal flow"
    elif ratio >= settings["moderate_flow_ratio"]:
        reduction, condition = settings["moderate_reduction_mph"], "moderate predicted slowdown"
    else:
        reduction, condition = settings["severe_reduction_mph"], "severe predicted slowdown"
    traffic_reference = min(f, legal_limit) if legal_max_speed_mph is not None else base_limit
    traffic_cap = traffic_reference - reduction
    result.update(traffic_ratio=ratio, traffic_cap_mph=traffic_cap)
    result["reasons"] += [f"Predicted speed {p:.1f} mph is {ratio:.0%} of {f:.1f} mph normal flow ({condition}).",
                          f"Traffic rule caps the demo advisory at {traffic_cap:g} mph."]
    category, cap, reasons, warnings = classify_weather(weather, legal_limit, settings)
    result.update(weather_category=category, weather_cap_mph=cap)
    result["reasons"] += reasons
    result["warnings"] += warnings
    unsupported = legal_limit < settings["minimum_supported_limit_mph"] or road_type not in {"freeway", "motorway"}
    if unsupported:
        result["warnings"].append("Unsupported/unknown road: freeway-oriented demo reductions are not validated for local or signalized roads.")
    if category == "severe" or (unsupported and not settings["allow_unsupported_road_demo"]) or traffic_cap < 5 or (cap is not None and cap < 5):
        result.update(status="manual_review", controlling_factor="weather" if category == "severe" else "unavailable")
        result["warnings"].append("Manual review required; no automatic advisory is supplied.")
        return result
    if unsupported:
        result["warnings"].append("EXPLICIT UNSUPPORTED-ROAD DEMO OVERRIDE - illustrative only, not a safe-road policy.")
    if category == "unavailable":
        result.update(status="partial_data", candidate_speed_mph=min(legal_limit, traffic_cap), controlling_factor="unavailable")
        result["reasons"].append("Traffic-only candidate; weather unavailable, so no combined advisory is supplied.")
        return result
    candidate = min(legal_limit, traffic_cap, cap)
    factor = "none"
    if traffic_cap < legal_limit or cap < legal_limit:
        factor = "both" if traffic_cap == cap else "traffic" if traffic_cap < cap else "weather"
    advisory = math.floor(candidate / settings["increment_mph"]) * settings["increment_mph"]
    result.update(candidate_speed_mph=candidate, advisory_speed_mph=advisory,
                  status="demo_advisory", controlling_factor=factor)
    result["reasons"].append(
        f"Base speed {base_limit:g} mph; recommendations are capped at the verified {legal_limit:g} mph legal maximum."
    )
    result["reasons"].append(f"{factor.capitalize()} controls the {advisory:g}-mph demo advisory; quantized downward to 5 mph.")
    return result
