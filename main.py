"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only): an empirical launch-shape forecaster.

The model, per case:

  - The anchor is hour 1's print p1 (the case's `known` row for hour 1; a print below 1 counts as 1).
  - The shape comes from the case's own `comparables`: every past launch whose release day is fully enough
    observed contributes its release-day curve as log ratios ln(v_h / v_1) for h = 2..24, aligned on the UTC
    release day. Launches whose hour 1 is missing, or is a sliver of their day's peak (their unlock came later
    in the UTC day, so hour 1 is not a launch hour at all), are left out, as is the one comparable that is not
    a launch (the target category's own pre-release history).
  - Each comparable is weighted by how close its hour-1 size is to this case's (log distance) and by how many
    of the day's hours it has. The median curve is the weighted median of the log ratios per hour, lightly
    smoothed over neighbouring hours.
  - The spread is the weighted dispersion of the comparables' log ratios around that median per hour,
    expressed as log-normal quantile offsets z_tau * s_h with s_h floored and capped; it widens with the hour,
    as the launches disagree more about later hours.
  - A small, damped tilt comes from the case's own recorded window: if the reads inside hour 1 are minute
    reads (the final) and the hour's second half runs hotter than its first, the next few hours are lifted a
    little (and the reverse when it fades); the tilt decays hour by hour and is capped, so it cannot explode.
    With hourly points only (the practice) there is no such signal and the tilt is 0.

Quantiles are q_tau(h) = p1 * exp(m_h + z_tau * s_h), rounded to 2 decimals (half up), clamped to (0, 1e9]
and made non-decreasing. The forecast is a deterministic function of the case file: no randomness, no
network, nothing read but /input (or $MAKEFUN_INPUT_DIR) and nothing written but /output (or
$MAKEFUN_OUTPUT_DIR).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard
library only. -P keeps this file's directory OFF sys.path; nothing here imports a sibling module, so that is
fine. Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and
/output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (chosen on the comparables' own launch days, leave-one-out) ---
# A comparable counts only if its hour 1 is at least this share of its release day's peak hour: below that,
# its hour 1 fell before its unlock and its ratios measure the unlock clock, not the day's shape.
MIN_HOUR1_SHARE = 0.05
# A comparable counts only if at least this many of hours 2..24 are present.
MIN_DAY_HOURS = 18
# Weight by log distance between the comparable's hour 1 and this case's: w = exp(-|ln(v1/p1)| / SIZE_SCALE),
# floored at SIZE_FLOOR. A scale near 1 concentrates the vote on launches of a similar size, whose release-day
# shapes proved the most informative; distant launches still vote, faintly.
SIZE_SCALE = 1.0
SIZE_FLOOR = 0.0
# The median log-ratio curve is shrunk toward 0 (persistence) by this factor. Leave-one-out over the
# comparables put the best value at 0: the empirical curve, taken at face value, beat every shrunk version.
SHRINK = 0.0
# Whether the median curve is smoothed over neighbouring hours before shrinkage.
SMOOTH = True
# The spread s_h (log units) is floored and capped.
SPREAD_FLOOR = 0.16
SPREAD_CAP = 0.95
# Within-hour-1 momentum tilt: capped log tilt at hour 2, decaying by TILT_DECAY each hour.
TILT_CAP = 0.30
TILT_DECAY = 0.62


def parse_ts(text: str) -> datetime:
    """An ISO 8601 UTC timestamp from the case file, as an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """The weighted quantile of (value, weight) pairs: the first value whose cumulative weight reaches tau."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= target:
            return value
    return ordered[-1][0]


def comparable_curve(comp: dict) -> tuple[float, dict[int, float]] | None:
    """A comparable's release-day (hour-1 viewers, {hour: log ratio to hour 1}), or None when unusable.

    Unusable: hour 1 missing or not positive, too few of the day's hours present, hour 1 a sliver of the
    day's peak (a late UTC unlock), or the comparable is not a launch at all.
    """
    if str(comp.get("unlock", "")).startswith("Not a launch"):
        return None
    day_start = parse_ts(comp["day_start"])
    by_hour: dict[int, float] = {}
    for point in comp["hourly"]:
        when = parse_ts(point["from"])
        delta = (when - day_start).total_seconds()
        if delta % 3600 == 0:
            hour = int(delta // 3600) + 1
            if 1 <= hour <= 24 and point["viewers"] and point["viewers"] > 0:
                by_hour[hour] = float(point["viewers"])
    v1 = by_hour.get(1)
    if v1 is None or len(by_hour) - 1 < MIN_DAY_HOURS:
        return None
    if v1 < MIN_HOUR1_SHARE * max(by_hour.values()):
        return None
    return v1, {h: math.log(v / v1) for h, v in by_hour.items() if h >= 2}


def shape_from_comparables(case: dict, p1: float) -> tuple[dict[int, float], dict[int, list[float]]]:
    """(centre log ratio, five log offsets) per forecast hour, from the case's comparables."""
    samples: dict[int, list[tuple[float, float]]] = {h: [] for h in range(2, 25)}
    for comp in case["comparables"]:
        curve = comparable_curve(comp)
        if curve is None:
            continue
        v1, log_ratios = curve
        size_w = max(SIZE_FLOOR, math.exp(-abs(math.log(v1 / p1)) / SIZE_SCALE))
        weight = size_w * (len(log_ratios) / 23.0)
        for hour, log_ratio in log_ratios.items():
            samples[hour].append((log_ratio, weight))
    centres: dict[int, float] = {}
    offsets: dict[int, list[float]] = {}
    for hour in range(2, 25):
        pairs = samples[hour]
        if not pairs:
            centres[hour] = 0.0
            offsets[hour] = [z * 0.35 for z in Z]
            continue
        centres[hour] = weighted_quantile(pairs, 0.5)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, (q75 - q25) / 1.349))
        offsets[hour] = [z * spread for z in Z]
    # Light smoothing of the centre curve over neighbouring hours, then shrinkage toward persistence.
    if SMOOTH:
        smoothed: dict[int, float] = {}
        for hour in range(2, 25):
            neighbours = [centres[h] for h in (hour - 1, hour, hour + 1) if h in centres]
            smoothed[hour] = sum(neighbours) / len(neighbours)
        centres = smoothed
    for hour in range(2, 25):
        centres[hour] = (1.0 - SHRINK) * centres[hour]
    return centres, offsets


def hour1_momentum(case: dict) -> float:
    """The within-hour-1 log trend from the case's history: second half of hour 1 vs first half.

    Only minute-spaced reads carry this signal (the final's recorder); with hourly points (the practice)
    hour 1 holds a single point and the trend is 0.
    """
    day_start = parse_ts(case["day_start"])
    checkpoint = parse_ts(case["checkpoint"])
    inside = [
        (parse_ts(point["at"]), float(point["viewers"]))
        for point in case["history"]
        if day_start <= parse_ts(point["at"]) < checkpoint and point["viewers"] and point["viewers"] > 0
    ]
    if len(inside) < 20:
        return 0.0
    inside.sort()
    mid = inside[len(inside) // 2][0]
    first = [v for t, v in inside if t < mid]
    second = [v for t, v in inside if t >= mid]
    if not first or not second:
        return 0.0
    trend = math.log((sum(second) / len(second)) / (sum(first) / len(first)))
    return max(-TILT_CAP, min(TILT_CAP, trend))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    centres, offsets = shape_from_comparables(case, p1)
    tilt = hour1_momentum(case)
    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        centre = centres[hour] + tilt * (TILT_DECAY ** (hour - 2))
        quantiles = []
        for off in offsets[hour]:
            value = p1 * math.exp(centre + off)
            value = min(1e9, max(0.01, value))
            quantiles.append(round2(value))
        # Rounding must not break the order: hold each quantile at least at the one before it.
        for j in range(1, len(quantiles)):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
