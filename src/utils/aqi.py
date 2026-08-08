"""
Indian CPCB National Air Quality Index (AQI) computation.

Single source of truth used by BOTH training (label + features) and serving
(app.py), so the number the model learns and the number shown to users are
computed the exact same way.

CPCB methodology (2014):
- Each pollutant sub-index is a piecewise-linear function of a TIME-AVERAGED
  concentration: 24h average for PM2.5, PM10, NO2, SO2, NH3; 8h average for
  CO and O3.
- The overall AQI is the MAX of the available sub-indices.
- An AQI is only valid when at least 3 pollutants are present and at least one
  of them is PM2.5 or PM10.

Units: Open-Meteo reports every pollutant in ug/m3. CPCB CO breakpoints are in
mg/m3, so CO must be divided by 1000 before calling into this module (helpers
below do this for you via `CO_UGM3_TO_MGM3`).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Breakpoints as (C_lo, C_hi, I_lo, I_hi). Concentrations in ug/m3 except CO
# which is in mg/m3. Values are rounded (CO to 1 decimal, others to integer)
# before lookup, matching CPCB's integer-band convention.
CPCB_BREAKPOINTS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "pm2_5": [
        (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
        (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
        (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500),
    ],
    "nitrogen_dioxide": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
        (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500),
    ],
    "ozone": [
        (0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
        (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500),
    ],
    "carbon_monoxide": [  # mg/m3
        (0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
        (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500),
    ],
    "sulphur_dioxide": [
        (0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
        (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500),
    ],
    "ammonia": [
        (0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
        (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 2000, 401, 500),
    ],
}

# Averaging window (hours) per pollutant, per CPCB.
CPCB_AVG_WINDOW: Dict[str, int] = {
    "pm2_5": 24, "pm10": 24, "nitrogen_dioxide": 24,
    "sulphur_dioxide": 24, "ammonia": 24, "carbon_monoxide": 8, "ozone": 8,
}

PM_POLLUTANTS = ("pm2_5", "pm10")
CO_UGM3_TO_MGM3 = 1.0 / 1000.0

CPCB_CATEGORIES = [
    (50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
    (300, "Poor"), (400, "Very Poor"), (float("inf"), "Severe"),
]


def _decimals_for(pollutant: str) -> int:
    return 1 if pollutant == "carbon_monoxide" else 0


def sub_index(conc: Optional[float], breakpoints: List[Tuple[float, float, float, float]],
              decimals: int = 0) -> float:
    """Scalar CPCB sub-index for one pollutant's averaged concentration."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return float("nan")
    c = round(max(float(conc), 0.0), decimals)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo
    # Above the top band -> capped at 500.
    return 500.0


def cpcb_aqi(averages: Dict[str, float]) -> float:
    """
    Overall CPCB AQI from a dict of AVERAGED concentrations.

    Keys must match CPCB_BREAKPOINTS (ug/m3, except carbon_monoxide in mg/m3).
    Returns NaN when fewer than 3 sub-indices are available or no PM is present.
    """
    subs: List[float] = []
    has_pm = False
    for pollutant, conc in averages.items():
        bps = CPCB_BREAKPOINTS.get(pollutant)
        if bps is None:
            continue
        si = sub_index(conc, bps, _decimals_for(pollutant))
        if not math.isnan(si):
            subs.append(si)
            if pollutant in PM_POLLUTANTS:
                has_pm = True
    if len(subs) < 3 or not has_pm:
        return float("nan")
    return max(subs)


def sub_index_array(conc: np.ndarray, breakpoints: List[Tuple[float, float, float, float]],
                    decimals: int = 0) -> np.ndarray:
    """Vectorised CPCB sub-index over an array of averaged concentrations."""
    c = np.round(np.clip(np.asarray(conc, dtype="float64"), 0.0, None), decimals)
    out = np.full(c.shape, np.nan)
    valid = ~np.isnan(np.asarray(conc, dtype="float64"))
    assigned = np.zeros(c.shape, dtype=bool)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = valid & ~assigned & (c >= c_lo) & (c <= c_hi)
        out[mask] = (i_hi - i_lo) / (c_hi - c_lo) * (c[mask] - c_lo) + i_lo
        assigned |= mask
    over = valid & ~assigned & (c > breakpoints[-1][1])
    out[over] = 500.0
    return out


def cpcb_aqi_array(averages: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Vectorised overall CPCB AQI. `averages` maps pollutant -> array of averaged
    concentrations (CO in mg/m3). Returns NaN where the CPCB validity rule fails.
    """
    sub_arrays = {
        p: sub_index_array(arr, CPCB_BREAKPOINTS[p], _decimals_for(p))
        for p, arr in averages.items() if p in CPCB_BREAKPOINTS
    }
    stack = np.vstack(list(sub_arrays.values()))
    valid_count = np.sum(~np.isnan(stack), axis=0)
    pm_present = np.zeros(stack.shape[1], dtype=bool)
    for p in PM_POLLUTANTS:
        if p in sub_arrays:
            pm_present |= ~np.isnan(sub_arrays[p])
    with np.errstate(invalid="ignore"):
        aqi = np.nanmax(stack, axis=0)
    ok = (valid_count >= 3) & pm_present
    return np.where(ok, aqi, np.nan)


def cpcb_aqi_from_ugm3(avg_ugm3: Dict[str, float]) -> float:
    """Scalar CPCB AQI from Open-Meteo-unit averages (all ug/m3).

    Converts CO ug/m3 -> mg/m3 internally so callers pass raw averaged
    concentrations. Use this at serving time.
    """
    conv = dict(avg_ugm3)
    co = conv.get("carbon_monoxide")
    if co is not None and not (isinstance(co, float) and math.isnan(co)):
        conv["carbon_monoxide"] = co * CO_UGM3_TO_MGM3
    return cpcb_aqi(conv)


def cpcb_aqi_array_from_ugm3(avg_ugm3: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorised CPCB AQI from Open-Meteo-unit averages (all ug/m3).

    Converts CO ug/m3 -> mg/m3 internally. Use this to build the training label.
    """
    conv = dict(avg_ugm3)
    if "carbon_monoxide" in conv:
        conv["carbon_monoxide"] = np.asarray(conv["carbon_monoxide"], dtype="float64") * CO_UGM3_TO_MGM3
    return cpcb_aqi_array(conv)


def cpcb_category(aqi_value: float) -> str:
    """Map a CPCB AQI value to its category label."""
    if aqi_value is None or (isinstance(aqi_value, float) and math.isnan(aqi_value)):
        return "Unknown"
    for upper, label in CPCB_CATEGORIES:
        if aqi_value <= upper:
            return label
    return "Severe"
