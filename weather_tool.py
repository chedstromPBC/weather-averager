"""
Weather & Degree-Day Tool — Streamlit web app

One place for:
  • Temperature, wind speed/direction/gust, rain, dew point, humidity, soil temp
  • Daily values or a single summary for the whole period
  • An optional hour window (e.g. 10 PM–5 AM, wrapping past midnight)
  • Degree days with your pest's lower/upper thresholds, method, cutoff, and biofix

Data sources:
  • Open-Meteo (free, no key; ERA5 reanalysis + recent forecast-model data)
  • CIMIS weather stations (needs CIMIS_APPKEY environment variable; new 2026 CIMIS Web API)
  • AgriMet weather stations (US Bureau of Reclamation, Pacific Northwest; public, no key)

Deploy (Render):
  Start command: streamlit run weather_tool.py --server.port=10000 --server.address=0.0.0.0
  requirements.txt: streamlit, requests, pandas, numpy, altair
"""

import io
import math
import os
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

CIMIS_KEY = os.getenv("CIMIS_APPKEY", "")

APP_VERSION = "2.5.1"
APP_UPDATED = "2026-11-19"
# Changelog
#   2.5.1  Charts: month/year date axis, full-date tooltips
#   2.5.0  Imperial / Metric toggle (display, inputs, DD, CSV); saved with settings
#   2.4.0  Saved settings (bookmarkable link + settings file); version shown in sidebar
#   2.3.0  Open-Meteo fetches only selected variables; better rate-limit handling
#   2.2.0  AgriMet stations; multiple DD targets with markers on the chart; Visual Crossing removed
#   2.1.0  CIMIS stations (new 2026 CIMIS Web API)
#   2.0.0  Combined weather + degree-day tool

# Variables the user can pick: label -> (column name, how to aggregate)
VARIABLES = {
    "Air temperature (°F)": ("temp_f", "mean"),
    "Wind speed (mph)": ("wind_speed_mph", "mean"),
    "Wind direction": ("wind_dir_deg", "vector"),
    "Wind gust (mph)": ("wind_gust_mph", "max"),
    "Rainfall (in)": ("rain_in", "sum"),
    "Dew point (°F)": ("dew_point_f", "mean"),
    "Relative humidity (%)": ("rh_pct", "mean"),
    "Soil temp (°F)": ("soil_temp_f", "mean"),
}
# Variables a source can't supply
NOT_AVAILABLE = {
    "Open-Meteo": set(),
    "CIMIS station": {"Wind gust (mph)"},
    "AgriMet station": {"Soil temp (°F)"},
}
STATION_SOURCES = ("CIMIS station", "AgriMet station")

# ─────────────────────────── Units ───────────────────────────
# Everything is fetched and stored in imperial. With Metric on, the hourly data is
# converted once right after fetching, and DD thresholds/targets are entered in °C,
# so degree days come out in °C-days (= °F-days × 5/9).
UNIT_SYSTEMS = ["Imperial", "Metric"]
TEMP_COLS = ("temp_f", "dew_point_f", "soil_temp_f")
SPEED_COLS = ("wind_speed_mph", "wind_gust_mph")
RAIN_COLS = ("rain_in",)
METRIC_LABELS = {
    "Air temperature (°F)": "Air temperature (°C)",
    "Wind speed (mph)": "Wind speed (km/h)",
    "Wind gust (mph)": "Wind gust (km/h)",
    "Rainfall (in)": "Rainfall (mm)",
    "Dew point (°F)": "Dew point (°C)",
    "Soil temp (°F)": "Soil temp (°C)",
    "Max temp (°F)": "Max temp (°C)",
    "Min temp (°F)": "Min temp (°C)",
    "DD max temp (°F)": "DD max temp (°C)",
    "DD min temp (°F)": "DD min temp (°C)",
}


def f_to_c(f):
    return (f - 32) * 5 / 9


def c_to_f(c):
    return c * 9 / 5 + 32


def lbl(label, metric):
    """Display label for the chosen unit system (internal labels are imperial)."""
    return METRIC_LABELS.get(label, label) if metric else label


def relabel(df, metric):
    return df.rename(columns=METRIC_LABELS) if metric else df


def hourly_to_metric(df):
    """°F → °C, mph → km/h, in → mm. Returns a copy (fetch results are cached and shared)."""
    df = df.copy()
    for c in TEMP_COLS:
        if c in df:
            df[c] = f_to_c(df[c])
    for c in SPEED_COLS:
        if c in df:
            df[c] = df[c] * 1.609344
    for c in RAIN_COLS:
        if c in df:
            df[c] = df[c] * 25.4
    return df


def dist_txt(miles, metric):
    return f"{miles * 1.609344:.0f} km" if metric else f"{miles:.0f} mi"
SOIL_NOTE = {
    "Open-Meteo": "soil temp is modeled 0–7 cm (0 cm for the last few days)",
    "CIMIS station": "soil temp is measured at 15 cm under irrigated grass",
}


# ─────────────────────────── Data fetching ───────────────────────────

def _get_json(url, params, tries=4, retry_429=True):
    """GET with retry/backoff (Open-Meteo rate-limits shared server IPs).
    Errors never include the URL, because some sources put the API key in it."""
    host = url.split("/")[2]
    for i in range(tries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429 and retry_429 and i < tries - 1:
            time.sleep(min(int(r.headers.get("Retry-After", 0) or 0) or 10 * (i + 1), 30))  # ~60 s total
            continue
        if r.status_code == 429:
            raise RuntimeError(f"{host} says too many requests (HTTP 429). "
                               "Wait a minute and try again.")
        if r.status_code >= 400:
            raise RuntimeError(f"{host} returned HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


OM_HOURLY = {
    "temperature_2m": "temp_f",
    "wind_speed_10m": "wind_speed_mph",
    "wind_direction_10m": "wind_dir_deg",
    "wind_gusts_10m": "wind_gust_mph",
    "precipitation": "rain_in",
    "dew_point_2m": "dew_point_f",
    "relative_humidity_2m": "rh_pct",
    "soil_temperature_0_to_7cm": "soil_temp_f",
}


@lru_cache(maxsize=64)
def fetch_open_meteo(lat, lon, start, end, columns=tuple(OM_HOURLY.values())):
    """Hourly data, local time. Archive (ERA5) for older dates, forecast API for the last few days.
    Only the requested columns are fetched: Open-Meteo bills a request as
    (days / 14) × (variables / 10) calls, so fewer variables = far fewer calls."""
    common = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
    }
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    archive_end = min(end_d, date.today() - timedelta(days=6))
    frames, meta = [], {}
    wanted = [api for api, col in OM_HOURLY.items() if col in columns] or ["temperature_2m"]

    if start_d <= archive_end:
        p = dict(common, start_date=start, end_date=archive_end.isoformat(), hourly=",".join(wanted))
        j = _get_json("https://archive-api.open-meteo.com/v1/archive", p)
        frames.append(_om_frame(j))
        meta = {"grid_lat": j.get("latitude"), "grid_lon": j.get("longitude"), "elev_m": j.get("elevation")}

    recent_start = max(start_d, archive_end + timedelta(days=1))
    if recent_start <= end_d:
        hourly = ",".join(wanted).replace("soil_temperature_0_to_7cm", "soil_temperature_0cm")
        p = dict(common, start_date=recent_start.isoformat(), end_date=end, hourly=hourly)
        try:
            j = _get_json("https://api.open-meteo.com/v1/forecast", p)
            frames.append(_om_frame(j))
            meta = meta or {"grid_lat": j.get("latitude"), "grid_lon": j.get("longitude"), "elev_m": j.get("elevation")}
        except RuntimeError as e:
            if not frames:          # nothing else to show
                raise
            meta = dict(meta, warning=f"The last {(end_d - recent_start).days + 1} day(s) couldn't be fetched "
                                      f"({e}) — results stop at {archive_end}.")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return df, meta


def _om_frame(j):
    h = j.get("hourly", {})
    df = pd.DataFrame({"datetime": pd.to_datetime(h.get("time", []))})
    for api_name, col in OM_HOURLY.items():
        key = api_name if api_name in h else ("soil_temperature_0cm" if col == "soil_temp_f" else None)
        df[col] = h.get(key) if key and key in h else np.nan
    return df


# ── CIMIS (new Web API, live since June 2026; key goes in a header) ──
CIMIS_BASE = "https://et.water.ca.gov/StationWeb"
CIMIS_HOURLY = {
    "HlyAirTmp": "temp_f", "HlyWindSpd": "wind_speed_mph", "HlyWindDir": "wind_dir_deg",
    "HlyPrecip": "rain_in", "HlyDewPnt": "dew_point_f", "HlyRelHum": "rh_pct", "HlySoilTmp": "soil_temp_f",
}
CIMIS_HOURLY_ITEMS = "hly-air-tmp,hly-wind-spd,hly-wind-dir,hly-precip,hly-dew-pnt,hly-rel-hum,hly-soil-tmp"
CIMIS_BAD_QC = {"M", "N", "Q", "R", "S"}  # missing / not collected / related sensor missing / far out of range / not in service


def _cimis_get(path, key, params=None):
    r = requests.get(f"{CIMIS_BASE}/{path}", params=params, timeout=90,
                     headers={"Ocp-Apim-Subscription-Key": key, "Accept": "application/json"})
    if r.status_code == 401:
        raise RuntimeError("CIMIS rejected the key (401). Check CIMIS_APPKEY — it must be a key from the new CIMIS system.")
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        raise RuntimeError("CIMIS returned a non-JSON page (often a 'Request Rejected' firewall page).")


def _num(v):
    try:
        return float(str(v).split("/")[-1].strip())
    except (TypeError, ValueError):
        return np.nan


@lru_cache(maxsize=4)
def cimis_stations(key):
    j = _cimis_get("GetAllStations", key)
    rows = []
    for st_ in j.get("Stations", []):
        rows.append({
            "nbr": int(st_["StationNbr"]), "name": st_.get("Name", ""), "county": st_.get("County", ""),
            "lat": _num(st_.get("HmsLatitude")), "lon": _num(st_.get("HmsLongitude")),
            "elev_ft": _num(st_.get("Elevation")), "active": str(st_.get("IsActive")).lower() == "true",
        })
    return pd.DataFrame(rows)


def nearest_stations(stations, lat, lon, n=10):
    s = stations[stations["active"]].dropna(subset=["lat", "lon"]).copy()
    p1, p2 = np.radians(lat), np.radians(s["lat"])
    dlat, dlon = p2 - p1, np.radians(s["lon"] - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    s["miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))
    return s.nsmallest(n, "miles")


def _cimis_records(j):
    for prov in (j.get("Data") or {}).get("Providers", []) or []:
        yield from prov.get("Records", []) or []


def _cimis_val(rec, field):
    cell = rec.get(field) or {}
    if cell.get("Value") in (None, "") or str(cell.get("Qc", "")).strip() in CIMIS_BAD_QC:
        return np.nan
    return float(cell["Value"])


def _cimis_chunks(key, station, start, end, hourly, items, days=30):
    """Request in chunks; split a chunk in half if CIMIS refuses it (data-volume limit)."""
    out, d0, d1 = [], date.fromisoformat(start), date.fromisoformat(end)
    queue = []
    while d0 <= d1:
        queue.append((d0, min(d0 + timedelta(days=days - 1), d1)))
        d0 += timedelta(days=days)
    while queue:
        a, b = queue.pop(0)
        params = {"stationNbrs": station, "startDate": a.isoformat(), "endDate": b.isoformat(),
                  "isHourly": "true" if hourly else "false", "unitOfMeasure": "E", "dataItems": items}
        try:
            out.extend(_cimis_records(_cimis_get("GetDataByStationNumber", key, params)))
        except RuntimeError:
            raise
        except Exception:
            if a == b:
                raise
            mid = a + (b - a) // 2
            queue[:0] = [(a, mid), (mid + timedelta(days=1), b)]
    return out


@lru_cache(maxsize=32)
def fetch_cimis(station, start, end, key):
    """Hourly station data. CIMIS hours are hour-ending Pacific Standard Time (0100–2400);
    each is stamped at the start of its hour so all 24 stay on their calendar day."""
    rows = []
    for rec in _cimis_chunks(key, station, start, end, True, CIMIS_HOURLY_ITEMS):
        hr = int(rec["Hour"]) // 100
        row = {"datetime": pd.Timestamp(rec["Date"]) + pd.Timedelta(hours=hr - 1)}
        for field, col in CIMIS_HOURLY.items():
            row[col] = _cimis_val(rec, field)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["datetime"] + list(CIMIS_HOURLY.values()))
    df["wind_gust_mph"] = np.nan
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df, {}


@lru_cache(maxsize=32)
def fetch_cimis_daily_maxmin(station, start, end, key):
    """Daily max/min from the station's 1-minute readings (what UC IPM uses), midnight–midnight PST."""
    rows = []
    for rec in _cimis_chunks(key, station, start, end, False, "day-air-tmp-max,day-air-tmp-min", days=365):
        rows.append({"day": pd.Timestamp(rec["Date"]), "max": _cimis_val(rec, "DayAirTmpMax"),
                     "min": _cimis_val(rec, "DayAirTmpMin")})
    return pd.DataFrame(rows, columns=["day", "max", "min"]).drop_duplicates("day").set_index("day").sort_index()


# ── AgriMet (US Bureau of Reclamation, Columbia-Pacific Northwest; public, no key) ──
AGRIMET_BIN = "https://www.usbr.gov/pn-bin"
AGRIMET_MAP = "https://www.usbr.gov/pn/agrimet/agrimetmap/usbr_map.json"
# 15-minute codes: air temp, dew point, RH, wind speed, wind dir, gust, cumulative precip
AGRIMET_CODES = {"ob": "temp_f", "tp": "dew_point_f", "tu": "rh_pct", "ws": "wind_speed_mph",
                 "wd": "wind_dir_deg", "wg": "wind_gust_mph", "pc": "precip_cum_in"}


def _get_text(url, params, tries=3):
    host = url.split("/")[2]
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException:
            if i == tries - 1:
                raise RuntimeError(f"{host} didn't respond (it can be slow — try again or use a shorter range).")
            time.sleep(3)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{host} returned HTTP {r.status_code}.")
        return r.text


@lru_cache(maxsize=1)
def agrimet_stations():
    j = _get_json(AGRIMET_MAP, None)
    rows = []
    for f in j.get("features", []):
        p, (lo, la) = f.get("properties", {}), f["geometry"]["coordinates"][:2]
        if p.get("type", "agrimet") != "agrimet":
            continue
        rows.append({"nbr": p["siteid"].lower(), "name": p.get("title", "").replace(" AgriMet", "").replace(" Weather Station", ""),
                     "county": p.get("state", ""), "lat": float(la), "lon": float(lo),
                     "install": p.get("install", ""), "active": True})
    return pd.DataFrame(rows)


def _agrimet_csv(script, site, codes, start, end):
    params = {"list": ",".join(f"{site} {c}" for c in codes), "start": start, "end": end, "format": "csv"}
    txt = _get_text(f"{AGRIMET_BIN}/{script}", params)
    if not txt.lstrip().startswith("DateTime"):
        raise RuntimeError("AgriMet returned an unexpected page instead of CSV data.")
    df = pd.read_csv(io.StringIO(txt))
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df.pop("datetime"), errors="coerce")
    for c in df.columns.drop("datetime"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].abs() >= 99999, c] = np.nan  # hydromet missing-value codes (e.g. 998877)
    return df.dropna(subset=["datetime"])


@lru_cache(maxsize=32)
def fetch_agrimet(site, start, end):
    """15-minute AgriMet readings rolled up to hourly. Times are local and follow daylight saving."""
    parts, d0, d1 = [], date.fromisoformat(start), date.fromisoformat(end)
    while d0 <= d1:  # month-sized requests keep the server responsive
        e = min(d0 + timedelta(days=30), d1)
        parts.append(_agrimet_csv("instant.pl", site, AGRIMET_CODES, d0.isoformat(), e.isoformat()))
        d0 = e + timedelta(days=1)
    raw = pd.concat(parts, ignore_index=True).sort_values("datetime")
    raw = raw.rename(columns={f"{site}_{c}": col for c, col in AGRIMET_CODES.items()})
    for col in AGRIMET_CODES.values():
        if col not in raw:
            raw[col] = np.nan
    if raw["temp_f"].notna().sum() == 0:
        raise RuntimeError(f"AgriMet station {site.upper()} has no temperature data for these dates (it may be inactive).")
    # precip is a running season total: hourly rain = sum of positive increments (resets clip to 0)
    raw["rain_in"] = raw["precip_cum_in"].diff().clip(lower=0)
    # calm readings are logged as direction 0 (north); drop them so they don't skew the average direction
    raw.loc[(raw["wind_speed_mph"] == 0) & (raw["wind_dir_deg"] == 0), "wind_dir_deg"] = np.nan
    raw["hour"] = raw["datetime"].dt.floor("h")
    g = raw.groupby("hour")
    out = g[["temp_f", "dew_point_f", "rh_pct", "wind_speed_mph"]].mean().round(2)
    out["wind_dir_deg"] = g["wind_dir_deg"].apply(vector_mean_dir)
    out["wind_gust_mph"] = g["wind_gust_mph"].max()
    out["rain_in"] = g["rain_in"].sum(min_count=1).round(3)
    out["soil_temp_f"] = np.nan
    out = out.reset_index().rename(columns={"hour": "datetime"})
    return out, {}


@lru_cache(maxsize=32)
def fetch_agrimet_daily_maxmin(site, start, end):
    df = _agrimet_csv("daily.pl", site, ["mx", "mn"], start, end)
    df = df.rename(columns={f"{site}_mx": "max", f"{site}_mn": "min", "datetime": "day"})
    return df.set_index("day")[["max", "min"]].sort_index()


# ─────────────────────────── Aggregation ───────────────────────────

def vector_mean_dir(deg):
    deg = pd.Series(deg).dropna()
    if deg.empty:
        return np.nan
    r = np.radians(deg)
    return round(float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360), 0)


def compass(deg):
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return ""
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) / 22.5) % 16]


def fmt_hour(h):
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def window_filter(df, start_hour, end_hour):
    """Keep hours in the window and label each row with the date the window started on."""
    df = df.copy()
    hr = df["datetime"].dt.hour
    d = df["datetime"].dt.normalize()
    if start_hour <= end_hour:
        mask = (hr >= start_hour) & (hr <= end_hour)
        df["period"] = d
    else:  # wraps past midnight: early-morning hours belong to the previous evening
        mask = (hr >= start_hour) | (hr <= end_hour)
        df["period"] = d.where(hr >= start_hour, d - pd.Timedelta(days=1))
    return df[mask]


def aggregate(df, cols, group_key=None):
    """Aggregate selected columns. group_key=None -> one row for the whole data set."""
    def agg_one(g):
        out = {"hours": len(g)}
        for label, (col, how) in cols.items():
            s = g[col]
            if s.notna().sum() == 0:
                out[label] = np.nan
                continue
            if how == "mean":
                out[label] = round(s.mean(), 1)
            elif how == "sum":
                out[label] = round(s.sum(), 3)
            elif how == "max":
                out[label] = round(s.max(), 1)
            elif how == "vector":
                out[label] = vector_mean_dir(s)
                out["Wind direction (compass)"] = compass(out[label])
        if "Air temperature (°F)" in cols:
            out["Max temp (°F)"] = round(g["temp_f"].max(), 1)
            out["Min temp (°F)"] = round(g["temp_f"].min(), 1)
        return pd.Series(out)

    if group_key is None:
        res = agg_one(df).to_frame().T
    else:
        res = pd.DataFrame([agg_one(g).rename(k) for k, g in df.groupby(group_key)])
        res.index.name = "date"
        res = res.reset_index()
    for c in res.columns:  # mixed rows come back as object dtype; restore numbers for charts
        if c not in ("date", "Wind direction (compass)"):
            res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


# ─────────────────────────── Degree days ───────────────────────────
# Methods follow UC IPM conventions (single sine / single triangle / average)
# using each calendar day's max and min.

def _sine_above(mn, mx, t):
    """Area (°-days) of a single-sine daily curve above threshold t."""
    if mx <= t:
        return 0.0
    if mn >= t:
        return (mx + mn) / 2 - t
    a, m = (mx - mn) / 2, (mx + mn) / 2
    th = math.asin((t - m) / a)
    return ((m - t) * (math.pi / 2 - th) + a * math.cos(th)) / math.pi


def _sine_frac_above(mn, mx, t):
    if mx <= t:
        return 0.0
    if mn >= t:
        return 1.0
    a, m = (mx - mn) / 2, (mx + mn) / 2
    return (math.pi / 2 - math.asin((t - m) / a)) / math.pi


def _tri_above(mn, mx, t):
    if mx <= t:
        return 0.0
    if mn >= t:
        return (mx + mn) / 2 - t
    return (mx - t) ** 2 / (2 * (mx - mn))


def _tri_frac_above(mn, mx, t):
    if mx <= t:
        return 0.0
    if mn >= t:
        return 1.0
    return (mx - t) / (mx - mn)


def daily_dd(mn, mx, lower, upper=None, method="Single sine", cutoff="Horizontal"):
    """Degree days for one day. upper=None means no upper threshold."""
    if mn is None or mx is None or np.isnan(mn) or np.isnan(mx):
        return np.nan
    mn, mx = min(mn, mx), max(mn, mx)

    if method == "Average":
        hi = mx if upper is None or cutoff != "Horizontal" else min(mx, upper)
        avg = (hi + max(mn, lower)) / 2 if cutoff == "Horizontal" else (mx + mn) / 2
        if upper is not None and cutoff == "Vertical" and avg > upper:
            return 0.0
        return max(0.0, avg - lower)

    above, frac = (_sine_above, _sine_frac_above) if method == "Single sine" else (_tri_above, _tri_frac_above)
    dd = above(mn, mx, lower)
    if upper is None:
        return dd
    if cutoff == "Horizontal":       # development holds at the upper threshold
        return dd - above(mn, mx, upper)
    if cutoff == "Vertical":         # no development while above the upper threshold
        return dd - above(mn, mx, upper) - (upper - lower) * frac(mn, mx, upper)
    if cutoff == "Intermediate":     # development declines above the upper threshold
        return max(0.0, dd - 2 * above(mn, mx, upper))
    return dd


def degree_day_table(hourly, lower, upper, method, cutoff, biofix, end, daily_maxmin=None):
    """Daily max/min -> daily and cumulative DD from biofix.
    Uses station daily max/min when given (CIMIS), otherwise the max/min of the 24 hourly values."""
    d = hourly.assign(day=hourly["datetime"].dt.normalize()).groupby("day")["temp_f"].agg(["max", "min", "count"])
    if daily_maxmin is not None and len(daily_maxmin):
        d = d.reindex(d.index.union(daily_maxmin.index))
        for c in ("max", "min"):
            d[c] = daily_maxmin[c].reindex(d.index).fillna(d[c])  # fall back to hourly if a day is missing
        d["count"] = d["count"].fillna(0).astype(int)
    d = d[(d.index >= pd.Timestamp(biofix)) & (d.index <= pd.Timestamp(end))]
    d["DD"] = [round(daily_dd(r["min"], r["max"], lower, upper, method, cutoff), 2) for _, r in d.iterrows()]
    d["Cumulative DD"] = d["DD"].cumsum().round(1)
    d = d.rename(columns={"max": "DD max temp (°F)", "min": "DD min temp (°F)", "count": "hours of data"})
    d.index.name = "date"
    return d.reset_index()


def target_hits(dd, targets, biofix):
    """For each (event, DD) target: first date the cumulative total reaches it."""
    rows = []
    for ev, tgt in targets:
        hit = dd[dd["Cumulative DD"] >= tgt]
        if len(hit):
            d = pd.Timestamp(hit["date"].iloc[0])
            rows.append({"Event": ev, "DD target": tgt, "Date reached": d.date(),
                         "Days after biofix": (d - pd.Timestamp(biofix)).days, "Reached": True})
        else:
            short = tgt - (dd["Cumulative DD"].iloc[-1] if len(dd) else 0)
            rows.append({"Event": ev, "DD target": tgt, "Date reached": None,
                         "Days after biofix": None, "Reached": False, "DD still needed": round(short, 1)})
    return pd.DataFrame(rows)


def date_axis(dates):
    """Readable date axis: monthly ticks labeled "Apr 2026" for long ranges, weekly "Apr 19" for short ones."""
    import altair as alt
    d = pd.to_datetime(pd.Series(dates))
    span = (d.max() - d.min()).days if len(d) else 0
    if span > 75:
        return alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount={"interval": "month", "step": 1},
                                                         labelAngle=0, grid=True))
    if span > 14:
        return alt.X("date:T", title=None, axis=alt.Axis(format="%b %-d", tickCount={"interval": "week", "step": 1},
                                                         labelAngle=0))
    return alt.X("date:T", title=None, axis=alt.Axis(format="%b %-d", tickCount={"interval": "day", "step": 1},
                                                     labelAngle=0))


def temp_chart(daily, labels, u):
    """Daily mean/max/min temperature lines with a month-aware date axis and full-date tooltips."""
    import altair as alt
    avg, mx, mn = labels
    df = daily[["date", avg, mx, mn]].assign(date=pd.to_datetime(daily["date"]))
    long = df.melt("date", var_name="Series", value_name="Temp")
    color = alt.Color("Series:N", title=None, legend=alt.Legend(orient="bottom"),
                      scale=alt.Scale(domain=[avg, mx, mn], range=["#1f5fbf", "#7fbff5", "#e8352f"]))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=date_axis(df["date"]), y=alt.Y("Temp:Q", title=f"Temperature ({u})"), color=color)
    hover = alt.Chart(df).mark_rule(opacity=0, strokeWidth=12).encode(
        x="date:T",
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %b %-d, %Y")]
                + [alt.Tooltip(f"{c}:Q", title=c, format=".1f") for c in (mx, avg, mn)])
    return alt.layer(lines, hover).properties(height=320)


def dd_chart(dd, hits, y_title="Cumulative degree days"):
    """Cumulative DD line with a dot + label where each target is reached; dashed line for targets not reached."""
    import altair as alt

    LINE, DOT, MUTED = "#2a78d6", "#eb6834", "#8a8a85"
    base = dd[["date", "DD", "Cumulative DD"]].assign(date=pd.to_datetime(dd["date"]))
    line = alt.Chart(base).mark_line(color=LINE, strokeWidth=2).encode(
        x=date_axis(base["date"]), y=alt.Y("Cumulative DD:Q", title=y_title))
    hover = alt.Chart(base).mark_circle(size=120, opacity=0).encode(
        x="date:T", y="Cumulative DD:Q",
        tooltip=[alt.Tooltip("date:T", title="Date", format="%a %b %-d, %Y"), alt.Tooltip("Cumulative DD:Q", format=",.1f"),
                 alt.Tooltip("DD:Q", title="DD that day", format=".1f")])
    layers = [line, hover]
    if hits is not None and len(hits):
        # pass Altair only clean, typed columns (mixed None/number/date columns trip its serializer)
        got = hits[hits["Reached"]]
        got = pd.DataFrame({"Event": got["Event"].astype(str), "DD target": got["DD target"].astype(float),
                            "date": pd.to_datetime(got["Date reached"]),
                            "Days after biofix": pd.to_numeric(got["Days after biofix"])})
        miss = hits[~hits["Reached"]]
        miss = pd.DataFrame({"Event": miss["Event"].astype(str), "DD target": miss["DD target"].astype(float),
                             "DD still needed": pd.to_numeric(miss.get("DD still needed", pd.Series(dtype=float)))})
        if len(got):
            pts = alt.Chart(got).encode(x="date:T", y="DD target:Q",
                                        tooltip=["Event", alt.Tooltip("DD target:Q", format=",.0f"),
                                                 alt.Tooltip("date:T", title="Reached"), "Days after biofix"])
            layers += [pts.mark_point(filled=True, size=110, color=DOT, stroke="white", strokeWidth=2, opacity=1),
                       pts.mark_text(align="right", dx=-9, dy=-9, fontSize=12, color=MUTED).encode(text="Event")]
        if len(miss):
            rule = alt.Chart(miss).encode(y="DD target:Q", tooltip=["Event", "DD target", "DD still needed"])
            layers += [rule.mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1),
                       rule.mark_text(align="left", x=4, dy=-7, fontSize=12, color=MUTED)
                           .encode(text=alt.Text("Event"))]
    return alt.layer(*layers).properties(height=360)


# ─────────────────────────── Streamlit UI ───────────────────────────

# ─────────────────────────── Saved settings ───────────────────────────
# Every input lives in st.session_state under one of these keys. The same dict is
# mirrored into the page URL (bookmark it) and can be saved/loaded as a JSON file.

SOURCES = ["Open-Meteo", "CIMIS station", "AgriMet station"]
COL_TO_LABEL = {col: label for label, (col, _) in VARIABLES.items()}


def default_settings():
    today = date.today()
    return {
        "name": "Site 1", "lat": 39.090431, "lon": -121.423245,
        "start": today - timedelta(days=30), "end": today - timedelta(days=1),
        "source": "Open-Meteo", "station": "",
        "vars": ["Air temperature (°F)", "Wind speed (mph)", "Wind direction"],
        "output": "Daily values", "use_window": False, "sh": 22, "eh": 5,
        "use_dd": True, "lower": 55.0, "has_upper": True, "upper": 94.0,
        "method": "Single sine", "cutoff": "Horizontal", "biofix": today - timedelta(days=30),
        "station_maxmin": True, "targets": [], "units": "Imperial",
    }


def settings_to_flat(st_):
    """Current settings as plain strings (for the URL and the JSON file)."""
    ss = st_.session_state["cfg"]
    out = {
        "name": ss["name"], "lat": f"{ss['lat']:.6f}", "lon": f"{ss['lon']:.6f}",
        "start": ss["start"].isoformat(), "end": ss["end"].isoformat(),
        "source": ss["source"], "station": str(ss.get("station") or ""),
        "vars": ",".join(VARIABLES[v][0] for v in ss["vars"]),
        "output": ss["output"], "window": f"{int(ss['use_window'])}:{ss['sh']}-{ss['eh']}",
        "dd": int(ss["use_dd"]), "lower": f"{ss['lower']:g}",
        "upper": f"{ss['upper']:g}" if ss["has_upper"] else "none",
        "method": ss["method"], "cutoff": ss["cutoff"], "biofix": ss["biofix"].isoformat(),
        "stnmm": int(ss["station_maxmin"]), "units": ss["units"],
        "targets": "|".join(f"{ev}~{t:g}" for ev, t in st_.session_state.get("targets_saved", [])),
    }
    return {k: str(v) for k, v in out.items()}


def flat_to_settings(flat):
    """Parse URL/JSON values over the defaults; anything missing or malformed keeps its default."""
    s = default_settings()

    def get(k, conv):
        try:
            if k in flat and str(flat[k]) != "":
                return conv(str(flat[k]))
        except (ValueError, TypeError):
            pass
        return None

    for k, conv in (("name", str), ("lat", float), ("lon", float), ("lower", float),
                    ("start", date.fromisoformat), ("end", date.fromisoformat), ("biofix", date.fromisoformat)):
        v = get(k, conv)
        if v is not None:
            s[k] = v
    if flat.get("source") in SOURCES:
        s["source"] = flat["source"]
    s["station"] = str(flat.get("station", "") or "")
    if "vars" in flat:
        s["vars"] = [COL_TO_LABEL[c] for c in str(flat["vars"]).split(",") if c in COL_TO_LABEL]
    if flat.get("output") in ("Daily values", "Whole-period summary", "Both"):
        s["output"] = flat["output"]
    w = get("window", lambda x: (bool(int(x.split(":")[0])), *map(int, x.split(":")[1].split("-"))))
    if w and all(0 <= h <= 23 for h in w[1:]):
        s["use_window"], s["sh"], s["eh"] = w
    d = get("dd", int)
    if d is not None:
        s["use_dd"] = bool(d)
    if str(flat.get("upper", "")) == "none":
        s["has_upper"] = False
    else:
        u = get("upper", float)
        if u is not None:
            s["upper"], s["has_upper"] = u, True
    if flat.get("method") in ("Single sine", "Single triangle", "Average"):
        s["method"] = flat["method"]
    if flat.get("cutoff") in ("Horizontal", "Vertical", "Intermediate"):
        s["cutoff"] = flat["cutoff"]
    if flat.get("units") in UNIT_SYSTEMS:
        s["units"] = flat["units"]
    m = get("stnmm", int)
    if m is not None:
        s["station_maxmin"] = bool(m)
    tg = []
    for part in str(flat.get("targets", "") or "").split("|"):
        if "~" in part:
            ev, _, t = part.rpartition("~")
            try:
                tg.append((ev, float(t)))
            except ValueError:
                pass
    s["targets"] = tg
    return s


def apply_settings(st_, s):
    """Make s the current settings. Widget keys are cleared so every input redraws from cfg."""
    ss = st_.session_state
    ss["cfg"] = {k: v for k, v in s.items() if k != "targets"}
    for k in [k for k in ss.keys() if str(k).startswith("w_")]:
        del ss[k]
    ss["targets_saved"] = list(s["targets"])
    ss["targets_df"] = pd.DataFrame({"Event": [e for e, _ in s["targets"]],
                                     "DD target": [t for _, t in s["targets"]]}).astype({"Event": "str", "DD target": "float"})
    ss["targets_ver"] = ss.get("targets_ver", 0) + 1   # fresh editor so loaded rows show up


def wkey(st_, k):
    """Widget key for setting k. Values live in session_state['cfg'] (which Streamlit never clears);
    the widget's own key is re-seeded from cfg whenever the widget was hidden and comes back."""
    ss = st_.session_state
    if "w_" + k not in ss:
        ss["w_" + k] = ss["cfg"][k]
    return "w_" + k


def keep_valid(st_, k, options):
    """A saved value that isn't offered any more (e.g. a station list changed) falls back to the first option."""
    ss = st_.session_state
    if ss.get("w_" + k, ss["cfg"].get(k)) not in options:
        ss["cfg"][k] = options[0]
        ss["w_" + k] = options[0]


def sync_cfg(st_):
    ss = st_.session_state
    for k in ss["cfg"]:
        if "w_" + k in ss:
            ss["cfg"][k] = ss["w_" + k]


def convert_units_on_toggle(st_):
    """When the unit toggle flips, convert the DD thresholds and targets so they mean the same thing."""
    ss = st_.session_state
    new, old = ss["w_units"], ss["cfg"]["units"]
    if new == old:
        return
    to_metric = new == "Metric"
    for k in ("lower", "upper"):
        v = ss.get("w_" + k, ss["cfg"][k])
        v = round(f_to_c(v), 2) if to_metric else round(c_to_f(v), 1)
        ss["cfg"][k] = v
        ss["w_" + k] = v
    scale = 5 / 9 if to_metric else 9 / 5
    targets = [(ev, round(t * scale, 1)) for ev, t in ss.get("targets_saved", [])]
    ss["targets_saved"] = targets
    ss["targets_df"] = pd.DataFrame({"Event": [e for e, _ in targets],
                                     "DD target": [t for _, t in targets]}).astype({"Event": "str", "DD target": "float"})
    ss["targets_ver"] = ss.get("targets_ver", 0) + 1
    ss["cfg"]["units"] = new


def main():
    import streamlit as st

    st.set_page_config(page_title="Weather & Degree-Day Tool", layout="wide")
    st.title("Weather & Degree-Day Tool")
    ss = st.session_state

    # First load: settings come from the link (if any) over the defaults
    if "_init" not in ss:
        apply_settings(st, flat_to_settings(dict(st.query_params)))
        ss["_init"] = True

    with st.sidebar:
        with st.expander("Saved settings"):
            st.caption("The page address updates as you change inputs — bookmark it to reopen this exact query. "
                       "Or save the settings as a file and load it later.")
            up = st.file_uploader("Load a settings file", type="json", label_visibility="collapsed")
            if up is not None and ss.get("_loaded_file") != up.file_id:
                import json
                try:
                    apply_settings(st, flat_to_settings(json.loads(up.getvalue().decode("utf-8"))))
                    ss["_loaded_file"] = up.file_id
                    st.rerun()
                except ValueError:
                    st.error("That file isn't a settings file from this app.")
            settings_slot = st.empty()  # the download button is drawn after the inputs below

        st.radio("Units", UNIT_SYSTEMS, horizontal=True, key=wkey(st, "units"),
                 on_change=convert_units_on_toggle, args=(st,),
                 help="Metric: °C, km/h, mm, and °C degree days. Your DD thresholds and targets convert automatically.")
        metric = ss["w_units"] == "Metric"
        u = "°C" if metric else "°F"

        st.header("Location")
        st.text_input("Location name", key=wkey(st, "name"))
        c1, c2 = st.columns(2)
        c1.number_input("Latitude", format="%.6f", key=wkey(st, "lat"))
        c2.number_input("Longitude", format="%.6f", key=wkey(st, "lon"))
        name, lat, lon = ss["w_name"], ss["w_lat"], ss["w_lon"]

        st.header("Dates")
        start = st.date_input("Start date", key=wkey(st, "start"))
        end = st.date_input("End date", key=wkey(st, "end"))

        source = st.radio("Data source", SOURCES, key=wkey(st, "source"))

        station, station_label, cimis_key = None, "", CIMIS_KEY
        if source == "CIMIS station":
            if not cimis_key:
                cimis_key = st.text_input("CIMIS app key", type="password",
                                          help="Set CIMIS_APPKEY on Render so you don't have to paste it.")
            if cimis_key:
                try:
                    near = nearest_stations(cimis_stations(cimis_key), lat, lon)
                    labels = {str(r.nbr): f"#{r.nbr} {r.name} ({r.county}) · {dist_txt(r.miles, metric)}"
                              for r in near.itertuples()}
                    keep_valid(st, "station", list(labels))
                    station = st.selectbox("Station (nearest active to your lat/long)", list(labels),
                                           format_func=labels.get, key=wkey(st, "station"))
                    station_label, station = labels[station], int(station)
                except Exception as e:
                    st.error(f"Couldn't load the CIMIS station list: {e}")
        elif source == "AgriMet station":
            try:
                near = nearest_stations(agrimet_stations(), lat, lon)
                labels = {r.nbr: f"{r.nbr.upper()} {r.name} · {dist_txt(r.miles, metric)}" for r in near.itertuples()}
                keep_valid(st, "station", list(labels))
                station = st.selectbox("Station (nearest to your lat/long)", list(labels), format_func=labels.get,
                                       key=wkey(st, "station"),
                                       help="AgriMet's list includes some retired stations; they'll report no data.")
                station_label = labels[station]
            except Exception as e:
                st.error(f"Couldn't load the AgriMet station list: {e}")

    # ── Weather section ──
    st.subheader("1 · Weather variables")
    avail = [v for v in VARIABLES if v not in NOT_AVAILABLE[source]]
    vk = wkey(st, "vars")
    ss[vk] = [v for v in ss[vk] if v in avail]
    picked = st.multiselect("Variables", avail, format_func=lambda v: lbl(v, metric), key=wkey(st, "vars"))

    c1, c2, c3 = st.columns([1.2, 1, 1])
    output_mode = c1.radio("Output", ["Daily values", "Whole-period summary", "Both"], key=wkey(st, "output"))
    use_window = c2.checkbox("Limit to an hour window", key=wkey(st, "use_window"),
                             help="E.g. 10 PM–5 AM for nighttime averages. Windows that cross midnight are labeled by the starting evening.")
    hours = list(range(24))
    sh = c2.selectbox("From", hours, format_func=fmt_hour, disabled=not use_window, key=wkey(st, "sh"))
    eh = c3.selectbox("To (inclusive)", hours, format_func=fmt_hour, disabled=not use_window, key=wkey(st, "eh"))

    # ── Degree-day section ──
    st.subheader("2 · Degree days")
    use_dd = st.checkbox("Calculate degree days", key=wkey(st, "use_dd"))
    if use_dd:
        c1, c2, c3, c4, c5 = st.columns(5)
        lower = c1.number_input(f"Lower threshold ({u})", step=0.5, key=wkey(st, "lower"))
        has_upper = c2.checkbox("Use upper threshold", key=wkey(st, "has_upper"))
        upper = c2.number_input(f"Upper threshold ({u})", step=0.5, disabled=not has_upper, key=wkey(st, "upper"))
        method = c3.selectbox("Method", ["Single sine", "Single triangle", "Average"], key=wkey(st, "method"))
        cut_opts = ["Horizontal", "Vertical"] + ([] if method == "Average" else ["Intermediate"])
        keep_valid(st, "cutoff", cut_opts)
        cutoff = c4.selectbox("Cutoff", cut_opts, disabled=not has_upper, key=wkey(st, "cutoff"))
        biofix = c5.date_input("Biofix", key=wkey(st, "biofix"),
                               help="DD accumulate starting on this date (inclusive). Can be before the start date.")
        st.markdown(f"**DD targets** ({u}-days, optional) — add a row per life-stage event, e.g. egg lay, egg hatch, first flight")
        tgt_table = st.data_editor(
            ss["targets_df"], num_rows="dynamic", use_container_width=True, hide_index=True,
            key=f"dd_targets_{ss['targets_ver']}",
            column_config={"Event": st.column_config.TextColumn("Event"),
                           "DD target": st.column_config.NumberColumn(f"DD target ({u}-days)", min_value=0, step=0.1, format="%.1f")})
        targets = []
        for _, r in tgt_table.dropna(subset=["DD target"]).iterrows():
            ev = str(r["Event"]).strip() if pd.notna(r["Event"]) and str(r["Event"]).strip() else f"Target {len(targets) + 1}"
            if r["DD target"] > 0:
                targets.append((ev, float(r["DD target"])))
        targets.sort(key=lambda t: t[1])
        ss["targets_saved"] = targets
        if not has_upper:
            upper = None
        use_station_maxmin = source in STATION_SOURCES and st.checkbox(
            "Use the station's daily max/min for degree days (matches UC IPM / USPEST)", key=wkey(st, "station_maxmin"),
            help="Station daily max/min come from the logger's full-resolution readings. "
                 "Unchecked = max/min of the hourly averages, which runs a bit lower.")

    # Save: mirror settings into the URL and offer them as a file
    sync_cfg(st)
    flat = settings_to_flat(st)
    try:
        st.query_params.from_dict(flat)
    except Exception:
        pass  # very long settings can exceed URL limits; the file still works
    import json
    safe = "".join(ch if ch.isalnum() else "_" for ch in name) or "settings"
    settings_slot.download_button("Save settings to a file", json.dumps(flat, indent=1),
                                  f"{safe}_settings.json", "application/json", use_container_width=True)
    st.sidebar.caption(f"Version {APP_VERSION} · updated {APP_UPDATED}")

    if not st.button("Fetch & calculate", type="primary"):
        st.caption("UC IPM's default is single sine with a horizontal cutoff.")
        return

    if end < start:
        st.error("End date is before start date.")
        return

    if source in STATION_SOURCES and station is None:
        st.error("Pick a station first." + (" (CIMIS needs a valid key.)" if source == "CIMIS station" else ""))
        return

    fetch_start = min(start, biofix) if use_dd else start
    wraps = use_window and sh > eh
    # An overnight window that starts on the end date finishes the next morning
    fetch_end = min(end + timedelta(days=1), date.today()) if wraps else end
    try:
        with st.spinner(f"Fetching hourly data from {source}…"):
            if source == "CIMIS station":
                hourly, meta = fetch_cimis(station, fetch_start.isoformat(), fetch_end.isoformat(), cimis_key)
            elif source == "AgriMet station":
                hourly, meta = fetch_agrimet(station, fetch_start.isoformat(), fetch_end.isoformat())
            else:
                need = {VARIABLES[v][0] for v in picked} | ({"temp_f"} if use_dd else set())
                hourly, meta = fetch_open_meteo(round(lat, 6), round(lon, 6), fetch_start.isoformat(),
                                                fetch_end.isoformat(), tuple(sorted(need)))
            station_maxmin = None
            if use_dd and source in STATION_SOURCES and use_station_maxmin:
                if source == "CIMIS station":
                    station_maxmin = fetch_cimis_daily_maxmin(station, fetch_start.isoformat(), end.isoformat(), cimis_key)
                else:
                    station_maxmin = fetch_agrimet_daily_maxmin(station, fetch_start.isoformat(), end.isoformat())
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return
    if hourly.empty:
        st.error("No data returned for that location/date range.")
        return
    if metric:
        hourly = hourly_to_metric(hourly)
        if station_maxmin is not None:
            station_maxmin = f_to_c(station_maxmin)

    if meta.get("warning"):
        st.warning(meta["warning"])
    if meta.get("grid_lat") is not None:
        elev = f", elevation {meta['elev_m']:.0f} m" if meta.get("elev_m") is not None else ""
        st.caption(f"{source} data point used: {meta['grid_lat']:.4f}, {meta['grid_lon']:.4f}{elev}. "
                   "Values are modeled (gridded), not station observations.")
    if source == "CIMIS station":
        st.caption(f"CIMIS station {station_label}. Hours are Pacific Standard Time (no daylight saving). "
                   "Readings flagged missing or far out of range by CIMIS QC are dropped.")
    if source == "AgriMet station":
        st.caption(f"AgriMet station {station_label}. 15-minute readings averaged to hourly; times are local and "
                   "follow daylight saving. AgriMet data are provisional.")
    soil = "Soil temp (°F)" in picked and SOIL_NOTE.get(source)
    if soil:
        st.caption(f"Note: {soil}.")

    in_range = hourly[(hourly["datetime"] >= pd.Timestamp(start)) &
                      (hourly["datetime"] < pd.Timestamp(fetch_end) + pd.Timedelta(days=1))]
    cols = {k: VARIABLES[k] for k in picked}
    downloads = {}

    # Weather results
    if cols:
        w = window_filter(in_range, sh, eh) if use_window else in_range.assign(period=in_range["datetime"].dt.normalize())
        w = w[(w["period"] >= pd.Timestamp(start)) & (w["period"] <= pd.Timestamp(end))]
        win_label = f"{fmt_hour(sh)}–{fmt_hour(eh)}" if use_window else "all hours"
        if output_mode in ("Whole-period summary", "Both"):
            st.subheader(f"Period summary · {start} to {end} · {win_label}")
            summ = relabel(aggregate(w, cols), metric)
            show = [c for c in summ.columns if c != "hours"]
            mcols = st.columns(min(len(show), 5))
            for i, c in enumerate(show):
                v = summ.iloc[0][c]
                mcols[i % len(mcols)].metric(c, "—" if pd.isna(v) else v)
            st.caption(f"{int(summ.iloc[0]['hours'])} hourly records. Wind direction is a vector mean; gust is the period max; rain is the total.")
            downloads["summary"] = summ
        if output_mode in ("Daily values", "Both"):
            st.subheader(f"Daily values · {win_label}")
            daily = relabel(aggregate(w, cols, "period"), metric)
            daily["date"] = pd.to_datetime(daily["date"]).dt.date
            st.dataframe(daily, use_container_width=True, hide_index=True)
            if "Air temperature (°F)" in cols:
                labels = [lbl(c, metric) for c in ("Air temperature (°F)", "Max temp (°F)", "Min temp (°F)")]
                st.altair_chart(temp_chart(daily, labels, u), use_container_width=True)
            downloads["daily"] = daily

    # Degree-day results
    if use_dd:
        dd = relabel(degree_day_table(hourly, lower, upper, method, cutoff, biofix, end, station_maxmin), metric)
        dd["date"] = pd.to_datetime(dd["date"]).dt.date
        total = dd["Cumulative DD"].iloc[-1] if len(dd) else 0
        up_txt = f"{upper:g}{u} ({cutoff.lower()} cutoff)" if upper is not None else "none"
        st.subheader("Degree days")
        m1, m2 = st.columns(2)
        m1.metric(f"Accumulated DD ({u}-days) since biofix {biofix}", f"{total:,.1f}")
        m2.metric("Thresholds", f"{lower:g}{u} / {up_txt.split(' ')[0]}")
        hits = target_hits(dd, targets, biofix) if targets else None
        st.caption(f"Method: {method}; lower {lower:g}{u}; upper {up_txt}. " + ("Uses the station's daily max/min." if station_maxmin is not None else "Uses each calendar day's hourly max/min."))
        incomplete = dd[dd["hours of data"] < 20]
        if len(incomplete):
            st.warning(f"{len(incomplete)} day(s) have fewer than 20 hourly readings; their max/min may be off.")
        st.altair_chart(dd_chart(dd, hits, f"Cumulative degree days ({u}-days)"), use_container_width=True)
        if hits is not None:
            st.markdown("**Targets**")
            show_hits = hits.drop(columns="Reached")
            st.dataframe(show_hits, use_container_width=True, hide_index=True)
            downloads["dd_targets"] = show_hits
        st.dataframe(dd, use_container_width=True, hide_index=True)
        downloads["degree_days"] = dd

    # Download: one CSV with labeled sections
    if downloads:
        buf = io.StringIO()
        src_txt = f"{source} {station_label}" if station_label else source
        buf.write(f"# {name} ({lat}, {lon}) | source: {src_txt} | {start} to {end} | units: {ss['w_units'].lower()}\n")
        if use_dd:
            buf.write(f"# Degree days ({u}-days): {method}, lower {lower:g}{u}, upper {up_txt}, biofix {biofix}\n")
        for k, df in downloads.items():
            buf.write(f"\n# {k}\n")
            df.to_csv(buf, index=False)
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        st.download_button("Download CSV", buf.getvalue(), f"{safe}_{start}_{end}.csv", "text/csv")

    st.caption("Sources: Open-Meteo Historical Weather API (open-meteo.com) · "
               "CIMIS Web API (cimis.water.ca.gov) · AgriMet, US Bureau of Reclamation (usbr.gov/pn/agrimet) · "
               "Degree-day methods per UC IPM (ipm.ucanr.edu/WEATHER/ddconcepts.html)")


if __name__ == "__main__":
    main()
