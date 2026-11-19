"""
Weather Data Averager, Streamlit web app (Open-Meteo)
Modes:
  - Hourly: overnight window averages + hourly temps (slow for long ranges)
  - Daily: one row per day with rain total and temp mean/max/min (fast)
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, date, timedelta
from io import StringIO

st.set_page_config(page_title="Weather Data Averager", layout="wide")
st.title("Weather Data Averager")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/Los_Angeles"


# ── Helpers ──

def deg_to_compass(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int((deg + 11.25) / 22.5) % 16]


def fmt_hour(h):
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def get_with_retry(params):
    """Call the archive API, retrying with backoff on 429."""
    delay = 2
    for attempt in range(3):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if resp.status_code == 429 and attempt < 2:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()


@st.cache_data(ttl=3600)
def fetch_hourly(lat, lon, start, end):
    data = get_with_retry({
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,rain",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": TZ,
    })
    h = data["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "temp_f": h["temperature_2m"],
        "wind_speed_mph": h["wind_speed_10m"],
        "wind_dir_deg": h["wind_direction_10m"],
        "rain_in": h["rain"],
    })
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    return df


@st.cache_data(ttl=3600)
def fetch_daily(lat, lon, start, end):
    data = get_with_retry({
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": TZ,
    })
    d = data["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).date,
        "temp_mean_f": d["temperature_2m_mean"],
        "temp_max_f": d["temperature_2m_max"],
        "temp_min_f": d["temperature_2m_min"],
        "rain_in": d["precipitation_sum"],
    })


def csv_header(lines):
    return "".join(f"# {l}\n" for l in lines) + "#\n"


# ── Hourly processing ──

def process_hourly(df, start_hour, end_hour):
    span = (end_hour - start_hour) % 24
    if start_hour > end_hour:
        mask = (df["hour"] >= start_hour) | (df["hour"] <= end_hour)
        early = df["hour"] <= end_hour
    else:
        mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
        early = pd.Series(False, index=df.index)

    f = df[mask].copy()
    if f.empty:
        return None
    f["night_of"] = f["date"] - pd.to_timedelta(early[mask].astype(int), unit="D")
    f["wu"] = np.sin(np.radians(f["wind_dir_deg"]))
    f["wv"] = np.cos(np.radians(f["wind_dir_deg"]))

    agg = f.groupby("night_of").agg(
        avg_temp_f=("temp_f", "mean"),
        avg_wind_mph=("wind_speed_mph", "mean"),
        total_rain_in=("rain_in", "sum"),
        mu=("wu", "mean"),
        mv=("wv", "mean"),
    ).reset_index()
    agg["avg_wind_dir_deg"] = (np.degrees(np.arctan2(agg["mu"], agg["mv"])) % 360).round(1)
    agg["avg_wind_dir_compass"] = agg["avg_wind_dir_deg"].apply(deg_to_compass)
    agg = agg.drop(columns=["mu", "mv"])
    agg["avg_temp_f"] = agg["avg_temp_f"].round(1)
    agg["avg_wind_mph"] = agg["avg_wind_mph"].round(1)
    agg["total_rain_in"] = agg["total_rain_in"].round(3)

    # Hourly temps as columns, in chronological window order
    f["slot"] = f["hour"].apply(lambda h: f"{int(h):02d}")
    wide = f.pivot_table(index="night_of", columns="slot", values="temp_f", aggfunc="first")
    ordered_slots = [f"{(start_hour + i) % 24:02d}" for i in range(span + 1)]
    wide = wide.reindex(columns=ordered_slots)
    wide.columns = [f"temp_f_{c}" for c in wide.columns]
    wide = wide.reset_index()

    agg = agg.merge(wide, on="night_of", how="left")
    return agg


# ── Sidebar ──

yesterday = date.today() - timedelta(days=1)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Output", ["Daily values", "Hourly (overnight window)"], index=0)
    location_name = st.text_input("Location name", value="Madera CA")
    lat = st.number_input("Latitude", value=36.9613, format="%.4f")
    lon = st.number_input("Longitude", value=-120.0607, format="%.4f")
    start_date = st.date_input("Start date", value=date(2024, 1, 1))
    end_date = st.date_input("End date", value=yesterday)
    if mode.startswith("Hourly"):
        start_hour = st.slider("Start hour (0-23)", 0, 23, 22)
        end_hour = st.slider("End hour (0-23)", 0, 23, 5)

if end_date > yesterday:
    st.warning("Archive data usually lags a day or two. Consider an end date of yesterday or earlier.")

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ── Main ──

if st.button("Fetch Data", type="primary"):
    sd, ed = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    safe_name = location_name.lower().replace(" ", "_").replace(",", "")
    filename_base = f"{safe_name}_{'daily' if mode.startswith('Daily') else 'hourly'}_{sd.replace('-', '_')}_to_{ed.replace('-', '_')}"
    access = datetime.now().strftime("%Y-%m-%d %H:%M")

    with st.spinner("Fetching data..."):
        try:
            if mode.startswith("Daily"):
                df = fetch_daily(lat, lon, sd, ed)
                total_rain = df["rain_in"].sum()
                st.subheader(f"Daily values, {location_name} ({sd} to {ed})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total rainfall", f"{total_rain:.2f} in")
                c2.metric("Mean temp (period)", f"{df['temp_mean_f'].mean():.1f} °F")
                c3.metric("Days with rain", int((df["rain_in"] > 0).sum()))
                st.dataframe(df, use_container_width=True, hide_index=True)

                header = csv_header([
                    f"{location_name} daily summary",
                    "Source: Open-Meteo archive API (open-meteo.com)",
                    f"Date range: {sd} to {ed}",
                    f"Data accessed: {access}",
                    f"Total rainfall: {total_rain:.2f} in",
                    "Rain is daily total (precipitation_sum)",
                ])
                df_out = df.round({"temp_mean_f": 1, "temp_max_f": 1, "temp_min_f": 1, "rain_in": 2})
                st.download_button("Download CSV", header + df_out.to_csv(index=False),
                                   file_name=f"{filename_base}.csv", mime="text/csv")

            else:
                df = fetch_hourly(lat, lon, sd, ed)
                agg = process_hourly(df, start_hour, end_hour)
                if agg is None:
                    st.error("No data found for the specified hours.")
                else:
                    fmt_s, fmt_e = fmt_hour(start_hour), fmt_hour(end_hour)
                    overall = agg["avg_temp_f"].mean()
                    hi = agg.loc[agg["avg_temp_f"].idxmax()]
                    lo = agg.loc[agg["avg_temp_f"].idxmin()]
                    total_rain = agg["total_rain_in"].sum()

                    st.subheader(f"Averages ({fmt_s} to {fmt_e}), {location_name}")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Overall mean temp", f"{overall:.1f} °F")
                    c2.metric("Warmest night", f"{hi['avg_temp_f']} °F", str(hi["night_of"]))
                    c3.metric("Coldest night", f"{lo['avg_temp_f']} °F", str(lo["night_of"]))
                    c4.metric("Total rainfall", f"{total_rain:.3f} in")

                    display_cols = [c for c in agg.columns if not c.startswith("temp_f_")]
                    st.subheader("Nightly averages")
                    st.dataframe(agg[display_cols], use_container_width=True, hide_index=True)

                    hourly_cols = ["night_of"] + [c for c in agg.columns if c.startswith("temp_f_")]
                    st.subheader("Hourly temperatures (°F)")
                    st.dataframe(agg[hourly_cols], use_container_width=True, hide_index=True)

                    header = csv_header([
                        f"{location_name} weather summary",
                        "Source: Open-Meteo archive API (open-meteo.com)",
                        f"Date range: {sd} to {ed}",
                        f"Hours averaged: {fmt_s} to {fmt_e} (America/Los_Angeles)",
                        f"Data accessed: {access}",
                        f"Overall mean temp: {overall:.1f} F",
                        f"Warmest night: {hi['avg_temp_f']} F on {hi['night_of']}",
                        f"Coldest night: {lo['avg_temp_f']} F on {lo['night_of']}",
                        f"Total rainfall: {total_rain:.3f} in",
                        "Wind direction averaged using vector (sin/cos) method",
                        "Rain is TOTAL per period (not average)",
                    ])
                    st.download_button("Download CSV", header + agg.to_csv(index=False),
                                       file_name=f"{filename_base}.csv", mime="text/csv")

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                st.error("Open-Meteo is rate-limiting this server. Wait a minute and try again, or use Daily values.")
            else:
                st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

st.info("Data source: Open-Meteo (open-meteo.com)")
