"""
Weather Data Averager — Visual Crossing version
Fetches hourly weather from Visual Crossing and computes averages.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from io import StringIO

# ── Config ──
st.set_page_config(page_title="Weather Data Averager", layout="wide")
st.title("Weather Data Averager")

# Get API key from environment variable
VISUAL_CROSSING_KEY = os.getenv("VISUAL_CROSSING_API_KEY")

if not VISUAL_CROSSING_KEY:
    st.error("API key not configured. Please contact administrator.")
    st.stop()

# ── Helper functions ──

def avg_wind_dir(group):
    """Average wind direction using vector mean."""
    rads = np.radians(group["wind_dir_deg"])
    mean_u = np.sin(rads).mean()
    mean_v = np.cos(rads).mean()
    return round(np.degrees(np.arctan2(mean_u, mean_v)) % 360, 1)


def deg_to_compass(deg):
    """Convert degrees to 16-point compass direction."""
    dirs = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    return dirs[int((deg + 11.25) / 22.5) % 16]


def fmt_hour(h):
    """Format 0-23 hour as '10 PM' style."""
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


@st.cache_data(ttl=3600)
def fetch_visualcrossing(lat, lon, start_date, end_date):
    """Fetch Visual Crossing data."""
    
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/v1/timeline"
    
    # Visual Crossing uses lat,lon format for location
    location = f"{lat},{lon}"
    
    params = {
        "key": VISUAL_CROSSING_KEY,
        "unitGroup": "us",
        "contentType": "json",
        "include": "hours",
        "elements": "temp,windspeed,winddir,precip",
    }
    
    endpoint = f"{url}/{location}/{start_date}/{end_date}"
    
    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()
    data = resp.json()

    all_rows = []
    for day in data["days"]:
        date = day["datetime"]
        if "hours" not in day:
            continue
        
        for hour_data in day["hours"]:
            all_rows.append({
                "datetime": datetime.strptime(f"{date} {hour_data['datetime']}", "%Y-%m-%d %H:%M:%S"),
                "temp_f": hour_data.get("temp"),
                "wind_speed_mph": hour_data.get("windspeed", 0),
                "wind_dir_deg": hour_data.get("winddir", 0),
                "rain_in": hour_data.get("precip", 0),
            })
    
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["temp_f"])
    
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date

    return df


def process_data(df, start_hour, end_hour, location_name, start_date, end_date, source):
    """Process hourly data into nightly averages."""
    
    if df.empty:
        return None, "No data to process."
    
    # Filter to time window
    if start_hour > end_hour:
        hour_mask = (df["hour"] >= start_hour) | (df["hour"] <= end_hour)
        early_hours = df["hour"] <= end_hour
    else:
        hour_mask = (df["hour"] >= start_hour) & (df["hour"] <= end_hour)
        early_hours = pd.Series(False, index=df.index)

    filtered = df[hour_mask].copy()

    if filtered.empty:
        return None, "No data found for the specified hours."

    filtered["night_of"] = filtered["date"] - pd.to_timedelta(
        early_hours[hour_mask].astype(int), unit="D"
    )

    # Wind direction
    filtered["wind_dir_rad"] = np.radians(filtered["wind_dir_deg"])
    filtered["wind_u"] = np.sin(filtered["wind_dir_rad"])
    filtered["wind_v"] = np.cos(filtered["wind_dir_rad"])

    grouped = filtered.groupby("night_of")

    # Aggregate
    agg = grouped.agg(
        avg_temp_f=("temp_f", "mean"),
        avg_wind_mph=("wind_speed_mph", "mean"),
        total_rain_in=("rain_in", "sum"),
    ).reset_index()

    # Round
    for col in agg.columns:
        if col.startswith("avg_") or col.startswith("total_"):
            agg[col] = agg[col].round(1)

    # Wind direction (vector average)
    wind_dirs = grouped.apply(avg_wind_dir).reset_index()
    wind_dirs.columns = ["night_of", "avg_wind_dir_deg"]
    agg = agg.merge(wind_dirs, on="night_of")
    agg["avg_wind_dir_compass"] = agg["avg_wind_dir_deg"].apply(deg_to_compass)

    # Hourly temps as separate columns
    hourly_pivot = filtered.sort_values("datetime").pivot_table(
        index="night_of",
        columns="hour",
        values="temp_f",
        aggfunc="first"
    ).reset_index()
    
    # Flatten column names
    hourly_pivot.columns = [f"temp_f_{col:02d}" if isinstance(col, int) else col 
                            for col in hourly_pivot.columns]
    
    agg = agg.merge(hourly_pivot, on="night_of", how="left")

    # Column order for display
    col_order = [
        "night_of", "avg_temp_f", "avg_wind_mph", "avg_wind_dir_deg",
        "avg_wind_dir_compass", "total_rain_in",
    ]
    
    # Add hourly temp columns in order
    hourly_cols = sorted([c for c in agg.columns if c.startswith("temp_f_")])
    col_order.extend(hourly_cols)
    
    agg = agg[[c for c in col_order if c in agg.columns]]

    # Summary stats
    fmt_s = fmt_hour(start_hour)
    fmt_e = fmt_hour(end_hour)
    overall_temp = agg["avg_temp_f"].mean()
    max_row = agg.loc[agg["avg_temp_f"].idxmax()]
    min_row = agg.loc[agg["avg_temp_f"].idxmin()]
    total_rain = agg["total_rain_in"].sum()

    # Build CSV in memory
    access_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_name = location_name.lower().replace(" ", "_").replace(",", "")
    date_s = start_date.replace("-", "_")
    date_e = end_date.replace("-", "_")
    filename = f"{safe_name}_weather_{date_s}_to_{date_e}.csv"

    csv_header = (
        f"# {location_name} weather summary\n"
        f"# Source: {source}\n"
        f"# Date range: {start_date} to {end_date}\n"
        f"# Hours averaged: {fmt_s} to {fmt_e} (America/Los_Angeles)\n"
        f"# Data accessed: {access_date}\n"
        f"# Overall mean temp: {overall_temp:.1f} F\n"
        f"# Warmest night: {max_row['avg_temp_f']} F on {max_row['night_of']}\n"
        f"# Coldest night: {min_row['avg_temp_f']} F on {min_row['night_of']}\n"
        f"# Total rainfall: {total_rain:.3f} in\n"
        f"# Wind direction averaged using vector (sin/cos) method\n"
        f"# Rain is TOTAL per period (not average)\n"
        f"#\n"
    )

    csv_buffer = StringIO()
    csv_buffer.write(csv_header)
    agg.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    return {
        "agg": agg,
        "overall_temp": overall_temp,
        "max_row": max_row,
        "min_row": min_row,
        "total_rain": total_rain,
        "fmt_s": fmt_s,
        "fmt_e": fmt_e,
        "csv_content": csv_content,
        "filename": filename,
    }, None


# ── Sidebar inputs ──
with st.sidebar:
    st.header("Settings")
    
    location_name = st.text_input("Location name", value="Madera CA")
    lat = st.number_input("Latitude", value=36.9613, format="%.4f")
    lon = st.number_input("Longitude", value=-120.0607, format="%.4f")
    start_date = st.date_input("Start date", value=datetime(2026, 4, 14))
    end_date = st.date_input("End date", value=datetime(2026, 5, 7))
    start_hour = st.slider("Start hour (0-23)", min_value=0, max_value=23, value=22)
    end_hour = st.slider("End hour (0-23)", min_value=0, max_value=23, value=5)

# ── Main ──
if st.button("Fetch Data", type="primary"):
    with st.spinner("Fetching data..."):
        try:
            df = fetch_visualcrossing(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            result, error = process_data(df, start_hour, end_hour, location_name, 
                                        start_date.strftime("%Y-%m-%d"), 
                                        end_date.strftime("%Y-%m-%d"),
                                        "Visual Crossing weather data")
            
            if error:
                st.error(error)
            else:
                agg = result["agg"]
                
                # Summary stats
                st.subheader(f"Weather Averages ({result['fmt_s']} – {result['fmt_e']}), {location_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Overall Mean Temp", f"{result['overall_temp']:.1f} °F")
                col2.metric("Warmest Night", f"{result['max_row']['avg_temp_f']} °F")
                col3.metric("Coldest Night", f"{result['min_row']['avg_temp_f']} °F")
                col4.metric("Total Rainfall", f"{result['total_rain']:.3f} in")
                
                # Data table
                st.subheader("Nightly Averages")
                display_cols = [c for c in agg.columns if not c.startswith("temp_f_")]
                st.dataframe(agg[display_cols], use_container_width=True, hide_index=True)
                
                # Hourly temps table
                st.subheader("Hourly Temperatures (°F)")
                hourly_cols = ["night_of"] + sorted([c for c in agg.columns if c.startswith("temp_f_")])
                st.dataframe(agg[hourly_cols], use_container_width=True, hide_index=True)
                
                # Download button
                st.download_button(
                    label="Download CSV",
                    data=result["csv_content"],
                    file_name=result["filename"],
                    mime="text/csv",
                )
        
        except Exception as e:
            st.error(f"Error: {e}")

st.info("Data source: Visual Crossing weather data (visualcrossing.com)")
