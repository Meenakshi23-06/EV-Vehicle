
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(page_title="EV Forecast", layout="wide")

# Load model
model = joblib.load('forecasting_ev_model.pkl')

# === Theme toggle ===
theme = st.radio("Choose Theme", ["Dark", "Light"], horizontal=True)

# === Custom CSS based on theme ===
if theme == "Dark":
    st.markdown("""
        <style>
        html, body, .stApp {
            background: linear-gradient(to right, #2e3b4e, #1e1f21);
            color: white;
            font-family: 'Open Sans', sans-serif;
        }
        .highlight { color: #76c7c0; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        html, body, .stApp {
            background-color: #f0f2f6;
            color: black;
            font-family: 'Open Sans', sans-serif;
        }
        .highlight { color: #135589; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown(f"""
    <div style='text-align: center; font-size: 40px; font-weight: bold; margin-top: 20px;'>
        🔋 EV Adoption Forecaster - Washington State
    </div>
""", unsafe_allow_html=True)

# Subtitle
st.markdown(f"""
    <div style='text-align: center; font-size: 20px; margin-bottom: 30px;'>
        Predict and compare Electric Vehicle trends across counties.
    </div>
""", unsafe_allow_html=True)

# === New image (Line 53) ===
st.image("Next-Generation-Electric-Vehicle.jpg", use_container_width=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Select county
st.markdown("## 🔎 Select a County to Forecast")
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

# Filter county data
county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# Forecasting logic
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# Combine historical + forecast data
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Plotly chart
st.subheader(f"📊 Interactive EV Forecast for {county} County")
fig = px.line(combined, x='Date', y='Cumulative EV', color='Source',
              markers=True, title=f'Cumulative EV Trend - {county}')
fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white")
st.plotly_chart(fig, use_container_width=True)

# Forecast percentage growth
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(f"In **{county}**, EV adoption is expected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years.")
else:
    st.warning("Cannot compute forecast change due to zero historical EVs.")

# CSV download
csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("📁 Download Forecast Data as CSV", data=csv, file_name=f"{county}_EV_Forecast.csv", mime="text/csv")

# Multi-county comparison
with st.expander("🔍 Compare up to 3 Counties"):
    multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)

    if multi_counties:
        comparison_data = []

        for cty in multi_counties:
            cty_df = df[df['County'] == cty].sort_values("Date")
            cty_code = cty_df['county_encoded'].iloc[0]

            hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = cty_df['months_since_start'].max()
            last_date = cty_df['Date'].max()

            future_rows_cty = []
            for i in range(1, forecast_horizon + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cum = cum_ev[-6:]
                ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

                new_row = {
                    'months_since_start': months_since,
                    'county_encoded': cty_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }
                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)

                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)

            hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

            fc_df = pd.DataFrame(future_rows_cty)
            fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

            combined_cty = pd.concat([
                hist_cum[['Date', 'Cumulative EV']],
                fc_df[['Date', 'Cumulative EV']]
            ], ignore_index=True)

            combined_cty['County'] = cty
            comparison_data.append(combined_cty)

        comp_df = pd.concat(comparison_data, ignore_index=True)
        fig2 = px.line(comp_df, x="Date", y="Cumulative EV", color="County", title="EV Adoption: Multi-County Comparison", markers=True)
        fig2.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #bbb;">
    <div style='text-align: center; padding-top: 10px; font-size: 14px;'>
        🚗 Built by Meenakshi for <span class='highlight'>AICTE Internship Cycle 2</span> | 
    </div>
""", unsafe_allow_html=True)
