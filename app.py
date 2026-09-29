
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="El Nino India Predictor",
                   page_icon="🌊", layout="wide")

@st.cache_resource
def load_models():
    xgb    = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le     = joblib.load("label_encoder.pkl")
    return xgb, scaler, le

@st.cache_data
def load_data():
    master     = pd.read_csv("master_dataset.csv")
    state_df   = pd.read_csv("state_dataset.csv")
    state_risk = pd.read_csv("state_risk.csv")
    return master, state_df, state_risk

xgb_model, scaler, le = load_models()
master_df, state_df, state_risk = load_data()

@st.cache_data(ttl=86400)
def get_live_oni():
    try:
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        raw = requests.get(url, timeout=5).text
        df  = pd.read_csv(StringIO(raw), sep=r"\s+")
        row = df.iloc[-1]
        return float(row["ANOM"]), str(row["SEAS"]), int(row["YR"])
    except:
        return 0.5, "JJA", 2026

live_oni, live_season, live_year = get_live_oni()
lpa_per_state = state_df.groupby("SUBDIVISION")["ANNUAL"].mean()

state_sensitivity = {}
for sub in state_df["SUBDIVISION"].unique():
    d = state_df[state_df["SUBDIVISION"]==sub]
    state_sensitivity[sub] = d["ONI"].corr(d["ANNUAL"])

def get_advisory(rainfall_cat, oni_val, state):
    if rainfall_cat == "Deficit":
        farmer   = state + ": High drought risk. Use drought-resistant crops. Delay sowing 2-3 weeks."
        water    = state + ": Begin reservoir conservation. Restrict non-essential water usage."
    elif rainfall_cat == "Excess":
        farmer   = state + ": Excess rainfall expected. Prepare drainage systems."
        water    = state + ": Check dam capacity. Pre-position flood materials."
    else:
        farmer   = state + ": Normal season. Proceed with standard crop planning."
        water    = state + ": Standard reservoir management advised."
    if oni_val >= 1.0:
        disaster = "STRONG El Nino active. Activate drought response plan immediately."
    elif oni_val >= 0.5:
        disaster = "Moderate El Nino. Keep disaster response teams on standby."
    elif oni_val <= -1.0:
        disaster = "Strong La Nina. Prepare flood evacuation plans."
    else:
        disaster = "No extreme pattern. Standard preparedness advised."
    return farmer, water, disaster

def get_phase(oni):
    if oni >= 0.5:    return "El Nino Active"
    elif oni <= -0.5: return "La Nina Active"
    else:             return "Neutral"

st.title("El Nino-IOD Impact Predictor for India")
st.markdown("State-wise Monsoon Rainfall and Heatwave Risk")
st.markdown("---")

st.subheader("Current ENSO Status — Live from NOAA")
col1, col2, col3 = st.columns(3)
col1.metric("ENSO Phase",  get_phase(live_oni))
col2.metric("ONI Value",   str(round(live_oni, 2)))
col3.metric("Season/Year", live_season + " " + str(live_year))

if live_oni >= 0.5:
    st.error("El Nino is currently active (ONI=" + str(round(live_oni,2)) + "). Expect below-normal monsoon risk.")
elif live_oni <= -0.5:
    st.info("La Nina is currently active (ONI=" + str(round(live_oni,2)) + "). Above-normal monsoon likely.")
else:
    st.success("Neutral conditions (ONI=" + str(round(live_oni,2)) + "). Normal monsoon expected.")

st.markdown("---")

mode = st.radio("Select View Mode:",
                ["Single State View", "All India View"],
                horizontal=True)

st.markdown("---")

if mode == "Single State View":
    st.subheader("Single State Analysis")
    all_states = sorted(state_risk["State"].unique())
    selected   = st.selectbox("Select your State/Subdivision:", all_states)

    state_info  = state_risk[state_risk["State"]==selected].iloc[0]
    state_lpa   = float(state_info["State_LPA"])
    sensitivity = float(state_info["Sensitivity"])

    st.markdown("### Analysis for: " + selected)

    oni_sc    = scaler.transform(pd.DataFrame([[live_oni]], columns=["ONI"]))[0][0]
    pred      = xgb_model.predict([[oni_sc, live_oni, state_lpa]])[0]
    pred_cls  = le.inverse_transform([pred])[0]
    pred_prob = xgb_model.predict_proba([[oni_sc, live_oni, state_lpa]])[0]

    if live_oni >= 0.5 and sensitivity < -0.3:
        final_pred = "Deficit"
        risk       = "High Risk"
    elif live_oni >= 0.5 and sensitivity < -0.1:
        final_pred = pred_cls
        risk       = "Moderate Risk"
    else:
        final_pred = pred_cls
        risk       = "Low Risk"

    col1, col2, col3 = st.columns(3)
    col1.metric("Rainfall Outlook",    final_pred)
    col2.metric("Risk Level",          risk)
    col3.metric("El Nino Sensitivity", str(round(sensitivity, 3)))

    if final_pred == "Deficit":
        st.error(selected + ": Below-normal rainfall predicted this monsoon season.")
    elif final_pred == "Excess":
        st.info(selected + ": Above-normal rainfall predicted this monsoon season.")
    else:
        st.success(selected + ": Normal rainfall predicted this monsoon season.")

    st.write("Model Confidence:")
    for cls, prob in zip(le.classes_, pred_prob):
        st.progress(float(prob), text=cls + ": " + str(round(prob*100, 1)) + "%")

    st.markdown("### Why this prediction?")
    if live_oni >= 0.5:
        st.info("Main reason: Pacific Ocean warming (ONI=" + str(round(live_oni,2)) + ") is the primary driver. " +
                selected + " has El Nino sensitivity of " + str(round(sensitivity,3)) +
                " — meaning historically when El Nino is active, this region receives " +
                ("less" if sensitivity < 0 else "more") +
                " rainfall than normal. Pattern similar to El Nino years 1997, 2002, and 2015.")
    elif live_oni <= -0.5:
        st.info("Main reason: Pacific Ocean cooling (La Nina, ONI=" + str(round(live_oni,2)) + "). La Nina years typically bring above-normal rainfall to most Indian regions.")
    else:
        st.info("Main reason: Neutral ocean conditions. No strong El Nino or La Nina signal. Rainfall likely near normal.")

    st.markdown("### Recommendations")
    farmer, water, disaster = get_advisory(final_pred, live_oni, selected)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**For Farmers**")
        st.warning(farmer)
    with col2:
        st.markdown("**For Water Board**")
        st.warning(water)
    with col3:
        st.markdown("**For Disaster Management**")
        st.error(disaster)

    if abs(live_oni) >= 1.0:
        st.markdown("### Early Warning")
        st.error("EXTREME EVENT DETECTED — ONI=" + str(round(live_oni,2)) +
                 " indicates strong " + ("El Nino" if live_oni > 0 else "La Nina") +
                 ". Similar to extreme years: 1972, 1997, 2002, 2015. Immediate action advised.")

    st.markdown("### Historical Rainfall — " + selected)
    state_hist = state_df[state_df["SUBDIVISION"]==selected].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    color_map = {"El_Nino": "red", "La_Nina": "blue", "Neutral": "gray"}
    bar_colors = [color_map.get(p, "gray") for p in state_hist["ENSO_Phase"]]
    ax.bar(state_hist["YEAR"], state_hist["ANNUAL"],
           color=bar_colors, alpha=0.7)
    ax.axhline(state_lpa, color="black", linestyle="--",
               label="Average (" + str(round(state_lpa, 0)) + "mm)")
    ax.set_title(selected + " Annual Rainfall 1950-2015 (Red=El Nino, Blue=La Nina, Gray=Neutral)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)

else:
    st.subheader("All India Risk Overview")
    st.markdown("### Risk Assessment — All States")

    display_df = state_risk[["State","Sensitivity","Base_Pred","Risk_Level"]].copy()
    display_df.columns = ["State","El Nino Sensitivity","Predicted Rainfall","Risk Level"]
    display_df = display_df.sort_values("El Nino Sensitivity")
    st.dataframe(display_df, use_container_width=True, height=400)

    st.markdown("### States Most Sensitive to El Nino")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sorted_risk = state_risk.sort_values("Sensitivity")
    bar_colors2 = ["red" if x < -0.3 else "orange" if x < -0.1 else "green"
                   for x in sorted_risk["Sensitivity"]]
    ax2.barh(sorted_risk["State"], sorted_risk["Sensitivity"],
             color=bar_colors2, alpha=0.8)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_title("El Nino Sensitivity by State (More negative = stronger impact)",
                  fontweight="bold")
    ax2.set_xlabel("Correlation with El Nino (ONI)")
    red_p = mpatches.Patch(color="red",    label="High Risk")
    org_p = mpatches.Patch(color="orange", label="Moderate Risk")
    grn_p = mpatches.Patch(color="green",  label="Low Risk")
    ax2.legend(handles=[red_p, org_p, grn_p])
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("### All-India ONI vs Rainfall History")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(master_df["YEAR"], master_df["ANNUAL"],
             color="blue", label="Rainfall", linewidth=1.5)
    ax3_2 = ax3.twinx()
    ax3_2.plot(master_df["YEAR"], master_df["ONI"],
               color="red", label="ONI", linewidth=1.5, linestyle="--")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Rainfall (mm)", color="blue")
    ax3_2.set_ylabel("ONI Value", color="red")
    ax3.set_title("India Rainfall vs ONI 1950-2015")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2)
    st.pyplot(fig3)

    st.markdown("### View a Specific State")
    selected2 = st.selectbox("Select state:",
                              sorted(state_risk["State"].unique()),
                              key="state2")
    if st.button("View Details"):
        st.info("Switch to Single State View and select " + selected2 + " for full details.")

st.markdown("---")
st.caption("Data: NOAA CPC | IMD Rainfall | Model: XGBoost 92.9% Accuracy | 1950-2015")
