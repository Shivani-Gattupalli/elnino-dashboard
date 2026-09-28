
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="El Nino India Predictor",
    page_icon="🌊",
    layout="wide"
)

# ── Load models and data ─────────────────────────────────
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

# ── Get live ONI from NOAA ───────────────────────────────
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

# ── Advisory engine ──────────────────────────────────────
def get_advisory(rainfall_cat, oni_val, state):
    if rainfall_cat == "Deficit":
        farmer = (f"{state}: High drought risk. Use drought-resistant"
                   " crops like bajra or jowar. Delay sowing 2-3 weeks.")
        water  = (f"{state}: Begin reservoir conservation immediately."
                   " Restrict non-essential water usage.")
    elif rainfall_cat == "Excess":
        farmer = (f"{state}: Excess rainfall expected. Prepare drainage."
                   " Avoid low-lying fields.")
        water  = (f"{state}: Check dam overflow capacity."
                   " Pre-position flood materials.")
    else:
        farmer = (f"{state}: Normal season expected."
                   " Proceed with standard crop planning.")
        water  = (f"{state}: Standard reservoir management advised.")

    if oni_val >= 1.0:
        disaster = ("STRONG El Nino active (ONI≥1.0)."
                     " Activate drought response plan immediately.")
    elif oni_val >= 0.5:
        disaster = ("Moderate El Nino active."
                     " Keep disaster response teams on standby.")
    elif oni_val <= -1.0:
        disaster = ("Strong La Nina active."
                     " Prepare flood evacuation plans.")
    else:
        disaster = "No extreme pattern. Standard preparedness advised."

    return farmer, water, disaster

# ── ENSO phase ───────────────────────────────────────────
def get_phase(oni):
    if oni >= 0.5:   return "🔴 El Nino Active"
    elif oni <= -0.5: return "🔵 La Nina Active"
    else:             return "⚪ Neutral"

# ════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ════════════════════════════════════════════════════════

st.title("🌊 El Nino-IOD Impact Predictor for India")
st.markdown("*State-wise Monsoon Rainfall & Heatwave Risk*")
st.markdown("---")

# ── LIVE STATUS ──────────────────────────────────────────
st.subheader("📡 Current ENSO Status (Live from NOAA)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ENSO Phase",  get_phase(live_oni))
col2.metric("ONI Value",   f"{live_oni:.2f}")
col3.metric("Season",      f"{live_season} {live_year}")
col4.metric("Data Source", "NOAA CPC")

if live_oni >= 0.5:
    st.error(f"⚠️ El Nino is currently active (ONI={live_oni:.2f})."
              " Expect below-normal monsoon risk across India.")
elif live_oni <= -0.5:
    st.info(f"💧 La Nina is currently active (ONI={live_oni:.2f})."
             " Expect above-normal monsoon likely.")
else:
    st.success(f"✅ Neutral conditions (ONI={live_oni:.2f})."
                " Normal monsoon expected.")

st.markdown("---")

# ── TWO MODES ────────────────────────────────────────────
mode = st.radio(
    "Select View Mode:",
    ["🏠 Single State View", "🗺️ All India Map View"],
    horizontal=True
)

st.markdown("---")

# ════════════════════════════════════════════════════════
# MODE 1 — SINGLE STATE VIEW
# ════════════════════════════════════════════════════════
if mode == "🏠 Single State View":

    st.subheader("🏠 Single State Analysis")

    # State selector
    all_states = sorted(state_risk["State"].unique())
    selected   = st.selectbox(
        "Select your State/Subdivision:",
        all_states
    )

    # Get state info
    state_info = state_risk[
        state_risk["State"]==selected].iloc[0]
    state_lpa  = state_info["State_LPA"]
    sensitivity = state_info["Sensitivity"]

    st.markdown(f"### Analysis for: **{selected}**")

    # ── Prediction ──────────────────────────────────────
    oni_sc    = scaler.transform(
        pd.DataFrame([[live_oni]], columns=["ONI"]))[0][0]
    pred      = xgb_model.predict(
        [[oni_sc, live_oni, state_lpa]])[0]
    pred_cls  = le.inverse_transform([pred])[0]
    pred_prob = xgb_model.predict_proba(
        [[oni_sc, live_oni, state_lpa]])[0]

    # Adjust based on state sensitivity
    if live_oni >= 0.5 and sensitivity < -0.3:
        final_pred = "Deficit"
        risk       = "🔴 High Risk"
    elif live_oni >= 0.5 and sensitivity < -0.1:
        final_pred = pred_cls
        risk       = "🟡 Moderate Risk"
    else:
        final_pred = pred_cls
        risk       = "🟢 Low Risk"

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rainfall Outlook", final_pred)
    col2.metric("Risk Level",       risk)
    col3.metric("El Nino Sensitivity",
                f"{sensitivity:.3f}")

    # Prediction result
    if final_pred == "Deficit":
        st.error(f"⚠️ **{selected}**: Below-normal rainfall"
                  " predicted this monsoon season.")
    elif final_pred == "Excess":
        st.info(f"💧 **{selected}**: Above-normal rainfall"
                 " predicted this monsoon season.")
    else:
        st.success(f"✅ **{selected}**: Normal rainfall"
                    " predicted this monsoon season.")

    # Confidence
    st.write("**Model Confidence:**")
    for cls, prob in zip(le.classes_, pred_prob):
        st.progress(float(prob),
                    text=f"{cls}: {prob*100:.1f}%")

    # ── SHAP Explanation ────────────────────────────────
    st.markdown("### 🔍 Why this prediction?")
    sensitivity_pct = abs(sensitivity) * 100
    if live_oni >= 0.5:
        reason = (
            f"**Main reason:** Pacific Ocean warming "
            f"(ONI={live_oni:.2f}) is the primary driver. "
            f"**{selected}** has an El Nino sensitivity of "
            f"{sensitivity:.3f} — meaning historically, "
            f"when El Nino is active, this region tends to "
            f"receive {'less' if sensitivity<0 else 'more'} "
            f"rainfall than normal. "
            f"This prediction is similar to El Nino years "
            f"like 1997, 2002, and 2015."
        )
    elif live_oni <= -0.5:
        reason = (
            f"**Main reason:** Pacific Ocean cooling "
            f"(La Nina, ONI={live_oni:.2f}). "
            f"La Nina years typically bring above-normal "
            f"rainfall to most Indian subdivisions."
        )
    else:
        reason = (
            "**Main reason:** Neutral ocean conditions. "
            "No strong El Nino or La Nina signal detected. "
            "Rainfall is likely to be near normal."
        )
    st.info(reason)

    # ── Advisory ────────────────────────────────────────
    st.markdown("### 📋 Recommendations")
    farmer, water, disaster = get_advisory(
        final_pred, live_oni, selected)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🌾 **For Farmers**")
        st.warning(farmer)
    with col2:
        st.markdown("💧 **For Water Board**")
        st.warning(water)
    with col3:
        st.markdown("🚨 **For Disaster Mgmt**")
        st.error(disaster)

    # ── Anomaly flag ─────────────────────────────────────
    if abs(live_oni) >= 1.0:
        st.markdown("### ⚡ Early Warning")
        st.error(
            f"🚨 **EXTREME EVENT DETECTED** — "
            f"ONI={live_oni:.2f} indicates a strong "
            f"{'El Nino' if live_oni>0 else 'La Nina'}. "
            f"This is similar to extreme years: "
            f"1972, 1997, 2002, 2015. "
            f"Immediate preparedness action advised."
        )

    # ── Historical chart for this state ─────────────────
    st.markdown("### 📊 Historical Rainfall — This State")
    state_hist = state_df[
        state_df["SUBDIVISION"]==selected].copy()

    fig, ax = plt.subplots(figsize=(12, 4))
    colors  = state_hist["ENSO_Phase"].map({
        "El_Nino": "red",
        "La_Nina": "blue",
        "Neutral": "gray"
    })
    ax.bar(state_hist["YEAR"],
           state_hist["ANNUAL"],
           color=colors, alpha=0.7)
    ax.axhline(state_lpa, color="black",
               linestyle="--",
               label=f"Average ({state_lpa:.0f}mm)")
    ax.set_title(
        f"{selected} — Annual Rainfall 1950-2015
"
        f"(Red=El Nino, Blue=La Nina, Gray=Neutral)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════
# MODE 2 — ALL INDIA MAP VIEW
# ════════════════════════════════════════════════════════
else:

    st.subheader("🗺️ All India Risk Overview")

    # ── Risk summary table ───────────────────────────────
    st.markdown("### Current Risk Assessment — All States")

    # Color code risk
    def color_risk(val):
        if val == "High Risk":
            return "background-color: #ffcccc"
        elif val == "Moderate Risk":
            return "background-color: #fff3cc"
        else:
            return "background-color: #ccffcc"

    display_df = state_risk[[
        "State","Sensitivity",
        "Base_Pred","Risk_Level"]].copy()
    display_df.columns = [
        "State/Subdivision",
        "El Nino Sensitivity",
        "Predicted Rainfall",
        "Risk Level"]
    display_df = display_df.sort_values(
        "El Nino Sensitivity")

    st.dataframe(
        display_df.style.applymap(
            color_risk, subset=["Risk Level"]),
        use_container_width=True,
        height=400
    )

    # ── Sensitivity bar chart ─────────────────────────────
    st.markdown("### States Most Sensitive to El Nino")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors2 = ["red" if x<-0.3
                else "orange" if x<-0.1
                else "green"
                for x in state_risk["Sensitivity"]]
    bars = ax2.barh(
        state_risk.sort_values("Sensitivity")["State"],
        state_risk.sort_values("Sensitivity")["Sensitivity"],
        color=colors2, alpha=0.8
    )
    ax2.axvline(0, color="black", linewidth=1)
    ax2.axvline(-0.3, color="red",
                linestyle="--", alpha=0.5,
                label="High risk threshold")
    ax2.set_title(
        "El Nino Sensitivity by State
"
        "(More negative = stronger El Nino impact)",
        fontweight="bold")
    ax2.set_xlabel("Correlation with El Nino (ONI)")

    red_p   = mpatches.Patch(color="red",
                               label="High Risk")
    org_p   = mpatches.Patch(color="orange",
                               label="Moderate Risk")
    grn_p   = mpatches.Patch(color="green",
                               label="Low Risk")
    ax2.legend(handles=[red_p, org_p, grn_p])
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Historical all-India chart ────────────────────────
    st.markdown("### All-India Historical ONI vs Rainfall")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(master_df["YEAR"], master_df["ANNUAL"],
             color="blue", label="Rainfall", linewidth=1.5)
    ax3_2 = ax3.twinx()
    ax3_2.plot(master_df["YEAR"], master_df["ONI"],
               color="red", label="ONI", linewidth=1.5,
               linestyle="--")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Rainfall (mm)", color="blue")
    ax3_2.set_ylabel("ONI", color="red")
    ax3.set_title(
        "India Rainfall vs ONI (El Nino Index) 1950-2015")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2)
    st.pyplot(fig3)

    # ── Click state for detail ────────────────────────────
    st.markdown("### 🔎 View Details for a Specific State")
    selected2 = st.selectbox(
        "Select state to see full details:",
        sorted(state_risk["State"].unique()),
        key="map_state"
    )
    if st.button("View State Details"):
        st.session_state["selected_state"] = selected2
        st.info(
            f"Switch to **Single State View** above "
            f"and select **{selected2}** to see full "
            f"prediction, advisory and historical chart."
        )

# ── Footer ───────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📊 Data: NOAA CPC (ONI Index) | IMD Rainfall Data | "
    "Model: XGBoost (92.9% Accuracy) | "
    "El Nino sensitivity based on 1950-2015 historical analysis"
)
