
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

st.set_page_config(
    page_title="India Monsoon Risk",
    page_icon="🌧️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size:2.2rem; font-weight:bold;
    color:#1B2A4A; text-align:center; padding:10px;
}
.sub-title {
    font-size:1rem; color:#666;
    text-align:center; margin-bottom:15px;
}
.status-card {
    background:linear-gradient(135deg,#1B2A4A,#2E4A7A);
    color:white; padding:18px; border-radius:12px;
    text-align:center; margin:4px;
}
.risk-high {
    background:#FF4444; color:white; padding:14px;
    border-radius:10px; font-size:1.1rem;
    font-weight:bold; text-align:center; margin:4px;
}
.risk-moderate {
    background:#FF8C00; color:white; padding:14px;
    border-radius:10px; font-size:1.1rem;
    font-weight:bold; text-align:center; margin:4px;
}
.risk-low {
    background:#28A745; color:white; padding:14px;
    border-radius:10px; font-size:1.1rem;
    font-weight:bold; text-align:center; margin:4px;
}
.action-box {
    background:#FFF3CD; border-left:5px solid #FFC107;
    padding:15px; border-radius:8px; margin:8px 0;
}
.info-box {
    background:#E8F4FD; border-left:5px solid #2196F3;
    padding:15px; border-radius:8px; margin:8px 0;
}
.timeline-box {
    background:#F0F4FF; border:2px solid #1B2A4A;
    padding:18px; border-radius:12px; margin:10px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    xgb_rain = joblib.load("xgb_model.pkl")
    xgb_heat = joblib.load("xgb_heat_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    iod_sc   = joblib.load("iod_scaler.pkl")
    le_rain  = joblib.load("label_encoder.pkl")
    le_heat  = joblib.load("le_heat.pkl")
    return xgb_rain, xgb_heat, scaler, iod_sc, le_rain, le_heat

@st.cache_data
def load_data():
    master     = pd.read_csv("master_dataset.csv")
    state_df   = pd.read_csv("state_dataset.csv")
    state_risk = pd.read_csv("state_risk.csv")
    iod_data   = pd.read_csv("iod_data.csv")
    return master, state_df, state_risk, iod_data

xgb_rain,xgb_heat,scaler,iod_sc,le_rain,le_heat = load_models()
master_df,state_df,state_risk,iod_df = load_data()

@st.cache_data(ttl=86400)
def get_live():
    try:
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        raw = requests.get(url, timeout=10).text
        df  = pd.read_csv(StringIO(raw), sep=r"\\s+")
        recent   = df.tail(12).reset_index(drop=True)
        live_oni = float(df.iloc[-1]["ANOM"])
        sea      = str(df.iloc[-1]["SEAS"])
        yr       = int(df.iloc[-1]["YR"])

        # Find when El Nino started from recent data
        start_season = None
        months_active = 0
        if live_oni >= 0.5:
            for _, row in recent.iterrows():
                if float(row["ANOM"]) >= 0.5:
                    if start_season is None:
                        start_season = str(row["SEAS"])+" "+str(int(row["YR"]))
                    months_active += 1
        elif live_oni <= -0.5:
            for _, row in recent.iterrows():
                if float(row["ANOM"]) <= -0.5:
                    if start_season is None:
                        start_season = str(row["SEAS"])+" "+str(int(row["YR"]))
                    months_active += 1

        # Estimate end
        if live_oni >= 1.5:
            est_end = "Early 2027 (Strong El Nino typically lasts 12-18 months)"
        elif live_oni >= 0.5:
            remaining = max(3, 9 - months_active)
            est_end   = f"In about {remaining} months (El Nino started {months_active} seasons ago)"
        elif live_oni <= -1.0:
            remaining = max(3, 12 - months_active)
            est_end   = f"In about {remaining} months (La Nina started {months_active} seasons ago)"
        elif live_oni <= -0.5:
            remaining = max(3, 9 - months_active)
            est_end   = f"In about {remaining} months"
        else:
            est_end = "Currently normal — no active event"

        if start_season is None:
            start_season = "Not applicable (ocean is currently normal)"

        return live_oni, 0.2, sea, yr, recent, start_season, est_end, months_active
    except Exception as e:
        return 0.5, 0.2, "JJA", 2026, None, "March 2026", "December 2026 (estimated)", 4

live_oni,live_iod,live_season,live_year,recent_oni,start_season,est_end,months_active = get_live()

lpa_per_state = state_df.groupby("SUBDIVISION")["ANNUAL"].mean()
state_sensitivity = {}
for sub in state_df["SUBDIVISION"].unique():
    d = state_df[state_df["SUBDIVISION"]==sub]
    state_sensitivity[sub] = d["ONI"].corr(d["ANNUAL"])

def classify_heatwave(oni, iod):
    score = oni + (iod * 0.5)
    if score>=1.0: return "High"
    elif score>=0.4: return "Moderate"
    else: return "Low"

if live_oni >= 1.0:
    phase_name = "STRONG El Nino"
    phase_emoji = "🔴"
    phase_plain = "The Pacific Ocean is MUCH warmer than normal. This strongly suppresses India's monsoon."
elif live_oni >= 0.5:
    phase_name = "El Nino Active"
    phase_emoji = "🟠"
    phase_plain = "The Pacific Ocean is warmer than normal. India's monsoon is at risk of being weaker."
elif live_oni <= -1.0:
    phase_name = "STRONG La Nina"
    phase_emoji = "🔵"
    phase_plain = "The Pacific Ocean is MUCH cooler than normal. India's monsoon is likely to be very good."
elif live_oni <= -0.5:
    phase_name = "La Nina Active"
    phase_emoji = "🔵"
    phase_plain = "The Pacific Ocean is cooler than normal. India's monsoon is expected to be good."
else:
    phase_name = "Normal Conditions"
    phase_emoji = "🟢"
    phase_plain = "Ocean temperatures are normal. No major disruption to India's monsoon expected."

# ════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════
st.markdown(
    "<div class=\"main-title\">🌧️ India Monsoon & Heatwave Risk Predictor</div>",
    unsafe_allow_html=True)
st.markdown(
    "<div class=\"sub-title\">Real-time ocean data from NOAA → "
    "Your state's risk → What to do right now</div>",
    unsafe_allow_html=True)
st.markdown("---")

# ════════════════════════════════════════════════════════
# OCEAN STATUS
# ════════════════════════════════════════════════════════
st.markdown("### 🌊 What Is the Ocean Doing Right Now?")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown(
        f"<div class=\"status-card\">"
        f"<div style=\"font-size:3rem\">{phase_emoji}</div>"
        f"<div style=\"font-size:1.6rem;font-weight:bold;margin:5px\">"
        f"{phase_name}</div>"
        f"<div style=\"font-size:1rem;opacity:0.9\">{phase_plain}</div>"
        f"<div style=\"margin-top:12px;font-size:0.85rem;opacity:0.75\">"
        f"Pacific Ocean (ONI) = {round(live_oni,2)} | "
        f"Indian Ocean (IOD) = {round(live_iod,2)}</div>"
        f"</div>",
        unsafe_allow_html=True)

with col2:
    st.markdown("#### ⏱️ Duration & Timeline")
    st.markdown(
        f"<div class=\"timeline-box\">"
        f"<table width=\"100%\"><tr>"
        f"<td style=\"padding:8px;vertical-align:top\">"
        f"<span style=\"font-size:0.85rem;color:#666\">📅 WHEN DID IT START?</span><br>"
        f"<b style=\"color:#1B2A4A\">{start_season}</b></td></tr>"
        f"<tr><td style=\"padding:8px;vertical-align:top\">"
        f"<span style=\"font-size:0.85rem;color:#666\">⏳ HOW LONG ACTIVE?</span><br>"
        f"<b style=\"color:#1B2A4A\">{months_active} seasons (≈{months_active} months)</b>"
        f"</td></tr>"
        f"<tr><td style=\"padding:8px;vertical-align:top\">"
        f"<span style=\"font-size:0.85rem;color:#666\">🔚 WHEN WILL IT END?</span><br>"
        f"<b style=\"color:#CC0000\">{est_end}</b>"
        f"</td></tr></table>"
        f"</div>",
        unsafe_allow_html=True)

# ONI Timeline chart
if recent_oni is not None and len(recent_oni) > 0:
    st.markdown("#### 📈 Ocean Temperature — Last 12 Seasons (Live from NOAA)")
    fig_t, ax_t = plt.subplots(figsize=(12,3))
    fig_t.patch.set_facecolor("#F8F9FA")
    ax_t.set_facecolor("#F8F9FA")
    try:
        seasons = [str(r["SEAS"])+" "+str(int(r["YR"]))
                   for _,r in recent_oni.iterrows()]
        values  = [float(r["ANOM"]) for _,r in recent_oni.iterrows()]
        colors  = ["#FF4444" if v>=0.5 else "#4444FF" if v<=-0.5 else "#888888"
                   for v in values]
        ax_t.bar(range(len(seasons)), values, color=colors, alpha=0.85,
                 edgecolor="white")
        ax_t.axhline(0.5,  color="#FF4444",linestyle="--",linewidth=1.5,
                     label="El Nino starts (+0.5)")
        ax_t.axhline(-0.5, color="#4444FF",linestyle="--",linewidth=1.5,
                     label="La Nina starts (-0.5)")
        ax_t.axhline(0, color="black", linewidth=0.5)
        ax_t.set_xticks(range(len(seasons)))
        ax_t.set_xticklabels(seasons, rotation=45, ha="right", fontsize=8)
        ax_t.set_ylabel("Ocean Temperature Anomaly")
        ax_t.set_title(
            "Pacific Ocean Temperature — Last 12 Months | "
            "Red = El Nino | Blue = La Nina",
            fontweight="bold", color="#2C3E50", fontsize=10)
        ax_t.legend(fontsize=8)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        st.pyplot(fig_t)
    except:
        st.info("Timeline chart loading...")

st.markdown("---")

# ════════════════════════════════════════════════════════
# STATE SELECTOR
# ════════════════════════════════════════════════════════
st.markdown("### 📍 What Does This Mean For YOUR State?")
all_states = ["-- Select Your State --"] + sorted(state_risk["State"].unique())
selected   = st.selectbox(
    "Choose your state to get a personalized risk report:",
    all_states, index=0)

if selected == "-- Select Your State --":
    hr = len(state_risk[state_risk["Risk_Level"]=="High Risk"])
    mr = len(state_risk[state_risk["Risk_Level"]=="Moderate Risk"])
    lr = len(state_risk[state_risk["Risk_Level"]=="Low Risk"])
    st.markdown(
        "<div class=\"info-box\"><b>👆 Select your state above</b> "
        "to get your personalized monsoon and heatwave risk report, "
        "with clear advice on what to do right now.</div>",
        unsafe_allow_html=True)
    st.markdown("**India Overview This Season:**")
    c1,c2,c3 = st.columns(3)
    c1.markdown(
        f"<div class=\"risk-high\">🔴 {hr} States<br>"
        f"<small>High Risk</small></div>",
        unsafe_allow_html=True)
    c2.markdown(
        f"<div class=\"risk-moderate\">🟠 {mr} States<br>"
        f"<small>Moderate Risk</small></div>",
        unsafe_allow_html=True)
    c3.markdown(
        f"<div class=\"risk-low\">🟢 {lr} States<br>"
        f"<small>Low Risk</small></div>",
        unsafe_allow_html=True)

else:
    state_info  = state_risk[state_risk["State"]==selected].iloc[0]
    state_lpa   = float(state_info["State_LPA"])
    sensitivity = float(state_info["Sensitivity"])

    oni_sc    = scaler.transform([[live_oni]])[0][0]
    iod_sc_v  = iod_sc.transform([[live_iod]])[0][0]
    rain_pred = xgb_rain.predict(
        [[oni_sc,iod_sc_v,live_oni,live_iod,state_lpa]])[0]
    rain_cls  = le_rain.inverse_transform([rain_pred])[0]
    rain_prob = xgb_rain.predict_proba(
        [[oni_sc,iod_sc_v,live_oni,live_iod,state_lpa]])[0]
    heat_pred = xgb_heat.predict(
        [[oni_sc,iod_sc_v,live_oni,live_iod]])[0]
    heat_cls  = le_heat.inverse_transform([heat_pred])[0]

    if live_oni>=0.5 and sensitivity<-0.3:
        final_rain = "Deficit"
    elif live_oni>=0.5 and sensitivity<-0.1:
        final_rain = rain_cls
    else:
        final_rain = rain_cls

    # Risk banners
    if final_rain=="Deficit":
        r_emoji,r_text = "⚠️","BELOW NORMAL RAINFALL EXPECTED"
        r_plain = "Less rain than usual this monsoon. Drought risk is real."
        r_class = "risk-high"
    elif final_rain=="Excess":
        r_emoji,r_text = "🌊","ABOVE NORMAL RAINFALL EXPECTED"
        r_plain = "More rain than usual. Flooding in low areas is possible."
        r_class = "risk-moderate"
    else:
        r_emoji,r_text = "✅","NORMAL RAINFALL EXPECTED"
        r_plain = "Rainfall likely to be around the usual average. Good season."
        r_class = "risk-low"

    if heat_cls=="High":
        h_emoji,h_text = "🔥","HIGH HEATWAVE RISK"
        h_plain = "Severe heat likely. Protect yourself April-June."
        h_class = "risk-high"
    elif heat_cls=="Moderate":
        h_emoji,h_text = "🌡️","MODERATE HEAT RISK"
        h_plain = "Hotter than usual. Stay hydrated."
        h_class = "risk-moderate"
    else:
        h_emoji,h_text = "😊","LOW HEAT RISK"
        h_plain = "Normal temperatures expected."
        h_class = "risk-low"

    st.markdown(f"## 📍 {selected} — Your Risk Report")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class=\"{r_class}\">{r_emoji} {r_text}<br>"
            f"<span style=\"font-size:0.9rem;font-weight:normal\">"
            f"{r_plain}</span></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class=\"{h_class}\">{h_emoji} {h_text}<br>"
            f"<span style=\"font-size:0.9rem;font-weight:normal\">"
            f"{h_plain}</span></div>",
            unsafe_allow_html=True)

    st.markdown("")

    # Timeline for this state
    st.markdown(f"### ⏱️ How Long Will This Affect {selected}?")
    st.markdown(
        f"<div class=\"timeline-box\">"
        f"<table width=\"100%\"><tr>"
        f"<td width=\"33%\" style=\"text-align:center;padding:10px\">"
        f"<div style=\"font-size:1.5rem\">🌊</div>"
        f"<div style=\"font-weight:bold;color:#1B2A4A\">{phase_name} Started</div>"
        f"<div style=\"color:#555;font-size:0.95rem\">{start_season}</div></td>"
        f"<td width=\"33%\" style=\"text-align:center;padding:10px;"
        f"border-left:2px dashed #CBD5E0;border-right:2px dashed #CBD5E0\">"
        f"<div style=\"font-size:1.5rem\">📍</div>"
        f"<div style=\"font-weight:bold;color:#1B2A4A\">Right Now</div>"
        f"<div style=\"color:#555;font-size:0.95rem\">"
        f"{live_season} {live_year} | ONI = {round(live_oni,2)}</div>"
        f"<div style=\"color:#555;font-size:0.85rem\">"
        f"Active for {months_active} months</div></td>"
        f"<td width=\"33%\" style=\"text-align:center;padding:10px\">"
        f"<div style=\"font-size:1.5rem\">🔚</div>"
        f"<div style=\"font-weight:bold;color:#CC0000\">Expected End</div>"
        f"<div style=\"color:#CC0000;font-size:0.9rem\">{est_end}</div></td>"
        f"</tr></table>"
        f"<div style=\"text-align:center;margin-top:6px;"
        f"font-size:0.8rem;color:#888\">"
        f"Historical average: El Nino lasts 9-12 months | "
        f"Based on NOAA CPC data"
        f"</div></div>",
        unsafe_allow_html=True)

    # Actions
    st.markdown("---")
    st.markdown(f"### ⚡ What Should You Do RIGHT NOW?")
    st.markdown(
        f"*Time-sensitive actions for {selected} based on "
        f"current {phase_name} conditions*")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("#### 👨‍🌾 Farmers")
        if final_rain=="Deficit":
            st.markdown(
                "<div class=\"action-box\">"
                "<b>🚨 Act Now — Drought Risk</b><br><br>"
                "✅ Switch to drought-resistant crops<br>"
                "&nbsp;&nbsp;&nbsp;(bajra, jowar, pulses)<br><br>"
                "✅ Delay sowing by 2-3 weeks<br><br>"
                "✅ Install drip irrigation<br><br>"
                "✅ Apply for crop insurance now<br><br>"
                "❌ Avoid water-heavy crops<br>"
                "&nbsp;&nbsp;&nbsp;(sugarcane, rice in dry areas)"
                "</div>",
                unsafe_allow_html=True)
        elif final_rain=="Excess":
            st.markdown(
                "<div class=\"action-box\">"
                "<b>🌊 Act Now — Flood Risk</b><br><br>"
                "✅ Build drainage channels<br><br>"
                "✅ Avoid low-lying fields<br><br>"
                "✅ Stock fungicide for crops<br><br>"
                "✅ Store seeds in dry place"
                "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class=\"action-box\">"
                "<b>✅ Normal Season</b><br><br>"
                "✅ Normal crop planning applies<br><br>"
                "✅ Follow standard sowing calendar<br><br>"
                "✅ Monitor weekly forecasts"
                "</div>",
                unsafe_allow_html=True)

    with c2:
        st.markdown("#### 💧 Water Managers")
        if final_rain=="Deficit":
            st.markdown(
                f"<div class=\"action-box\">"
                f"<b>🚨 Start Conservation Now</b><br><br>"
                f"✅ Begin reservoir conservation<br><br>"
                f"✅ Restrict non-essential water usage<br><br>"
                f"✅ Plan rationing for summer<br><br>"
                f"✅ Fix all leakages immediately<br><br>"
                f"⏰ Plan until: {est_end}"
                f"</div>",
                unsafe_allow_html=True)
        elif final_rain=="Excess":
            st.markdown(
                "<div class=\"action-box\">"
                "<b>🌊 High Water Incoming</b><br><br>"
                "✅ Check dam overflow systems<br><br>"
                "✅ Pre-position flood pumps<br><br>"
                "✅ Alert downstream communities"
                "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class=\"action-box\">"
                "<b>✅ Normal Management</b><br><br>"
                "✅ Standard reservoir operation<br><br>"
                "✅ Normal distribution schedule<br><br>"
                "✅ Routine maintenance now"
                "</div>",
                unsafe_allow_html=True)

    with c3:
        st.markdown("#### 🚨 Disaster Management")
        if heat_cls=="High":
            st.markdown(
                "<div class=\"action-box\">"
                "<b>🔥 High Heat — Act Now</b><br><br>"
                "✅ Open cooling centers<br><br>"
                "✅ Stock ORS and medicines<br><br>"
                "✅ Issue public heat advisory<br><br>"
                "✅ Restrict outdoor work 12pm-4pm<br><br>"
                "✅ Check on elderly daily"
                "</div>",
                unsafe_allow_html=True)
        elif heat_cls=="Moderate":
            st.markdown(
                "<div class=\"action-box\">"
                "<b>🟠 Moderate Heat — Stay Ready</b><br><br>"
                "✅ Keep teams on standby<br><br>"
                "✅ Issue early heat advisory<br><br>"
                "✅ Ensure water supply<br><br>"
                "✅ Monitor vulnerable groups"
                "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class=\"action-box\">"
                "<b>✅ Standard Preparedness</b><br><br>"
                "✅ Normal emergency planning<br><br>"
                "✅ Routine monitoring"
                "</div>",
                unsafe_allow_html=True)
        if final_rain=="Deficit":
            st.markdown(
                f"<div style=\"background:#FFE0E0;border-left:"
                f"4px solid red;padding:10px;border-radius:5px;"
                f"margin-top:8px\">"
                f"🆘 Pre-position drought relief<br>"
                f"📅 Plan resources until: {est_end}"
                f"</div>",
                unsafe_allow_html=True)

    # Early warning
    if abs(live_oni) >= 1.0:
        st.markdown("---")
        st.error(
            f"🚨 EXTREME EVENT WARNING | "
            f"ONI = {round(live_oni,2)} — STRONG {phase_name} | "
            f"Started: {start_season} | "
            f"Expected end: {est_end} | "
            f"Similar to 1997 and 2015 — act immediately for {selected}")

    # Expandable sections
    st.markdown("---")
    with st.expander(
            "🔍 Why is this predicted? (Click to see explanation)"):
        if live_oni >= 0.5:
            st.markdown(
                f"**El Nino** (ONI={round(live_oni,2)}) started {start_season} "
                f"and has been active for {months_active} months. "
                f"The Pacific Ocean is warmer than normal, which weakens "
                f"India's monsoon circulation. "
                f"**{selected}** has a historical El Nino sensitivity of "
                f"{round(sensitivity,3)} — meaning in past El Nino years, "
                f"this region has received "
                f"{'less' if sensitivity<0 else 'more'} than normal rainfall. "
                f"Our model predicts **{final_rain}** rainfall with "
                f"{max(rain_prob)*100:.0f}% confidence. "
                f"Similar past events: 1972 (ONI=0.81), 1997 (ONI=1.14), "
                f"2002 (ONI=0.51), 2015 (ONI=1.48).")
        else:
            st.markdown(
                f"Currently **{phase_name}** (ONI={round(live_oni,2)}). "
                f"No strong ocean disruption active. "
                f"Rainfall predicted as **{final_rain}** for {selected} "
                f"with {max(rain_prob)*100:.0f}% confidence.")

    with st.expander(
            "📊 Historical rainfall chart for "+selected+" (Click to expand)"):
        sh = state_df[state_df["SUBDIVISION"]==selected].copy()
        fig2,ax2 = plt.subplots(figsize=(14,5))
        fig2.patch.set_facecolor("#F8F9FA")
        ax2.set_facecolor("#F8F9FA")
        cm = {"El_Nino":"#FF6B6B","La_Nina":"#4ECDC4","Neutral":"#95A5A6"}
        bc = [cm.get(p,"#95A5A6") for p in sh["ENSO_Phase"]]
        ax2.bar(sh["YEAR"],sh["ANNUAL"],color=bc,alpha=0.85,
                edgecolor="white",linewidth=0.5)
        ax2.axhline(state_lpa,color="#2C3E50",linestyle="--",linewidth=2,
                    label=f"Average ({round(state_lpa,0):.0f} mm)")
        ax2.set_title(
            f"{selected} — Annual Rainfall 1950-2015
"
            f"Red = El Nino years | Blue = La Nina years | Gray = Normal",
            fontweight="bold",color="#2C3E50")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Rainfall (mm)")
        legend_e = [
            mpatches.Patch(color="#FF6B6B",label="El Nino (less rain)"),
            mpatches.Patch(color="#4ECDC4",label="La Nina (more rain)"),
            mpatches.Patch(color="#95A5A6",label="Normal year"),
        ]
        handles,_ = ax2.get_legend_handles_labels()
        ax2.legend(handles=legend_e+handles,fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)

        sh_el = sh[sh["ENSO_Phase"]=="El_Nino"][
            ["YEAR","ANNUAL","IOD","Rainfall_Category","Heatwave_Risk"]
        ].copy()
        sh_el["vs Average"] = (
            (sh_el["ANNUAL"]-state_lpa)/state_lpa*100
        ).round(1).astype(str)+"%"
        sh_el.columns = ["Year","Rainfall (mm)","IOD",
                         "Category","Heatwave","vs Average"]
        st.dataframe(sh_el.set_index("Year"),use_container_width=True)

with st.expander("🗺️ All India Risk — All 36 States"):
    disp = state_risk[["State","Base_Pred","Heatwave","Risk_Level"]].copy()
    disp.columns = ["State","Rainfall Prediction","Heatwave Risk","Risk Level"]
    st.dataframe(disp,use_container_width=True,height=300)
    fig4,ax4 = plt.subplots(figsize=(12,8))
    fig4.patch.set_facecolor("#F8F9FA")
    ax4.set_facecolor("#F8F9FA")
    sr  = state_risk.sort_values("Sensitivity")
    bc4 = ["#FF4444" if x<-0.3 else "#FF8C00" if x<-0.1 else "#28A745"
            for x in sr["Sensitivity"]]
    ax4.barh(sr["State"],sr["Sensitivity"],color=bc4,alpha=0.85)
    ax4.axvline(0,color="black",linewidth=1.5)
    ax4.set_title(
        "El Nino Sensitivity — All Indian States
"
        "States on the LEFT are most affected by El Nino",
        fontweight="bold",color="#2C3E50")
    ax4.set_xlabel("← More affected by El Nino | Less affected →")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    rp = mpatches.Patch(color="#FF4444",label="High Risk")
    op = mpatches.Patch(color="#FF8C00",label="Moderate Risk")
    gp = mpatches.Patch(color="#28A745",label="Low Risk")
    ax4.legend(handles=[rp,op,gp])
    plt.tight_layout()
    st.pyplot(fig4)
