import json
import warnings
from io import StringIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="India Monsoon Risk", page_icon="🌧️", layout="wide")

st.markdown("""
<style>
.main-title {font-size:2.2rem; font-weight:bold; color:#1B2A4A; text-align:center; padding:10px;}
.sub-title {font-size:1rem; color:#666; text-align:center; margin-bottom:15px;}
.status-card {background:linear-gradient(135deg,#1B2A4A,#2E4A7A); color:white; padding:18px;
              border-radius:12px; text-align:center; margin:4px;}
.risk-high {background:#FF4444; color:white; padding:14px; border-radius:10px; font-size:1.1rem;
            font-weight:bold; text-align:center; margin:4px;}
.risk-moderate {background:#FF8C00; color:white; padding:14px; border-radius:10px; font-size:1.1rem;
                font-weight:bold; text-align:center; margin:4px;}
.risk-low {background:#28A745; color:white; padding:14px; border-radius:10px; font-size:1.1rem;
           font-weight:bold; text-align:center; margin:4px;}
.action-box {background:#FFF3CD; border-left:5px solid #FFC107; padding:15px; border-radius:8px; margin:8px 0; color:#222;}
.info-box {background:#E8F4FD; border-left:5px solid #2196F3; padding:15px; border-radius:8px; margin:8px 0; color:#222;}
.timeline-box {background:#F0F4FF; border:2px solid #1B2A4A; padding:18px; border-radius:12px; margin:10px 0; color:#222;}
</style>
""", unsafe_allow_html=True)

COL = {"Red": "#e74c3c", "Orange": "#f39c12", "Green": "#27ae60"}
RANK = {"Green": 0, "Orange": 1, "Red": 2}
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
IOD_SLOPE, IOD_INTERCEPT = 0.901, 0.131   # maps NOAA PSL DMI to the units of master_dataset.csv (from Colab Cell 4)


# ════════════════════════════════════════════════════════
# DATA LOADING (all files are produced by the Colab notebook)
# ════════════════════════════════════════════════════════
@st.cache_data
def load_outputs():
    return dict(
        master=pd.read_csv("master_dataset.csv"),
        sa=pd.read_csv("state_annual_extended.csv"),
        fm=pd.read_csv("forecast_state_monthly.csv"),
        al=pd.read_csv("alerts.csv"),
        fo=pd.read_csv("oni_forecast.csv", parse_dates=["DATE"]),
        nxt=json.load(open("next_elnino.json")),
    )


O = load_outputs()
master_df, sa, fm, al, fo = O["master"], O["sa"], O["fm"], O["al"], O["fo"]
nxt = O["nxt"]


SEAS_TXT = {"DJF": "Dec-Feb", "JFM": "Jan-Mar", "FMA": "Feb-Apr", "MAM": "Mar-May", "AMJ": "Apr-Jun",
            "MJJ": "May-Jul", "JJA": "Jun-Aug", "JAS": "Jul-Sep", "ASO": "Aug-Oct", "SON": "Sep-Nov",
            "OND": "Oct-Dec", "NDJ": "Nov-Jan"}
SEAS_MID = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9,
            "SON": 10, "OND": 11, "NDJ": 12}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def season_text(code, yr):
    return f"{SEAS_TXT.get(code, code)} {yr}"


def season_short(code, yr):
    return f"{MONTHS[SEAS_MID[code] - 1]} {str(yr)[2:]}"


@st.cache_data(ttl=86400)
def get_live():
    """Live ONI from NOAA CPC. Returns a dict; ok=False if the download fails."""
    try:
        raw = requests.get(ONI_URL, timeout=15).text
        df = pd.read_csv(StringIO(raw), sep=r"\s+")          # single backslash: split on whitespace
        df = df.dropna(subset=["ANOM"]).reset_index(drop=True)
        vals = df["ANOM"].astype(float).values
        last = float(vals[-1])
        sign = 1 if last >= 0.5 else (-1 if last <= -0.5 else 0)
        n, start = 0, None
        if sign != 0:                                         # count back through the unbroken run
            i = len(vals) - 1
            while i >= 0 and sign * vals[i] >= 0.5:
                n += 1
                i -= 1
            s = len(vals) - n
            start = season_text(str(df["SEAS"].iloc[s]), int(df["YR"].iloc[s]))
        return dict(ok=True, oni=last, season=str(df["SEAS"].iloc[-1]), year=int(df["YR"].iloc[-1]),
                    recent=df.tail(12).reset_index(drop=True), start=start, n=n)
    except Exception:
        return dict(ok=False)


@st.cache_data(ttl=86400)
def get_live_iod():
    """Latest monthly DMI from NOAA PSL, mapped to the project's IOD units."""
    try:
        raw = requests.get(DMI_URL, timeout=15).text
        last = None
        for line in raw.splitlines():
            p = line.split()
            if len(p) == 13 and p[0].isdigit() and 1870 <= int(p[0]) <= 2100:
                for m, x in enumerate(p[1:], start=1):
                    if float(x) > -90:                         # -9999 / -99.99 are missing codes
                        last = (int(p[0]), m, float(x))
        if last is None:
            return None
        return round(last[2] * IOD_SLOPE + IOD_INTERCEPT, 2), f"{last[0]}-{last[1]:02d}"
    except Exception:
        return None


live = get_live()
if live["ok"]:
    live_oni, live_year = live["oni"], live["year"]
    live_season = season_text(live["season"], live["year"])
    recent_oni, start_season, months_active = live["recent"], live["start"], live["n"]
    live_note = "Live from NOAA CPC"
else:                                                         # fall back to the last saved snapshot, and say so
    live_oni, live_season = float(nxt["last_observed_oni"]), None
    live_year = int(str(nxt["last_observed_date"])[:4])
    live_season = pd.to_datetime(nxt["last_observed_date"]).strftime("%B %Y")
    recent_oni, start_season, months_active = None, None, 0
    live_note = f"NOAA unreachable, showing saved snapshot ({nxt['last_observed_date']})"

iod_live = get_live_iod()
live_iod = iod_live[0] if iod_live else 0.2
iod_note = f"as of {iod_live[1]}" if iod_live else "approximate"

# ENSO phase text
if live_oni >= 1.0:
    phase_name, phase_emoji = "STRONG El Nino", "🔴"
    phase_plain = "The Pacific Ocean is MUCH warmer than normal. This tends to suppress India's monsoon."
elif live_oni >= 0.5:
    phase_name, phase_emoji = "El Nino Active", "🟠"
    phase_plain = "The Pacific Ocean is warmer than normal. India's monsoon is at risk of being weaker."
elif live_oni <= -1.0:
    phase_name, phase_emoji = "STRONG La Nina", "🔵"
    phase_plain = "The Pacific Ocean is MUCH cooler than normal. India's monsoon tends to be wetter."
elif live_oni <= -0.5:
    phase_name, phase_emoji = "La Nina Active", "🔵"
    phase_plain = "The Pacific Ocean is cooler than normal. India's monsoon tends to be good."
else:
    phase_name, phase_emoji = "Normal Conditions", "🟢"
    phase_plain = "Ocean temperatures are near normal. No major ENSO disruption expected."

months_text = f"About {months_active + 2} months" if months_active > 0 else "Not active"

# Start / end of the event
if start_season is None:
    start_season = "Not applicable (no active event)" if abs(live_oni) < 0.5 else "Unavailable (offline)"
if abs(live_oni) < 0.5:
    est_end = "No active event"
elif live_oni >= 0.5 and nxt.get("currently_el_nino") and nxt.get("current_event_expected_end"):
    est_end = (f"Around {pd.to_datetime(nxt['current_event_expected_end']).strftime('%B %Y')} "
               "(forecast, low confidence)")
else:
    est_end = "Not available in the latest forecast"

# State list and helpers
PRETTY = {"MATATHWADA": "Marathwada"}                         # the Kaggle CSV misspells Marathwada


def pretty(s):
    if s == "":
        return "-- Select Your State --"
    return PRETTY.get(s, s.title())


# State sensitivity (correlation of annual ONI with state annual rainfall) and long-period average, 1950-2024
sad = sa.merge(master_df[["YEAR", "ONI", "ENSO_Phase"]], on="YEAR", how="left")
sens = {s: d["ONI"].corr(d["ANNUAL"]) for s, d in sad.groupby("SUBDIVISION")}
lpa = sa[sa.SOURCE == "IMD"].groupby("SUBDIVISION")["ANNUAL"].mean().to_dict()


_mths = sorted(fm.MONTH.unique())
_mname = lambda m: pd.to_datetime(m + "-01").strftime("%b %Y")
WIN_ALL = f"{_mname(_mths[0])} to {_mname(_mths[-1])}"
WIN_3 = f"{_mname(_mths[0])} to {_mname(_mths[min(2, len(_mths) - 1)])}"
WIN_6 = f"{_mname(_mths[0])} to {_mname(_mths[min(5, len(_mths) - 1)])}"
OUTLOOK_END = _mname(_mths[-1])


def state_signals(sub):
    """Monsoon-2027 rainfall signal and hot-season (Mar-Jun 2027) heat risk for one subdivision."""
    g = fm[fm.SUBDIVISION == sub]
    mon = g[g.MONTH.isin(["2027-06", "2027-07", "2027-08"]) & (g.Rain_Confidence != "Low")]
    rain = mon.Rainfall_Pred.value_counts().index[0] if len(mon) else "Normal"
    has_signal = bool(len(mon)) and rain != "Normal"
    hrank = {"Low": 0, "Moderate": 1, "High": 2}
    hot = g[g.MONTH.isin(["2027-03", "2027-04", "2027-05", "2027-06"])]
    heat = max(hot.Heat_Risk, key=lambda x: hrank[x]) if len(hot) else "Low"
    return rain, has_signal, heat


# ════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🌧️ India Monsoon & Heatwave Risk Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

_age = (pd.Timestamp.today() - pd.to_datetime(nxt["last_observed_date"])).days
if _age > 75:
    st.warning(f"The forecasts below were built from ocean data up to {nxt['last_observed_date']} "
               f"({_age} days ago) and may be out of date. The ocean status card is still live.")

# ════════════════════════════════════════════════════════
# OCEAN STATUS
# ════════════════════════════════════════════════════════
st.markdown("### 🌊 What Is the Ocean Doing Right Now?")
col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown(
        f'<div class="status-card"><div style="font-size:3rem">{phase_emoji}</div>'
        f'<div style="font-size:1.6rem;font-weight:bold;margin:5px">{phase_name}</div>'
        f'<div style="font-size:1rem;opacity:0.9">{phase_plain}</div>'
        f'<div style="margin-top:12px;font-size:0.85rem;opacity:0.75">'
        f'Pacific (ONI) = {live_oni:.2f} | Indian Ocean (IOD) = {live_iod:.2f} ({iod_note})<br>{live_note}'
        f'</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown("#### ⏱️ Duration & Timeline")
    st.markdown(
        f'<div class="timeline-box"><table width="100%">'
        f'<tr><td style="padding:8px"><span style="font-size:0.85rem;color:#666">📅 WHEN DID IT START?</span><br>'
        f'<b style="color:#1B2A4A">{start_season}</b></td></tr>'
        f'<tr><td style="padding:8px"><span style="font-size:0.85rem;color:#666">⏳ HOW LONG ACTIVE?</span><br>'
        f'<b style="color:#1B2A4A">{months_text}</b></td></tr>'
        f'<tr><td style="padding:8px"><span style="font-size:0.85rem;color:#666">🔚 WHEN WILL IT END?</span><br>'
        f'<b style="color:#CC0000">{est_end}</b></td></tr></table></div>', unsafe_allow_html=True)

# ONI: last 12 observed seasons + 12-month forecast
with st.expander("📈 See the ocean temperature chart and 12-month forecast"):
    fig_t, ax_t = plt.subplots(figsize=(12, 3.4))
    fig_t.patch.set_facecolor("#F8F9FA")
    ax_t.set_facecolor("#F8F9FA")
    labels, vals, cols_, alphas = [], [], [], []
    if recent_oni is not None:
        for _, r in recent_oni.iterrows():
            v = float(r["ANOM"])
            labels.append(season_short(str(r["SEAS"]), int(r["YR"])))
            vals.append(v)
            cols_.append("#FF4444" if v >= 0.5 else "#4444FF" if v <= -0.5 else "#888888")
            alphas.append(0.9)
    for _, r in fo.iterrows():
        v = float(r["ONI_forecast"])
        labels.append(r["DATE"].strftime("%b %y"))
        vals.append(v)
        cols_.append("#FF4444" if v >= 0.5 else "#4444FF" if v <= -0.5 else "#888888")
        alphas.append(0.45)
    for i, (v, c, a) in enumerate(zip(vals, cols_, alphas)):
        ax_t.bar(i, v, color=c, alpha=a, edgecolor="white")
    n_obs = len(labels) - len(fo)
    if n_obs > 0:
        ax_t.axvline(n_obs - 0.5, color="black", linestyle=":", linewidth=1)
        ax_t.text(n_obs - 0.4, ax_t.get_ylim()[1] * 0.9, "forecast →", fontsize=8)
    # Shade the El Nino period that contains the latest observation (start -> forecast end)
    if n_obs > 0 and vals[n_obs - 1] >= 0.5:
        i0 = n_obs - 1
        while i0 > 0 and vals[i0 - 1] >= 0.5:
            i0 -= 1
        i1 = n_obs - 1
        while i1 < len(vals) - 1 and vals[i1 + 1] >= 0.5:
            i1 += 1
        ax_t.axvspan(i0 - 0.5, i1 + 0.5, color="#FF4444", alpha=0.08)
        ax_t.text((i0 + i1) / 2, max(vals) * 1.08, "El Niño period", ha="center", fontsize=10,
                  fontweight="bold", color="#CC0000")
        ax_t.set_ylim(min(vals) * 1.15, max(vals) * 1.22)
    ax_t.axhline(0.5, color="#FF4444", linestyle="--", linewidth=1.2, label="El Nino (+0.5)")
    ax_t.axhline(-0.5, color="#4444FF", linestyle="--", linewidth=1.2, label="La Nina (-0.5)")
    ax_t.axhline(0, color="black", linewidth=0.5)
    ax_t.set_xticks(range(len(labels)))
    ax_t.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_t.set_ylabel("Ocean temperature vs normal (°C)", fontsize=9)
    ax_t.legend(fontsize=8, loc="lower left")
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    st.pyplot(fig_t)
    plt.close(fig_t)

    if nxt.get("next_el_nino_onset"):
        st.info(f"New El Niño forecast to start: {nxt['next_el_nino_onset']}")
    st.caption("Faded bars = forecast. Confidence drops beyond about 6 months.")

# ════════════════════════════════════════════════════════
# STATE SELECTOR
# ════════════════════════════════════════════════════════
st.markdown("### 📍 What Does This Mean For YOUR State?")
states = [""] + sorted(al.SUBDIVISION.unique())
selected = st.selectbox("Choose your state to get a personalized risk report:", states, index=0,
                        format_func=pretty)

if selected == "":
    cnt = al.OUTLOOK_ALERT.value_counts()
    st.markdown('<div class="info-box"><b>👆 Select your state above</b> to get its alerts, month-by-month '
                'outlook and advice.</div>', unsafe_allow_html=True)
    st.markdown(f"**India overview, outlook {WIN_ALL}:**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="risk-high">🔴 {cnt.get("Red", 0)} States<br><small>Red alert</small></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="risk-moderate">🟠 {cnt.get("Orange", 0)} States<br><small>Orange alert</small></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="risk-low">🟢 {cnt.get("Green", 0)} States<br><small>Green</small></div>',
                unsafe_allow_html=True)
else:
    a = al[al.SUBDIVISION == selected].iloc[0]
    sensitivity = float(sens[selected])
    state_lpa = float(lpa[selected])
    final_rain, has_signal, heat_cls = state_signals(selected)

    # Risk banners
    if final_rain == "Deficit":
        r_emoji, r_text, r_class = "⚠️", "BELOW-NORMAL RAINFALL SIGNAL (Jun-Aug 2027)", "risk-high"
        r_plain = "The ENSO/IOD signal for this state leans dry. Low confidence."
    elif final_rain == "Excess":
        r_emoji, r_text, r_class = "🌊", "ABOVE-NORMAL RAINFALL SIGNAL (Jun-Aug 2027)", "risk-moderate"
        r_plain = "The ENSO/IOD signal for this state leans wet. Low confidence."
    else:
        r_emoji, r_text, r_class = "✅", "NO CLEAR RAINFALL SIGNAL (Jun-Aug 2027)", "risk-low"
        r_plain = "No reliable ocean-based signal for this state. Follow normal planning."
    if heat_cls == "High":
        h_emoji, h_text, h_class = "🔥", "HIGH HEATWAVE RISK (Mar-Jun 2027)", "risk-high"
        h_plain = "Warm ENSO/IOD pattern in the hot season."
    elif heat_cls == "Moderate":
        h_emoji, h_text, h_class = "🌡️", "MODERATE HEAT RISK (Mar-Jun 2027)", "risk-moderate"
        h_plain = "Somewhat higher heat risk. Stay hydrated."
    else:
        h_emoji, h_text, h_class = "😊", "LOW HEAT RISK (Mar-Jun 2027)", "risk-low"
        h_plain = "ENSO/IOD pattern does not favour extra heat. Local weather can still bring heatwaves."

    st.markdown(f"## 📍 {pretty(selected)}: Your Risk Report")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="{r_class}">{r_emoji} {r_text}<br><span style="font-size:0.9rem;'
                f'font-weight:normal">{r_plain}</span></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="{h_class}">{h_emoji} {h_text}<br><span style="font-size:0.9rem;'
                f'font-weight:normal">{h_plain}</span></div>', unsafe_allow_html=True)

    # Alert panel
    st.markdown("### 🚦 Alerts")
    cols3 = st.columns(3)
    for c, lab, key in [(cols3[0], f"Next 3 months ({WIN_3})", "NEXT3_ALERT"),
                        (cols3[1], f"Next 6 months ({WIN_6})", "NEXT6_ALERT"),
                        (cols3[2], f"Outlook to {OUTLOOK_END}", "OUTLOOK_ALERT")]:
        c.markdown(f"<div style='background:{COL[a[key]]};color:white;padding:14px;border-radius:10px;"
                   f"text-align:center'><b>{lab}</b><br>{a[key]}</div>", unsafe_allow_html=True)
    if isinstance(a.FIRST_RED_MONTH, str) and a.FIRST_RED_MONTH:
        st.error(f"First Red alert month: {a.FIRST_RED_MONTH}")
    if a.OUTLOOK_ALERT != "Green":
        st.write(f"Main driver: {a.MAIN_DRIVER}")

    # Month-by-month
    st.markdown(f"### 📅 Month-by-month forecast ({WIN_ALL})")
    g = fm[fm.SUBDIVISION == selected][["MONTH", "Rainfall_Pred", "Rain_Confidence", "Heat_Risk", "Alert"]]
    st.dataframe(g.style.apply(
        lambda r: [f"background-color:{COL[r.Alert]};color:white" if c == "Alert" else "" for c in r.index],
        axis=1), hide_index=True)
    st.caption("Months beyond about 6 are outlook only.")

    # Timeline
    st.markdown(f"### ⏱️ How Long Will This Affect {pretty(selected)}?")
    st.markdown(
        f'<div class="timeline-box"><table width="100%"><tr>'
        f'<td width="33%" style="text-align:center;padding:10px"><div style="font-size:1.5rem">🌊</div>'
        f'<div style="font-weight:bold;color:#1B2A4A">{phase_name} Started</div>'
        f'<div style="color:#555;font-size:0.95rem">{start_season}</div></td>'
        f'<td width="33%" style="text-align:center;padding:10px;border-left:2px dashed #CBD5E0;'
        f'border-right:2px dashed #CBD5E0"><div style="font-size:1.5rem">📍</div>'
        f'<div style="font-weight:bold;color:#1B2A4A">Right Now</div>'
        f'<div style="color:#555;font-size:0.95rem">{live_season} | ONI = {live_oni:.2f}</div></td>'
        f'<td width="33%" style="text-align:center;padding:10px"><div style="font-size:1.5rem">🔚</div>'
        f'<div style="font-weight:bold;color:#CC0000">Expected End</div>'
        f'<div style="color:#CC0000;font-size:0.9rem">{est_end}</div></td></tr></table></div>',
        unsafe_allow_html=True)

    # Actions
    st.markdown("---")
    st.markdown("### ⚡ What Should You Do?")
    st.markdown(f"*Actions for {pretty(selected)} based on the outlook above. These are precautions "
                f"under uncertainty, not certain predictions.*")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 👨‍🌾 Farmers")
        if final_rain == "Deficit":
            body = ("<b>🚨 Prepare for a dry monsoon</b><br><br>✅ Plan drought-resistant crops "
                    "(bajra, jowar, pulses)<br><br>✅ Be ready to delay sowing<br><br>"
                    "✅ Apply for crop insurance<br><br>❌ Avoid water-heavy crops where irrigation is weak")
        elif final_rain == "Excess":
            body = ("<b>🌊 Prepare for a wet monsoon</b><br><br>✅ Build drainage channels<br><br>"
                    "✅ Avoid low-lying fields<br><br>✅ Stock fungicide<br><br>✅ Store seed in a dry place")
        else:
            body = ("<b>✅ No special signal</b><br><br>✅ Normal crop planning<br><br>"
                    "✅ Follow the standard sowing calendar<br><br>✅ Check IMD forecasts before sowing")
        st.markdown(f'<div class="action-box">{body}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("#### 💧 Water Managers")
        if final_rain == "Deficit":
            body = ("<b>🚨 Start conservation planning</b><br><br>✅ Conserve reservoir storage<br><br>"
                    "✅ Restrict non-essential use<br><br>✅ Plan summer rationing<br><br>"
                    f"⏰ Plan until: {est_end}")
        elif final_rain == "Excess":
            body = ("<b>🌊 Prepare for high inflows</b><br><br>✅ Check dam overflow systems<br><br>"
                    "✅ Pre-position flood pumps<br><br>✅ Alert downstream communities")
        else:
            body = ("<b>✅ Normal management</b><br><br>✅ Standard reservoir operation<br><br>"
                    "✅ Routine maintenance now")
        st.markdown(f'<div class="action-box">{body}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown("#### 🚨 Disaster Management")
        if heat_cls == "High":
            body = ("<b>🔥 High heat risk: prepare early</b><br><br>✅ Plan cooling centers<br><br>"
                    "✅ Stock ORS and medicines<br><br>✅ Prepare public heat advisories<br><br>"
                    "✅ Restrict outdoor work 12pm-4pm in season")
        elif heat_cls == "Moderate":
            body = ("<b>🟠 Moderate heat risk: stay ready</b><br><br>✅ Keep teams on standby<br><br>"
                    "✅ Prepare an early advisory<br><br>✅ Ensure water supply")
        else:
            body = ("<b>✅ Standard preparedness</b><br><br>✅ Normal emergency planning<br><br>"
                    "✅ Follow IMD heatwave warnings")
        st.markdown(f'<div class="action-box">{body}</div>', unsafe_allow_html=True)
        if final_rain == "Deficit":
            st.markdown("<div style='background:#FFE0E0;border-left:4px solid red;padding:10px;"
                        "border-radius:5px;margin-top:8px;color:#222'>🆘 Pre-position drought relief</div>",
                        unsafe_allow_html=True)

    if abs(live_oni) >= 1.0:
        st.markdown("---")
        st.error(f"🚨 {phase_name} | ONI = {live_oni:.2f} | Started: {start_season} | "
                 f"Expected end: {est_end}")

    st.markdown("---")
    with st.expander("🔍 Why is this predicted? (Click to see explanation)"):
        st.markdown(
            f"The Pacific is in **{phase_name}** (ONI = {live_oni:.2f}). Historically (1950-2024), "
            f"**{pretty(selected)}** has an El Niño sensitivity of **{sensitivity:.2f}**: in warm-ONI years "
            f"it received {'less' if sensitivity < 0 else 'more'} rain than usual. "
            f"The forecast ONI falls through the El Niño threshold around "
            f"{pd.to_datetime(nxt['current_event_expected_end']).strftime('%B %Y') if nxt.get('current_event_expected_end') else 'N/A'} "
            f"and turns La Niña by mid-2027. Month-wise rainfall for each state is estimated from how "
            f"that month's rainfall behaved in past El Niño and La Niña years (1950-2015). Where that link "
            f"is weak the state is marked low confidence and gets no rainfall alert. "
            f"Past years with a similar building El Niño: 1972, 1982, 2015 (Deficit) and 1997, 2023 (Normal).")

    with st.expander(f"📊 Historical rainfall chart for {pretty(selected)}"):
        sh = sad[sad.SUBDIVISION == selected].sort_values("YEAR")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        fig2.patch.set_facecolor("#F8F9FA")
        ax2.set_facecolor("#F8F9FA")
        cm = {"El_Nino": "#FF6B6B", "La_Nina": "#4ECDC4", "Neutral": "#95A5A6"}
        ax2.bar(sh["YEAR"], sh["ANNUAL"], color=[cm.get(p, "#95A5A6") for p in sh["ENSO_Phase"]],
                alpha=0.85, edgecolor="white", linewidth=0.5,
                hatch=None)
        era = sh[sh.SOURCE != "IMD"]
        ax2.bar(era["YEAR"], era["ANNUAL"], color="none", edgecolor="black", hatch="//", linewidth=0.6)
        ax2.axhline(state_lpa, color="#2C3E50", linestyle="--", linewidth=2,
                    label=f"IMD average ({state_lpa:.0f} mm)")
        ax2.set_title(f"{pretty(selected)}: Annual Rainfall 1950-2024\n"
                      "Red = El Nino | Teal = La Nina | Gray = Neutral | Hatched = ERA5-derived (2016-2024)",
                      fontweight="bold", color="#2C3E50")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Rainfall (mm)")
        legend_e = [mpatches.Patch(color="#FF6B6B", label="El Nino (less rain)"),
                    mpatches.Patch(color="#4ECDC4", label="La Nina (more rain)"),
                    mpatches.Patch(color="#95A5A6", label="Neutral"),
                    mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="ERA5-derived")]
        handles, _ = ax2.get_legend_handles_labels()
        ax2.legend(handles=legend_e + handles, fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)
        plt.close(fig2)
        sh_el = sh[sh["ENSO_Phase"] == "El_Nino"][["YEAR", "ANNUAL", "SOURCE"]].copy()
        sh_el["vs Average"] = ((sh_el["ANNUAL"] - state_lpa) / state_lpa * 100).round(1).astype(str) + "%"
        sh_el["ANNUAL"] = sh_el["ANNUAL"].round(0)
        sh_el.columns = ["Year", "Rainfall (mm)", "Source", "vs Average"]
        st.dataframe(sh_el.set_index("Year"))

# ════════════════════════════════════════════════════════
# ALL-INDIA TABLE, VALIDATION
# ════════════════════════════════════════════════════════
with st.expander("🗺️ All India Alerts: All 36 Subdivisions"):
    disp = al[["SUBDIVISION", "NEXT3_ALERT", "NEXT6_ALERT", "OUTLOOK_ALERT", "FIRST_RED_MONTH"]].copy()
    disp["SUBDIVISION"] = disp["SUBDIVISION"].map(pretty)
    disp.columns = ["State", "Next 3 months", "Next 6 months", f"Outlook to {OUTLOOK_END}", "First Red month"]
    st.dataframe(disp.style.apply(
        lambda r: [f"background-color:{COL[v]};color:white" if v in COL else "" for v in r], axis=1),
        hide_index=True, height=300)
    sr = pd.Series(sens).sort_values()
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    fig4.patch.set_facecolor("#F8F9FA")
    ax4.set_facecolor("#F8F9FA")
    ax4.barh([pretty(s) for s in sr.index], sr.values,
             color=["#FF4444" if x < -0.3 else "#FF8C00" if x < -0.1 else "#28A745" for x in sr.values],
             alpha=0.85)
    ax4.axvline(0, color="black", linewidth=1.5)
    ax4.set_title("El Nino Sensitivity (1950-2024): states on the LEFT lose most rain in El Nino years",
                  fontweight="bold", color="#2C3E50")
    ax4.set_xlabel("Correlation of annual ONI with annual rainfall")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.legend(handles=[mpatches.Patch(color="#FF4444", label="Strongly affected"),
                        mpatches.Patch(color="#FF8C00", label="Moderately affected"),
                        mpatches.Patch(color="#28A745", label="Weak / opposite")])
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
