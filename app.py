import streamlit as st
import pandas as pd
import engine
import os

st.set_page_config(page_title="Banking4all", layout="wide", page_icon="🍊")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    profiles = pd.read_csv("outputs/profiles.csv", index_col="account_id")
    fragility = pd.read_csv("outputs/fragility_table.csv")
    return profiles, fragility

profiles, fragility = load_data()

# --- HEADER & POPULATION FINDINGS ---
st.title("🍊 Tangerine: Banking4all")
st.markdown("### Financial Resilience Engine")

with st.expander("📊 View Population-Level Findings (Discovery)"):
    st.markdown("**Headline Insight:** 35% of customers cannot cover 6 months on EI at 55% income, and over 18% of 'safe-looking' customers hold hidden volatility risks.")
    st.dataframe(fragility, use_container_width=True)

# --- SIDEBAR: JUDGE CONTROLS ---
st.sidebar.header("🎯 Live Twist Controls")
account_id = st.sidebar.selectbox("1. Pick a Customer (account_id)", profiles.index)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Apply a Shock")
shock_type = st.sidebar.selectbox("Select Scenario", ["No Shock"] + list(engine.PRESETS.keys()) + ["Custom Twist"])

if shock_type == "Custom Twist":
    inc_mult = st.sidebar.slider("Income Retained", 0.0, 1.0, 0.55, help="0.55 = EI standard")
    bills_pct = st.sidebar.slider("Bills Increase", 0.0, 1.0, 0.15, help="0.15 = 15% jump in rent/utilities")
    one_off = st.sidebar.number_input("Surprise One-off Cost ($)", min_value=0, value=1500, step=100)
    months = st.sidebar.slider("Shock Duration (Months)", min_value=1, max_value=24, value=6)
    active_shock = engine.Shock(income_mult=inc_mult, bills_pct=bills_pct, one_off=one_off, months=months, name="Custom Twist")
elif shock_type != "No Shock":
    active_shock = engine.PRESETS[shock_type]
else:
    active_shock = None

st.sidebar.markdown("---")
horizon = st.sidebar.slider("3. Chart Timeline (Months)", min_value=3, max_value=36, value=12)

# --- PART 1: KNOW THE CUSTOMER ---
st.markdown("---")
st.subheader(f"👤 Customer Profile: `{account_id}`")
p = engine.get_profile(profiles, account_id)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Typical Income (I)", f"${p['I']:,.2f}")
col2.metric("Typical Spending (S)", f"${p['S']:,.2f}")
col3.metric("Available Buffer (B)", f"${p['B']:,.2f}")
col4.metric("Base Runway", engine.runway_label(p['runway_base']))

# --- PART 2: TAKE A SHOCK & INTERACTIVE CHART ---
if active_shock:
    st.markdown("---")
    st.subheader(f"⚡ Shock Applied: {active_shock.name}")

    r = engine.apply_shock(p, active_shock)

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Income Under Shock", f"${r['income_after']:,.2f}", delta=f"${r['income_after'] - p['I']:,.2f}")
    scol2.metric("Spending Under Shock", f"${r['spending_after']:,.2f}", delta=f"${r['spending_after'] - p['S']:,.2f}", delta_color="inverse")
    scol3.metric("Post-Shock Gap", f"${r['gap']:,.2f}", help="Positive means spending exceeds income")
    scol4.metric("New Runway", engine.runway_label(r['runway_shock']))

    if not r['survives_shock_period']:
        st.error(f"⚠️ **Warning:** Customer runs out of money in month {r['runout_month']}.")
    else:
        st.success("✅ Customer survives the shock period without running out of cash.")

    # --- VISUALIZATION ---
    st.markdown("### 📉 Projected Balance Over Time")
    proj_df = engine.project_balance(p, active_shock, horizon=horizon)
    chart_df = proj_df.rename(columns={
        "balance": "Balance Under Shock",
        "baseline": "Normal Trajectory"
    }).set_index("month")[["Balance Under Shock", "Normal Trajectory"]]
    st.line_chart(chart_df, color=["#FF4B4B", "#29B09D"])

# --- PART 3: MAKE A PLAN ---
st.markdown("---")
st.subheader("🗓️ Action Plans")

pcol1, pcol2 = st.columns(2)

with pcol1:
    st.markdown("#### 🎯 Baseline Emergency Goal")
    goal = engine.goal_plan(p, target_months=3)
    st.write(f"**Target:** Build a 3-month emergency fund (${goal['target_fund']:,.2f})")

    if goal['already_there']:
        st.success(goal['note'])
    else:
        st.warning(f"**Gap to target:** ${goal['gap_to_target']:,.2f}")
        st.write(f"To reach this in 12 months, they need to save **${goal['required_monthly_saving']:,.2f}/month**.")
        st.info(goal['note'])
        if goal.get('trim_suggestions'):
            st.write("**Suggested trims from flexible spending:**")
            st.dataframe(pd.DataFrame(goal['trim_suggestions']), use_container_width=True)

with pcol2:
    st.markdown("#### 🛡️ Backup Plan (Under Shock)")
    if active_shock:
        backup = engine.backup_plan(p, active_shock)
        st.write(f"**Objective:** Survive {backup['duration_months']} months under current shock conditions.")

        if not backup['needs_action']:
            st.success(backup['summary'])
        else:
            st.warning(backup['summary'])
            st.write(f"**Required Monthly Cut:** ${backup['required_cut']:,.2f}")

            st.write("**Available Relief Options:**")
            st.dataframe(pd.DataFrame(backup['options']), use_container_width=True)

            if not backup['covered_by_cuts']:
                st.error(f"Even with all cuts, a bridge of **${backup['bridge_needed_total']:,.2f}** is still required.")
    else:
        st.info("Apply a shock from the sidebar to generate a backup plan.")

# --- PART 4: AI SUMMARY ---
st.markdown("---")
st.subheader("✨ AI Insights")

if st.button("Generate AI Summary of this Situation"):
    import google.generativeai as genai

    # Attempt to load GEMINI_API_KEY from Colab's user_secrets or environment variables
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables, Colab Secrets, or Streamlit secrets.")
    else:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')

            prompt = f"Write a concise, 2-3 sentence summary of the following financial situation for a bank advisor. "
            prompt += f"Typical Income: ${p['I']:,.2f}. Typical Spending: ${p['S']:,.2f}. Available Buffer: ${p['B']:,.2f}. "

            if active_shock:
                prompt += f"They have experienced a shock: {active_shock.name}. "
                prompt += f"Their new income is ${r['income_after']:,.2f} and spending is ${r['spending_after']:,.2f}. "
                prompt += f"Their new runway is {r['runway_label']}. "
                if not r['survives_shock_period']:
                    prompt += "They will run out of money. "
            else:
                prompt += "They are not currently experiencing an economic shock. "

            prompt += "Make it empathetic but professional."

            with st.spinner("Generating summary..."):
                response = model.generate_content(prompt)
                st.success(response.text)

        except Exception as e:
            st.error(f"Error generating summary: {e}")
