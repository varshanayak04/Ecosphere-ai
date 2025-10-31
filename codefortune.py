# ==============================================================
# CodeFortune - AI-Powered Sustainability Platform
# Optimizing Industrial Waste & Resources
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="CodeFortune ESG Dashboard", layout="wide")
st.title("🌍 CodeFortune — Sustainability Intelligence Platform")
st.markdown("### AI-powered Sustainability Reporting, Prediction & Optimization Engine")

# ---------------------- SIDEBAR: UPLOAD DATA ----------------------
st.sidebar.header("📂 Upload Sustainability Data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ---------------------- LOAD DATA ----------------------
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    st.success("✅ Data uploaded successfully!")
else:
    st.info("No file uploaded — using demo industrial dataset.")
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=12, freq="M"),
        "Electricity_kWh": np.random.randint(1000, 8000, 12),
        "Diesel_Litres": np.random.randint(200, 1000, 12),
        "Water_Usage_kl": np.random.randint(50, 200, 12),
        "Waste_Generated_kg": np.random.randint(100, 500, 12)
    })

st.subheader("📊 Uploaded / Demo Data")
st.dataframe(df.head())

# ---------------------- STEP 1: CLEANING ----------------------
st.subheader("🧹 Data Cleaning & Preprocessing")
st.write(f"Initial Shape: {df.shape}")

df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))
st.write(f"After Cleaning: {df.shape}")
st.success("Missing values handled and duplicates removed.")

# ---------------------- STEP 2: EMISSION FACTORS ----------------------
st.subheader(♻️ Emission Factor Calculation")

# Standard factors (kgCO2e per unit)
emission_factors = {
    "electricity": 0.82,        # per kWh
    "diesel": 2.68,             # per litre
    "petrol": 2.31,
    "coal": 2.42,
    "lpg": 1.51,
    "natural_gas": 2.03,
    "water": 0.35,              # per kilolitre (energy for treatment)
    "waste": 1.8                # per kg (avg decomposition + transport)
}

st.write("**Emission Factors used (kgCO₂e per unit):**")
st.dataframe(pd.DataFrame(list(emission_factors.items()), columns=["Parameter", "Factor"]))

# Calculate emissions
df["Emission (kgCO2e)"] = 0.0
for col in df.columns:
    for name, factor in emission_factors.items():
        if name in col.lower():
            df["Emission (kgCO2e)"] += pd.to_numeric(df[col], errors="coerce").fillna(0) * factor

st.success("✅ Emission Calculations Complete")
st.dataframe(df.head())

# ---------------------- STEP 3: AI FORECASTING & OPTIMIZATION ----------------------
st.subheader("🤖 AI Forecasting & Optimization")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if "Emission (kgCO2e)" in numeric_cols and len(numeric_cols) > 1:
    X = df[[c for c in numeric_cols if c != "Emission (kgCO2e)"]]
    y = df["Emission (kgCO2e)"]

    if len(df) >= 4:
        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Model evaluation
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"**Model Performance:**  R² = {r2:.2f}   |   MAE = {mae:.2f} kgCO2e")

        # ------------------- USER INPUTS FOR FORECAST -------------------
        st.markdown("#### 🔧 Enter next-period operational values to predict future emissions:")
        input_vals = []
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            input_vals.append(val)

        # ------------------- PREDICTION & RECOMMENDATIONS -------------------
        if st.button("🔮 Predict & Optimize"):
            base_pred = model.predict(np.array([input_vals]))[0]
            st.success(f"🌱 **Predicted Future Emission:** {base_pred:,.2f} kgCO2e")

            st.markdown("---")
            st.markdown("### ♻️ Optimization Insights (Reduce Footprint)")

            # Test each variable with a 10 % reduction to estimate impact
            impacts = []
            for i, col in enumerate(X.columns):
                reduced = input_vals.copy()
                reduced[i] = reduced[i] * 0.9  # 10 % reduction
                new_pred = model.predict(np.array([reduced]))[0]
                savings = base_pred - new_pred
                if savings > 0:
                    impacts.append((col, savings))

            if impacts:
                for col, savings in sorted(impacts, key=lambda x: x[1], reverse=True):
                    st.write(
                        f"🔹 By reducing **{col}** by 10 %, "
                        f"you can lower emissions by approximately **{savings:,.2f} kgCO2e.**"
                    )
            else:
                st.info("Model did not detect any variable reductions that meaningfully lower emissions.")

            st.caption("Suggestions generated using sensitivity testing on the trained Random Forest model.")
    else:
        st.warning("Dataset too small for forecasting. Need at least 4 rows.")
else:
    st.warning("Please include numeric sustainability metrics for forecasting.")

# ---------------------- STEP 4: ADVANCED VISUALIZATION DASHBOARD ----------------------
st.subheader("📊 ESG Visualization Dashboard")

if "Emission (kgCO2e)" not in df.columns:
    st.error("Emission data not found. Please complete emission calculation first.")
else:
    # 🗓️ 1️⃣ MONTHLY EMISSION TREND
    st.markdown("### 📅 Monthly Emission Trend")
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df["Month"] = df[date_cols[0]].dt.to_period("M").astype(str)
        monthly_df = df.groupby("Month")["Emission (kgCO2e)"].sum().reset_index()

        fig_month = px.line(
            monthly_df,
            x="Month",
            y="Emission (kgCO2e)",
            title="Total Emission by Month",
            markers=True,
            color_discrete_sequence=["#2E8B57"]
        )
        fig_month.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Emission (kgCO2e)",
            template="plotly_white"
        )
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.info("No Date column found — skipping monthly trend visualization.")

    # ⚙️ 2️⃣ MONTHLY RESOURCE USAGE OVERVIEW
    st.markdown("### ⚙️ Monthly Resource Usage Overview")

    usage_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ["electric", "diesel", "water", "waste", "fuel", "coal", "gas", "petrol", "lpg"])
    ]

    if date_cols and usage_cols:
        df["Month"] = df[date_cols[0]].dt.to_period("M").astype(str)
        usage_df = df.groupby("Month")[usage_cols].sum().reset_index()
        usage_melted = usage_df.melt(id_vars="Month", var_name="Resource", value_name="Usage")

        fig_usage = px.line(
            usage_melted,
            x="Month",
            y="Usage",
            color="Resource",
            markers=True,
            title="Monthly Resource Usage by Category",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_usage.update_layout(
            xaxis_title="Month",
            yaxis_title="Usage (Units)",
            template="plotly_white"
        )
        st.plotly_chart(fig_usage, use_container_width=True)
    else:
        st.info("No suitable resource usage columns found for monthly usage overview.")

    # 🌱 3️⃣ FORECAST COMPARISON VISUALIZATION
    st.markdown("### 🌱 Forecast vs Current Emission Comparison")

    if "base_pred" in locals():
        last_value = df["Emission (kgCO2e)"].iloc[-1]
        comparison_df = pd.DataFrame({
            "Period": ["Last Recorded", "Predicted Next Period"],
            "Emission (kgCO2e)": [last_value, base_pred]
        })

        fig_forecast = px.bar(
            comparison_df,
            x="Period",
            y="Emission (kgCO2e)",
            text="Emission (kgCO2e)",
            color="Period",
            color_discrete_sequence=["#4682B4", "#3CB371"],
            title="Emission Forecast Comparison"
        )
        fig_forecast.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_forecast.update_layout(
            yaxis_title="Emission (kgCO2e)",
            template="plotly_white"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Run the AI Forecast above to view forecast comparison chart.")

    # 📊 4️⃣ ESG SUMMARY KPIs
    st.markdown("### 📈 ESG Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Total Emission", f"{df['Emission (kgCO2e)'].sum():,.2f} kgCO2e")
    col2.metric("📆 Records Analyzed", len(df))
    col3.metric("⚡ Avg Emission per Record", f"{df['Emission (kgCO2e)'].mean():,.2f} kgCO2e")


# ---------------------- STEP 5: REPORT GENERATION ---------------------- 
st.subheader("📄 Generate ESG Report (PDF)")

if st.button("📘 Generate ESG Report"):
    avg_emission = df["Emission (kgCO2e)"].mean()
    total_emission = df["Emission (kgCO2e)"].sum()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CodeFortune ESG Sustainability Report", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)
    intro_text = (
        "This automatically generated report summarizes sustainability metrics, "
        "AI-based emission forecasts, and actionable insights for industry adoption."
    )
    pdf.multi_cell(0, 8, intro_text.encode("latin-1", "replace").decode("latin-1"))
    pdf.ln(6)

    # Summary stats
    pdf.cell(0, 10, f"Average Emission: {avg_emission:,.2f} kgCO2e", ln=True)
    pdf.cell(0, 10, f"Total Emission: {total_emission:,.2f} kgCO2e", ln=True)
    pdf.ln(6)

    pdf.multi_cell(0, 8, "Emission factors applied per sustainability category:")
    for k, v in emission_factors.items():
        line = f" - {k}: {v} kgCO2e per unit"
        pdf.cell(0, 8, line.encode("latin-1", "replace").decode("latin-1"), ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 11)
    pdf.multi_cell(
        0, 8,
        "Generated automatically by CodeFortune ESG Intelligence System.\n"
        "For internal sustainability analysis and reporting use."
    )

    pdf.output("ESG_Report.pdf")

    with open("ESG_Report.pdf", "rb") as f:
        st.download_button("📥 Download ESG Report", f, "ESG_Report.pdf")

st.caption("End-to-End ESG Intelligence Prototype — built with Streamlit, Pandas, and Scikit-Learn.")
