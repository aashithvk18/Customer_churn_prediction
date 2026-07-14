import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background: #262730;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.big-font {
    font-size:28px !important;
    font-weight:bold;
    color:#00BFFF;
}

.small-font {
    font-size:16px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------

st.markdown("""
<h1 style='text-align:center;color:#00BFFF'>
📊 Customer Churn Intelligence Dashboard
</h1>

<h4 style='text-align:center;color:white'>
Machine Learning Powered Retention Analytics
</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv(
    r"D:\WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# -------------------------
# CLEANING
# -------------------------

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

df["AvgChargePerMonth"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1)
)

df["RiskLevel"] = np.where(
    df["tenure"] < 12,
    "High Risk",
    np.where(
        df["tenure"] < 24,
        "Medium Risk",
        "Low Risk"
    )
)

# -------------------------
# SAVE ORIGINAL DATA
# -------------------------

original_df = df.copy()

# -------------------------
# ENCODING
# -------------------------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load(
    r"C:\Users\aashi\customer_churn_model.pkl"
)

# ==================================================
# KPI SECTION
# ==================================================

st.subheader("📌 Executive Summary")

total_customers = len(df)

churn_rate = round(
    (df["Churn"].mean()) * 100,
    2
)

avg_monthly = round(
    df["MonthlyCharges"].mean(),
    2
)

avg_tenure = round(
    df["tenure"].mean(),
    2
)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "⚠️ Churn Rate %",
        churn_rate
    )

with col3:
    st.metric(
        "💰 Avg Monthly Charges",
        avg_monthly
    )

with col4:
    st.metric(
        "📅 Avg Tenure",
        avg_tenure
    )

st.markdown("---")

# ==================================================
# CUSTOMER EXPLORER
# ==================================================

st.subheader("🔍 Customer Explorer")

search_id = st.number_input(
    "Enter Customer Row Number",
    min_value=0,
    max_value=len(original_df)-1,
    value=0
)

if st.button("Show Customer"):

    st.dataframe(
        original_df.iloc[[search_id]],
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# CHARTS
# ==================================================

st.subheader("📈 Customer Churn Analytics")

left,right = st.columns(2)

# ------------------
# CHURN PIE
# ------------------

with left:

    churn_counts = original_df["Churn"].value_counts()

    fig = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------
# RISK LEVEL
# ------------------

with right:

    risk_counts = original_df["RiskLevel"].value_counts()

    fig2 = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title="Risk Level Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# MONTHLY CHARGES ANALYSIS
# ==================================================

col1,col2 = st.columns(2)

with col1:

    fig3 = px.histogram(
        original_df,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

with col2:

    fig4 = px.box(
        original_df,
        x="Churn",
        y="MonthlyCharges",
        title="Monthly Charges vs Churn"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# CONTRACT ANALYSIS
# ==================================================

st.subheader("📃 Contract Analysis")

contract_fig = px.histogram(
    original_df,
    x="Contract",
    color="Churn",
    barmode="group"
)

st.plotly_chart(
    contract_fig,
    use_container_width=True
)

st.markdown("---")

# ==================================================
# CORRELATION HEATMAP
# ==================================================

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(14,8)
)

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")

# ==================================================
# MODEL LEADERBOARD
# ==================================================

st.subheader("🤖 Model Performance Leaderboard")

results = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy":[
        0.807665,
        0.801278,
        0.797729
    ]

})

st.dataframe(
    results,
    use_container_width=True
)

leader_fig = px.bar(
    results,
    x="Model",
    y="Accuracy",
    text="Accuracy",
    title="Model Accuracy Comparison"
)

st.plotly_chart(
    leader_fig,
    use_container_width=True
)

st.success(
    "🏆 Best Model: Logistic Regression (80.77%)"
)

st.markdown("---")

# ==================================================
# TOP CHURN DRIVERS
# ==================================================

st.subheader("🔥 Top Churn Drivers")

importance = pd.DataFrame({

    "Feature":[
        "MonthlyCharges",
        "TotalCharges",
        "AvgChargePerMonth",
        "tenure",
        "Contract",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "RiskLevel",
        "gender"
    ],

    "Importance":[
        0.145522,
        0.144414,
        0.136977,
        0.125542,
        0.076012,
        0.044576,
        0.043697,
        0.034464,
        0.030913,
        0.024052
    ]

})

fig_imp = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Churn Drivers"
)

st.plotly_chart(
    fig_imp,
    use_container_width=True
)

st.markdown("---")

# ==================================================
# BUSINESS INSIGHTS
# ==================================================

st.subheader("🧠 AI Business Insights")

st.info(
    f"""
    📌 Average Monthly Charge : ₹{original_df['MonthlyCharges'].mean():.2f}

    📌 Average Tenure : {original_df['tenure'].mean():.2f} Months

    📌 Churn Rate : {(original_df['Churn'].eq('Yes').mean()*100):.2f}%
    """
)

if (original_df["Churn"].eq("Yes").mean()*100) > 25:

    st.error(
        """
        High customer churn detected.

        Recommended Actions:
        • Loyalty Programs
        • Discount Offers
        • Long-term Contract Benefits
        • Customer Retention Campaigns
        """
    )

else:

    st.success(
        "Customer retention is healthy."
    )

st.markdown("---")

# ==================================================
# DOWNLOAD CENTER
# ==================================================

st.subheader("📥 Download Center")

csv = original_df.to_csv(
    index=False
)

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="customer_churn_data.csv",
    mime="text/csv"
)

st.markdown("---")

# ==================================================
# PREDICTION CENTER
# ==================================================

st.subheader("🔮 Prediction Center")

st.write(
    "Enter encoded values used during training."
)

col1,col2,col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        [0,1]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        [0,1]
    )

    dependents = st.selectbox(
        "Dependents",
        [0,1]
    )

    tenure = st.number_input(
        "Tenure",
        0,
        100,
        12
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        [0,1]
    )

    multiple = st.number_input(
        "Multiple Lines",
        0,
        2,
        0
    )

    internet = st.number_input(
        "Internet Service",
        0,
        2,
        0
    )

    security = st.number_input(
        "Online Security",
        0,
        2,
        0
    )

    backup = st.number_input(
        "Online Backup",
        0,
        2,
        0
    )

with col3:

    protection = st.number_input(
        "Device Protection",
        0,
        2,
        0
    )

    support = st.number_input(
        "Tech Support",
        0,
        2,
        0
    )

    tv = st.number_input(
        "Streaming TV",
        0,
        2,
        0
    )

    movies = st.number_input(
        "Streaming Movies",
        0,
        2,
        0
    )

    contract = st.number_input(
        "Contract",
        0,
        2,
        0
    )

paperless = st.selectbox(
    "Paperless Billing",
    [0,1]
)

payment = st.number_input(
    "Payment Method",
    0,
    3,
    0
)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    50.0
)

total = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    500.0
)

avg_charge = total/(tenure+1)

if tenure < 12:
    risk = 0
elif tenure < 24:
    risk = 2
else:
    risk = 1

if st.button("🚀 Predict Churn"):

    features = np.array([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        phone,
        multiple,
        internet,
        security,
        backup,
        protection,
        support,
        tv,
        movies,
        contract,
        paperless,
        payment,
        monthly,
        total,
        avg_charge,
        risk
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    if prediction == 1:

        st.error(
            f"⚠️ Customer Likely To Churn\n\nProbability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Customer Likely To Stay\n\nProbability: {(1-probability):.2%}"
        )

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🚀 Customer Churn Intelligence Dashboard</h4>
    <p>Built using Python, Streamlit, Machine Learning & Data Analytics</p>
    </center>
    """,
    unsafe_allow_html=True
)

