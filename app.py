import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="RetentionIQ",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/saas_churn_data.csv")


df = load_data()

df["training_status"] = df["training_completed"].map({
    0: "Training Not Completed",
    1: "Training Completed"
})

df["billing_status"] = df["billing_issues"].map({
    0: "No Billing Issues",
    1: "Has Billing Issues"
})

df["customer_status"] = df["churned"].map({
    0: "Active Customer",
    1: "Churned Customer"
})

df["activity_level"] = pd.cut(
    df["days_active_last_30"],
    bins=[-1, 4, 14, 30],
    labels=["Low Activity", "Medium Activity", "High Activity"]
)

# Sidebar filters
st.sidebar.header("Filters")

selected_plans = st.sidebar.multiselect(
    "Plan Type",
    options=df["plan_type"].unique(),
    default=df["plan_type"].unique()
)

selected_company_sizes = st.sidebar.multiselect(
    "Company Size",
    options=df["company_size"].unique(),
    default=df["company_size"].unique()
)

selected_training = st.sidebar.multiselect(
    "Training Status",
    options=df["training_status"].unique(),
    default=df["training_status"].unique()
)

selected_billing = st.sidebar.multiselect(
    "Billing Status",
    options=df["billing_status"].unique(),
    default=df["billing_status"].unique()
)

filtered_df = df[
    (df["plan_type"].isin(selected_plans)) &
    (df["company_size"].isin(selected_company_sizes)) &
    (df["training_status"].isin(selected_training)) &
    (df["billing_status"].isin(selected_billing))
]


st.title("RetentionIQ: SaaS Churn and Revenue Risk Dashboard")

st.write(
    "RetentionIQ helps SaaS teams understand churn patterns, monitor customer risk signals, "
    "and make better retention decisions using product usage and customer behaviour data."
)

with st.expander("How to use this dashboard"):
    st.markdown(
        """
        This dashboard helps you understand which customers are more likely to churn and what factors may be causing it.

        **How to use it:**

        1. Use the filters on the left sidebar to focus on specific customer groups.
        2. Check the key metrics to see the total customers, churn rate, and revenue at risk.
        3. Review the dataset preview to see the type of customer records being analysed.
        4. Look for patterns across plan type, training status, billing status, and support activity.

        **What the labels mean:**

        - **Active Customer**: A customer who has not churned.
        - **Churned Customer**: A customer who has stopped using or cancelled the product.
        - **Training Completed**: The customer completed onboarding or product training.
        - **Has Billing Issues**: The customer experienced billing or payment-related problems.
        - **Revenue Lost**: Monthly revenue linked to customers who churned.
        """
    )

st.subheader("Key Metrics")

total_customers = len(filtered_df)
churned_customers = filtered_df["churned"].sum()
churn_rate = filtered_df["churned"].mean() * 100
monthly_revenue = filtered_df["monthly_fee"].sum()
revenue_lost = filtered_df.loc[filtered_df["churned"] == 1, "monthly_fee"].sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")
col5.metric("Revenue Lost", f"${revenue_lost:,.0f}")

st.subheader("Churn Rate by Plan Type")

churn_by_plan = (
    filtered_df.groupby("plan_type")["churned"]
    .mean()
    .reset_index()
)

churn_by_plan["churn_rate"] = churn_by_plan["churned"] * 100

fig_plan = px.bar(
    churn_by_plan,
    x="plan_type",
    y="churn_rate",
    text=churn_by_plan["churn_rate"].round(1),
    labels={
        "plan_type": "Plan Type",
        "churn_rate": "Churn Rate (%)"
    },
    title="Percentage of Customers Who Left by Plan"
)

fig_plan.update_traces(texttemplate="%{text}%", textposition="outside")
fig_plan.update_layout(yaxis_range=[0, churn_by_plan["churn_rate"].max() + 10])

st.plotly_chart(fig_plan, width='stretch')

st.subheader("Churn Rate by Customer Activity")

churn_by_activity = (
    filtered_df.groupby("activity_level")["churned"]
    .mean()
    .reset_index()
)

churn_by_activity["churn_rate"] = churn_by_activity["churned"] * 100

fig_activity = px.bar(
    churn_by_activity,
    x="activity_level",
    y="churn_rate",
    text=churn_by_activity["churn_rate"].round(1),
    labels={
        "activity_level": "Customer Activity",
        "churn_rate": "Churn Rate (%)"
    },
    title="Percentage of Customers Who Left by Activity Level"
)

fig_activity.update_traces(texttemplate="%{text}%", textposition="outside")
fig_activity.update_layout(yaxis_range=[0, churn_by_activity["churn_rate"].max() + 10])

st.plotly_chart(fig_activity, use_container_width=True)

st.info(
    "Customers with low activity are more likely to leave. This helps teams know which customers may need attention early."
)

st.subheader("Dataset Preview")
preview_columns = [
    "customer_id",
    "plan_type",
    "company_size",
    "days_active_last_30",
    "last_login_days_ago",
    "features_used",
    "support_tickets",
    "training_status",
    "billing_status",
    "customer_status",
    "monthly_fee"
]

st.dataframe(filtered_df[preview_columns].head())