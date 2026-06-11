import pandas as pd
import streamlit as st
import plotly.express as px
import joblib


st.set_page_config(
    page_title="RetentionIQ",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/saas_churn_data.csv")

@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    model_features = joblib.load("models/model_features.pkl")
    return model, model_features

df = load_data()
model, model_features = load_model()


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

df["feature_usage_level"] = pd.cut(
    df["features_used"],
    bins=[0, 2, 6, 10],
    labels=["Low Feature Usage", "Medium Feature Usage", "High Feature Usage"]
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

st.subheader("Churn Rate by Feature Usage")

churn_by_feature_usage = (
    filtered_df.groupby("feature_usage_level")["churned"]
    .mean()
    .reset_index()
)

churn_by_feature_usage["churn_rate"] = churn_by_feature_usage["churned"] * 100

fig_feature_usage = px.bar(
    churn_by_feature_usage,
    x="feature_usage_level",
    y="churn_rate",
    text=churn_by_feature_usage["churn_rate"].round(1),
    labels={
        "feature_usage_level": "Feature Usage",
        "churn_rate": "Churn Rate (%)"
    },
    title="Percentage of Customers Who Left by Feature Usage"
)

fig_feature_usage.update_traces(texttemplate="%{text}%", textposition="outside")
fig_feature_usage.update_layout(
    yaxis_range=[0, churn_by_feature_usage["churn_rate"].max() + 10]
)

st.plotly_chart(fig_feature_usage, use_container_width=True)

st.info(
    "Customers who use fewer features are more likely to leave. This suggests that helping customers discover and use more features can improve retention."
)

st.subheader("Churn Rate by Training and Billing Status")

col1, col2 = st.columns(2)

with col1:
    churn_by_training = (
        filtered_df.groupby("training_status")["churned"]
        .mean()
        .reset_index()
    )

    churn_by_training["churn_rate"] = churn_by_training["churned"] * 100

    fig_training = px.bar(
        churn_by_training,
        x="training_status",
        y="churn_rate",
        text=churn_by_training["churn_rate"].round(1),
        labels={
            "training_status": "Training Status",
            "churn_rate": "Churn Rate (%)"
        },
        title="Customers Who Left by Training Status"
    )

    fig_training.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_training.update_layout(
        yaxis_range=[0, churn_by_training["churn_rate"].max() + 10]
    )

    st.plotly_chart(fig_training, use_container_width=True)

with col2:
    churn_by_billing = (
        filtered_df.groupby("billing_status")["churned"]
        .mean()
        .reset_index()
    )

    churn_by_billing["churn_rate"] = churn_by_billing["churned"] * 100

    fig_billing = px.bar(
        churn_by_billing,
        x="billing_status",
        y="churn_rate",
        text=churn_by_billing["churn_rate"].round(1),
        labels={
            "billing_status": "Billing Status",
            "churn_rate": "Churn Rate (%)"
        },
        title="Customers Who Left by Billing Status"
    )

    fig_billing.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_billing.update_layout(
        yaxis_range=[0, churn_by_billing["churn_rate"].max() + 10]
    )

    st.plotly_chart(fig_billing, use_container_width=True)

st.info(
    "Customers who do not complete training or have billing problems are more likely to leave. "
    "This shows that onboarding and payment experience are important for retention."
)

st.subheader("Key Findings")
st.success(
    """
    - Customers with low activity are more likely to leave.
    - Customers who use fewer features have a higher churn rate.
    - Customers with billing problems are more likely to churn.
    - Customers who complete training are more likely to stay.
    - Enterprise customers show the lowest churn risk compared to other plans.
    """
)

st.subheader("Predict Customer Churn Risk")

st.write(
    "Enter a customer's details below to estimate how likely they are to leave."
)

with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        plan_type = st.selectbox(
            "Plan Type",
            ["Free", "Starter", "Pro", "Enterprise"]
        )

        company_size = st.selectbox(
            "Company Size",
            ["Small", "Medium", "Large"]
        )

        days_active_last_30 = st.slider(
            "Days Active in the Last 30 Days",
            min_value=0,
            max_value=30,
            value=10
        )

        last_login_days_ago = st.slider(
            "Days Since Last Login",
            min_value=0,
            max_value=30,
            value=7
        )

        features_used = st.slider(
            "Number of Features Used",
            min_value=1,
            max_value=10,
            value=4
        )

        team_members_added = st.slider(
            "Team Members Added",
            min_value=0,
            max_value=20,
            value=3
        )

    with col2:
        support_tickets = st.slider(
            "Support Requests",
            min_value=0,
            max_value=5,
            value=1
        )

        billing_status_input = st.selectbox(
            "Billing Status",
            ["No Billing Issues", "Has Billing Issues"]
        )

        subscription_age_months = st.slider(
            "Subscription Age in Months",
            min_value=1,
            max_value=36,
            value=12
        )

        emails_opened_last_30 = st.slider(
            "Emails Opened in the Last 30 Days",
            min_value=0,
            max_value=15,
            value=5
        )

        training_status_input = st.selectbox(
            "Training Status",
            ["Training Not Completed", "Training Completed"]
        )

    submitted = st.form_submit_button("Predict Churn Risk")


if submitted:
    monthly_fee_map = {
        "Free": 0,
        "Starter": 29,
        "Pro": 79,
        "Enterprise": 249
    }

    input_data = pd.DataFrame({
        "plan_type": [plan_type],
        "company_size": [company_size],
        "days_active_last_30": [days_active_last_30],
        "last_login_days_ago": [last_login_days_ago],
        "features_used": [features_used],
        "team_members_added": [team_members_added],
        "support_tickets": [support_tickets],
        "billing_issues": [1 if billing_status_input == "Has Billing Issues" else 0],
        "subscription_age_months": [subscription_age_months],
        "emails_opened_last_30": [emails_opened_last_30],
        "training_completed": [1 if training_status_input == "Training Completed" else 0],
        "monthly_fee": [monthly_fee_map[plan_type]]
    })

    input_encoded = pd.get_dummies(input_data)

    input_encoded = input_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    churn_probability = model.predict_proba(input_encoded)[0][1] * 100

    if churn_probability >= 70:
        risk_level = "High Risk"
        recommendation = (
            "This customer may need urgent attention. Reach out personally, "
            "check for blockers, and help them get more value from the product."
        )
    elif churn_probability >= 40:
        risk_level = "Medium Risk"
        recommendation = (
            "This customer should be monitored. Encourage product usage, "
            "offer support, and guide them toward useful features."
        )
    else:
        risk_level = "Low Risk"
        recommendation = (
            "This customer looks relatively healthy. Keep supporting their product usage."
        )

st.subheader("Prediction Result")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.metric("Churn Risk Score", f"{churn_probability:.1f}%")

with result_col2:
    st.metric("Risk Level", risk_level)

if risk_level == "High Risk":
    st.error(recommendation)
elif risk_level == "Medium Risk":
    st.warning(recommendation)
else:
    st.success(recommendation)

with st.expander("About this prediction"):
    st.markdown(
        """
        This prediction is based on a machine learning model trained on customer behaviour data.

        The model looks at signals such as:

        - How active the customer has been recently
        - How many product features they use
        - Whether they completed training
        - Whether they had billing problems
        - How many support requests they made
        - Their plan type and subscription age

        The score is not a final decision. It is a guide to help teams know which customers may need attention earlier.
        """
    )

st.subheader("Customer Data Preview")
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