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


st.title("RetentionIQ: SaaS Churn and Revenue Risk Dashboard")

st.write(
    "RetentionIQ helps SaaS teams understand churn patterns, monitor customer risk signals, "
    "and make better retention decisions using product usage and customer behaviour data."
)

st.subheader("Dataset Preview")
st.dataframe(df.head())