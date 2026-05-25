import os
import numpy as np
import pandas as pd


np.random.seed(42)


def generate_saas_churn_data(number_of_customers=2000):
    plans = ["Free", "Starter", "Pro", "Enterprise"]
    countries = ["UK", "US", "Nigeria", "Canada", "Germany"]
    company_sizes = ["Small", "Medium", "Large"]

    data = {
        "customer_id": [f"CUST-{i}" for i in range(1, number_of_customers + 1)],

        "plan_type": np.random.choice(
            plans,
            number_of_customers,
            p=[0.25, 0.35, 0.30, 0.10]
        ),
        "country": np.random.choice(countries, number_of_customers),
        "company_size": np.random.choice(
            company_sizes,
            number_of_customers,
            p=[0.55, 0.30, 0.15]
        ),

        "days_active_last_30": np.random.randint(0, 31, number_of_customers),
        "last_login_days_ago": np.random.randint(0, 31, number_of_customers),
        "features_used": np.random.randint(1, 11, number_of_customers),
        "team_members_added": np.random.randint(0, 21, number_of_customers),

        "support_tickets": np.random.randint(0, 6, number_of_customers),
        "billing_issues": np.random.choice(
            [0, 1],
            number_of_customers,
            p=[0.85, 0.15]
        ),

        "subscription_age_months": np.random.randint(1, 37, number_of_customers),
        "emails_opened_last_30": np.random.randint(0, 16, number_of_customers),
        "training_completed": np.random.choice(
            [0, 1],
            number_of_customers,
            p=[0.35, 0.65]
        ),
    }

    df = pd.DataFrame(data)

    monthly_fee_map = {
        "Free": 0,
        "Starter": 29,
        "Pro": 79,
        "Enterprise": 249
    }

    df["monthly_fee"] = df["plan_type"].map(monthly_fee_map)

    df["signup_date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 700, number_of_customers),
        unit="D"
    )

    churn_score = (
        0.05
        + (df["days_active_last_30"] < 5) * 0.20
        + (df["last_login_days_ago"] > 14) * 0.15
        + (df["features_used"] <= 2) * 0.15
        + (df["support_tickets"] >= 3) * 0.10
        + (df["billing_issues"] == 1) * 0.15
        + (df["training_completed"] == 0) * 0.10
        + (df["emails_opened_last_30"] <= 2) * 0.08
        + (df["plan_type"] == "Free") * 0.08
        - (df["plan_type"] == "Enterprise") * 0.12
        - (df["days_active_last_30"] >= 20) * 0.10
        - (df["features_used"] >= 7) * 0.08
    )

    churn_score = np.clip(churn_score, 0.01, 0.95)

    df["churned"] = np.random.binomial(1, churn_score)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_saas_churn_data()

    df.to_csv("data/saas_churn_data.csv", index=False)

    print("Dataset created successfully.")
    print("File saved to: data/saas_churn_data.csv")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Churn rate:", round(df["churned"].mean() * 100, 2), "%")
    print(df.head())