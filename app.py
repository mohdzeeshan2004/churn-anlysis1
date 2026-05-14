import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """Load uploaded data or default project data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("data/European_Bank.csv")


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean columns that are not useful for analysis/modeling."""
    data = dataframe.copy()
    data = data.drop_duplicates()

    # Keep CustomerId in raw data, but remove identifier/name columns from analysis dataset.
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")
    return data


def create_age_group(age):
    if age < 30:
        return "Below 30"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    return "60+"


def churn_label(value):
    return "Churned" if value == 1 else "Retained"


def get_filtered_data(data: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters used in dashboard pages."""
    filtered = data.copy()

    st.sidebar.header("Dashboard Filters")

    if "Year" in filtered.columns:
        years = sorted(filtered["Year"].dropna().unique())
        selected_years = st.sidebar.multiselect("Year", years, default=years)
        filtered = filtered[filtered["Year"].isin(selected_years)]

    if "Geography" in filtered.columns:
        countries = sorted(filtered["Geography"].dropna().unique())
        selected_countries = st.sidebar.multiselect("Country", countries, default=countries)
        filtered = filtered[filtered["Geography"].isin(selected_countries)]

    if "Gender" in filtered.columns:
        genders = sorted(filtered["Gender"].dropna().unique())
        selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
        filtered = filtered[filtered["Gender"].isin(selected_genders)]

    if "Age" in filtered.columns and not filtered.empty:
        min_age = int(filtered["Age"].min())
        max_age = int(filtered["Age"].max())
        selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
        filtered = filtered[(filtered["Age"] >= selected_age[0]) & (filtered["Age"] <= selected_age[1])]

    return filtered


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


# --------------------------------------------------
# Header and sidebar
# --------------------------------------------------
st.title("🏦 Customer Segmentation & Churn Pattern Analytics")
st.subheader("European Banking Churn Analysis Streamlit App")
st.markdown(
    "An interactive analytics app for studying churn behavior, customer segments, "
    "financial profiles, and retention opportunities in European banking data."
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset Overview",
        "Churn Dashboard",
        "Customer Segmentation",
        "ML Churn Prediction",
        "Insights & Recommendations",
    ],
)

uploaded_file = st.sidebar.file_uploader("Optional: Upload another CSV", type=["csv"])

try:
    raw_df = load_data(uploaded_file)
    df = clean_dataset(raw_df)
except Exception as error:
    raw_df = None
    df = None
    st.error(f"Could not load dataset: {error}")


# --------------------------------------------------
# Project Overview
# --------------------------------------------------
if page == "Project Overview":
    st.header("📌 Project Overview")

    st.markdown(
        """
        ### Background
        Customer churn is one of the biggest challenges in retail banking. When customers leave, banks lose revenue,
        relationship value, and future cross-selling opportunities.

        ### Problem Statement
        Banks may have rich customer-level data, but they need clear segmentation-based insights to understand which
        customers are most likely to churn and what actions can reduce churn risk.

        ### Project Objectives
        - Analyze churn distribution across customers
        - Compare churn across geography, gender, age groups, balance, products, and activity status
        - Identify high-risk and high-value churn groups
        - Segment customers using K-Means clustering
        - Build a basic machine learning churn prediction model
        - Convert analysis into business recommendations

        ### Tools Used
        Python, Pandas, NumPy, Plotly, Scikit-learn, and Streamlit.
        """
    )

    if raw_df is not None:
        st.success("Default dataset loaded successfully from `data/European_Bank.csv`.")
        st.write("Dataset shape:", raw_df.shape)


# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
elif page == "Dataset Overview":
    st.header("🔍 Dataset Overview")

    if df is None:
        st.error("Dataset not found. Please check `data/European_Bank.csv` or upload a CSV from the sidebar.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{raw_df.shape[0]:,}")
        col2.metric("Columns", f"{raw_df.shape[1]:,}")
        col3.metric("Missing Values", f"{raw_df.isnull().sum().sum():,}")
        col4.metric("Duplicate Rows", f"{raw_df.duplicated().sum():,}")

        st.subheader("Raw Dataset Preview")
        st.dataframe(raw_df.head(20), use_container_width=True)

        st.subheader("Analysis Dataset Preview")
        st.caption("Identifier/name columns such as CustomerId and Surname are removed for analysis/modeling.")
        st.dataframe(df.head(20), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Column Data Types")
            dtype_df = pd.DataFrame({"Column": raw_df.columns, "Data Type": raw_df.dtypes.astype(str).values})
            st.dataframe(dtype_df, use_container_width=True)
        with right:
            st.subheader("Missing Value Summary")
            missing_df = raw_df.isnull().sum().reset_index()
            missing_df.columns = ["Column", "Missing Values"]
            st.dataframe(missing_df, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(raw_df.describe(include="all"), use_container_width=True)


# --------------------------------------------------
# Churn Dashboard
# --------------------------------------------------
elif page == "Churn Dashboard":
    st.header("📊 Churn Dashboard")

    if df is None:
        st.error("Dataset not found. Please check `data/European_Bank.csv` or upload a CSV from the sidebar.")
    elif "Exited" not in df.columns:
        st.error("The dataset must contain an `Exited` column where 1 = churned and 0 = retained.")
    else:
        filtered_df = get_filtered_data(df)

        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
            st.stop()

        total_customers = len(filtered_df)
        churned_customers = int(filtered_df["Exited"].sum())
        retained_customers = total_customers - churned_customers
        churn_rate = churned_customers / total_customers * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churned Customers", f"{churned_customers:,}")
        col3.metric("Retained Customers", f"{retained_customers:,}")
        col4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            churn_count = filtered_df["Exited"].value_counts().reset_index()
            churn_count.columns = ["Exited", "Count"]
            churn_count["Status"] = churn_count["Exited"].apply(churn_label)
            fig = px.pie(churn_count, names="Status", values="Count", hole=0.45, title="Churned vs Retained Customers")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Geography" in filtered_df.columns:
                country_churn = filtered_df.groupby("Geography", as_index=False)["Exited"].mean()
                country_churn["Churn Rate (%)"] = country_churn["Exited"] * 100
                fig = px.bar(country_churn, x="Geography", y="Churn Rate (%)", text="Churn Rate (%)", title="Country-wise Churn Rate")
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "Age" in filtered_df.columns:
                age_df = filtered_df.copy()
                age_df["Age Group"] = age_df["Age"].apply(create_age_group)
                age_churn = age_df.groupby("Age Group", as_index=False)["Exited"].mean()
                age_churn["Churn Rate (%)"] = age_churn["Exited"] * 100
                fig = px.bar(age_churn, x="Age Group", y="Churn Rate (%)", text="Churn Rate (%)", title="Age Group Churn Rate")
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            if "NumOfProducts" in filtered_df.columns:
                product_churn = filtered_df.groupby("NumOfProducts", as_index=False)["Exited"].mean()
                product_churn["Churn Rate (%)"] = product_churn["Exited"] * 100
                fig = px.bar(product_churn, x="NumOfProducts", y="Churn Rate (%)", text="Churn Rate (%)", title="Products Used vs Churn Rate")
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            if "Balance" in filtered_df.columns:
                chart_data = filtered_df.copy()
                chart_data["Status"] = chart_data["Exited"].apply(churn_label)
                fig = px.box(chart_data, x="Status", y="Balance", title="Balance Distribution by Churn Status")
                st.plotly_chart(fig, use_container_width=True)

        with c6:
            if "CreditScore" in filtered_df.columns:
                chart_data = filtered_df.copy()
                chart_data["Status"] = chart_data["Exited"].apply(churn_label)
                fig = px.box(chart_data, x="Status", y="CreditScore", title="Credit Score Distribution by Churn Status")
                st.plotly_chart(fig, use_container_width=True)

        if "IsActiveMember" in filtered_df.columns:
            st.subheader("Active Member Churn Analysis")
            active_churn = filtered_df.groupby("IsActiveMember", as_index=False)["Exited"].mean()
            active_churn["Membership Status"] = active_churn["IsActiveMember"].map({0: "Inactive", 1: "Active"})
            active_churn["Churn Rate (%)"] = active_churn["Exited"] * 100
            fig = px.bar(active_churn, x="Membership Status", y="Churn Rate (%)", text="Churn Rate (%)", title="Churn Rate by Activity Status")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Customer Segmentation
# --------------------------------------------------
elif page == "Customer Segmentation":
    st.header("🧩 Customer Segmentation")

    if df is None:
        st.error("Dataset not found. Please check `data/European_Bank.csv` or upload a CSV from the sidebar.")
    else:
        numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) < 3:
            st.error("Not enough numerical columns for segmentation.")
        else:
            st.markdown("K-Means clustering is used to create customer segments using customer financial and behavioral features.")

            k = st.slider("Select Number of Segments", 2, 6, 3)

            segment_df = df[available_cols].dropna().copy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(segment_df)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            segment_df["Segment"] = kmeans.fit_predict(scaled_data)

            st.subheader("Segment Size")
            segment_count = segment_df["Segment"].value_counts().reset_index()
            segment_count.columns = ["Segment", "Customer Count"]
            fig = px.bar(segment_count, x="Segment", y="Customer Count", text="Customer Count", title="Customer Count by Segment")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Segment Profile Summary")
            summary = segment_df.groupby("Segment")[available_cols].mean().round(2)
            st.dataframe(summary, use_container_width=True)

            if "Age" in segment_df.columns and "Balance" in segment_df.columns:
                st.subheader("Age vs Balance Customer Segments")
                fig = px.scatter(segment_df, x="Age", y="Balance", color="Segment", title="Customer Segments: Age vs Balance")
                st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# ML Churn Prediction
# --------------------------------------------------
elif page == "ML Churn Prediction":
    st.header("🤖 Machine Learning Churn Prediction")

    if df is None:
        st.error("Dataset not found. Please check `data/European_Bank.csv` or upload a CSV from the sidebar.")
    elif "Exited" not in df.columns:
        st.error("The dataset must contain an `Exited` target column.")
    else:
        model_df = df.copy().dropna()
        X = model_df.drop(columns=["Exited"])
        y = model_df["Exited"]
        X = pd.get_dummies(X, drop_first=True)

        if len(X) < 20 or y.nunique() < 2:
            st.warning("Dataset is too small or has only one target class. Upload a larger dataset for model training.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

            left, right = st.columns(2)
            with left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Retained", "Actual Churned"],
                    columns=["Predicted Retained", "Predicted Churned"],
                )
                st.dataframe(cm_df, use_container_width=True)

            with right:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

            st.subheader("Top Feature Importance")
            feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False).head(15)
            fig = px.bar(feature_importance, x="Importance", y="Feature", orientation="h", title="Top 15 Important Features")
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Insights & Recommendations
# --------------------------------------------------
elif page == "Insights & Recommendations":
    st.header("💡 Insights & Recommendations")

    st.markdown(
        """
        ### Key Insights to Discuss
        - Geography-wise churn helps identify regions that need stronger retention strategies.
        - High-balance customers who churn represent high-value revenue risk.
        - Inactive members are important churn-risk candidates.
        - Product usage can indicate the depth of a customer's relationship with the bank.
        - Age groups with higher churn need targeted communication and offers.

        ### Business Recommendations
        - Create retention offers for high-balance customers showing churn risk.
        - Build reactivation campaigns for inactive members.
        - Offer personalized product bundles to customers using fewer services.
        - Track churn rate monthly as a core banking KPI.
        - Use customer segmentation to personalize marketing and relationship management.

        ### Final Outcome
        This app converts raw customer data into a business-ready churn dashboard, customer segmentation output,
        machine learning model results, and actionable banking recommendations.
        """
    )


st.sidebar.divider()
st.sidebar.caption("European Bank Churn Analytics Project")
