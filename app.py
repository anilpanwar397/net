import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# -----------------------------
# Load Model + Feature Columns
# -----------------------------
model = joblib.load('rf_model.pkl')

with open('feature_column.json', 'r') as f:
    feature_columns = json.load(f)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title='Netflix Customer Churn Prediction', page_icon='🎬', layout='centered')

st.markdown("<h1 style='text-align:center; color:#E50914;'>Netflix Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Enter customer details to predict churn likelihood.</p>", unsafe_allow_html=True)

st.header('Customer Information')

# Numeric inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 70, 30)
    watch_hours = st.number_input('Total Watch Hours (last month)', min_value=0.0, max_value=40.0, value=10.0, step=0.1)
    last_login_days = st.slider('Days since last login', 1, 60, 15)

with col2:
    monthly_fee = st.number_input('Monthly Fee ($)', min_value=5.0, max_value=20.0, value=12.0, step=0.01)
    number_of_profiles = st.slider('Number of Profiles', 1, 5, 2)
    avg_watch_time_per_day = st.number_input('Average Watch Time Per Day (hours)', min_value=0.0, max_value=9.0, value=1.0, step=0.01)

st.subheader('Account & Usage Details')

# Categorical inputs
col3, col4 = st.columns(2)

with col3:
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    subscription_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
    region = st.selectbox('Region', ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'])

with col4:
    device = st.selectbox('Device', ['Desktop', 'Laptop', 'Mobile', 'TV', 'Tablet'])
    payment_method = st.selectbox('Payment Method', ['Credit Card', 'Crypto', 'Debit Card', 'Gift Card', 'PayPal'])
    favorite_genre = st.selectbox('Favorite Genre', ['Action', 'Comedy', 'Documentary', 'Drama', 'Horror', 'Romance', 'Sci-Fi'])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_engineered_features(data):

    # Activity score
    data["activity_score"] = (
        data["watch_hours"]
        + data["avg_watch_time_per_day"] * 10
        - data["last_login_days"]
    )

    # Revenue contribution
    data["revenue_contribution"] = data["monthly_fee"] * data["number_of_profiles"]

    # Engagement level
    if data["avg_watch_time_per_day"] < 0.5:
        data["engagement_level"] = "Low"
    elif data["avg_watch_time_per_day"] < 2:
        data["engagement_level"] = "Medium"
    else:
        data["engagement_level"] = "High"

    # Inactivity level
    if data["last_login_days"] <= 5:
        data["inactivity_level"] = "Active"
    elif data["last_login_days"] <= 15:
        data["inactivity_level"] = "Semi-active"
    else:
        data["inactivity_level"] = "Inactive"

    return data


# Convert to dataframe
user_input = {
    "age": age,
    "watch_hours": watch_hours,
    "last_login_days": last_login_days,
    "monthly_fee": monthly_fee,
    "number_of_profiles": number_of_profiles,
    "avg_watch_time_per_day": avg_watch_time_per_day,
    "gender": gender,
    "subscription_type": subscription_type,
    "region": region,
    "device": device,
    "payment_method": payment_method,
    "favorite_genre": favorite_genre
}

input_df = pd.DataFrame([user_input])

# Apply engineered features
input_df = create_engineered_features(input_df)

# Define all possible categories (consistent with training)
categorical_cols = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
    "engagement_level",
    "inactivity_level"
]

all_categories = {
    "gender": ["Female","Male","Other"],
    "subscription_type": ["Basic","Premium","Standard"],
    "region": ["Africa","Asia","Europe","North America","Oceania","South America"],
    "device": ["Desktop","Laptop","Mobile","TV","Tablet"],
    "payment_method": ["Credit Card","Crypto","Debit Card","Gift Card","PayPal"],
    "favorite_genre": ["Action","Comedy","Documentary","Drama","Horror","Romance","Sci-Fi"],
    "engagement_level": ["Low","Medium","High"],
    "inactivity_level": ["Active","Semi-active","Inactive"]
}

# Convert to consistent categories
for col, cats in all_categories.items():
    input_df[col] = pd.Categorical(input_df[col], categories=cats)

# One-hot encoding
encoded_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align with model's features
final_df = pd.DataFrame(columns=feature_columns)
final_df = pd.concat([final_df, encoded_df], ignore_index=True).fillna(0)
final_df = final_df[feature_columns].astype(float)


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Churn"):
    pred = model.predict(final_df)[0]
    proba = model.predict_proba(final_df)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠️ Customer is likely to CHURN! (Probability: {proba:.2%})")
    else:
        st.success(f"🟢 Customer is likely to stay ACTIVE. (Probability: {proba:.2%})")

    st.write("---")
    st.subheader("Feature Summary")
    st.write(user_input)
    st.write("Engineered Features")
    st.write(input_df[["activity_score","revenue_contribution","engagement_level","inactivity_level"]])
