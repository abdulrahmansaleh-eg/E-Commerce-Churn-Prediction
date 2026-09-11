import streamlit as st
import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "xgboost_tuned.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()


ENCODED_COLUMNS = [
    'Age', 'Gender', 'Membership_Years', 'Login_Frequency', 'Session_Duration_Avg',
    'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Wishlist_Items', 'Total_Purchases',
    'Average_Order_Value', 'Days_Since_Last_Purchase', 'Discount_Usage_Rate',
    'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls',
    'Product_Reviews_Written', 'Social_Media_Engagement_Score', 'Mobile_App_Usage',
    'Payment_Method_Diversity', 'Lifetime_Value', 'Credit_Balance',
    'Country_Canada', 'Country_France', 'Country_Germany', 'Country_India',
    'Country_Japan', 'Country_UK', 'Country_USA',
    'City_Bangalore', 'City_Berlin', 'City_Birmingham', 'City_Brisbane', 'City_Calgary',
    'City_Chennai', 'City_Chicago', 'City_Cologne', 'City_Delhi', 'City_Frankfurt',
    'City_Glasgow', 'City_Hamburg', 'City_Houston', 'City_Hyderabad', 'City_Kyoto',
    'City_Leeds', 'City_London', 'City_Los Angeles', 'City_Lyon', 'City_Manchester',
    'City_Marseille', 'City_Melbourne', 'City_Montreal', 'City_Mumbai', 'City_Munich',
    'City_Nagoya', 'City_New York', 'City_Nice', 'City_Osaka', 'City_Ottawa',
    'City_Paris', 'City_Perth', 'City_Phoenix', 'City_Sydney', 'City_Tokyo',
    'City_Toronto', 'City_Toulouse', 'City_Vancouver', 'City_Yokohama',
    'Signup_Quarter_Q2', 'Signup_Quarter_Q3', 'Signup_Quarter_Q4',
]

COUNTRIES = ["Australia", "Canada", "France", "Germany", "India", "Japan", "UK", "USA"]

CITIES = [
    "Bangalore", "Berlin", "Birmingham", "Brisbane", "Calgary", "Chennai", "Chicago",
    "Cologne", "Delhi", "Frankfurt", "Glasgow", "Hamburg", "Houston", "Hyderabad",
    "Kyoto", "Leeds", "London", "Los Angeles", "Lyon", "Manchester", "Marseille",
    "Melbourne", "Montreal", "Mumbai", "Munich", "Nagoya", "New York", "Nice",
    "Osaka", "Ottawa", "Paris", "Perth", "Phoenix", "Sydney", "Tokyo", "Toronto",
    "Toulouse", "Vancouver", "Yokohama", "Other / Not Listed",
]

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

st.set_page_config(page_title="Customer Prediction", layout="centered")

# ============================================================
# تخصيص الشكل (CSS) — نسخة مودرن نهائية
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* خلفية متدرجة هادئة */
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1B1F3B 0%, #0B0D1A 60%);
        color: #E8E8F0;
    }

    /* العنوان */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #7F5AF0, #2CB1BC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    /* وصف تحت العنوان */
    .stApp p {
        color: #9A9AB5;
    }

    /* بطاقة زجاجية حوالين الـ inputs */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2em 1.4em;
        backdrop-filter: blur(6px);
    }

    /* تسميات الـ inputs */
    label {
        color: #C4C4E0 !important;
        font-weight: 500 !important;
        font-size: 0.85em !important;
    }

    /* صناديق الأرقام (زي Age) - خلفية غامقة + خط واضح + بوردر */
    .stNumberInput input {
        background-color: #1B1F3B !important;
        color: #FFFFFF !important;
        caret-color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
    }

    /* أزرار +/- بتاعة العداد */
    .stNumberInput button {
        background-color: #262B4D !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* صندوق الـ Selectbox (Gender, Country, City, Signup Quarter) - selector قوي وعام + نفس بوردر الأرقام */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #1B1F3B !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSelectbox"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSelectbox"] svg {
        fill: #C4C4E0 !important;
    }

    /* القائمة اللي بتفتح لما تدوس على الـ Selectbox */
    ul[data-baseweb="menu"],
    div[data-baseweb="popover"] ul {
        background-color: #1B1F3B !important;
    }

    ul[data-baseweb="menu"] li,
    div[data-baseweb="popover"] ul li {
        color: #FFFFFF !important;
        background-color: #1B1F3B !important;
    }

    ul[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] ul li:hover {
        background-color: #2CB1BC !important;
    }

    /* ============================================================
       زرار Predict
       ============================================================ */
    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 1.5em;
    }

    .stButton > button {
        background: linear-gradient(90deg, #7F5AF0, #2CB1BC);
        color: #FFFFFF !important;
        border: none;
        border-radius: 12px;
        padding: 0.75em 3em;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05em;
        font-weight: 700;
        letter-spacing: 0.3px;
        width: 100%;
        max-width: 420px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
        box-shadow: 0 4px 18px rgba(127, 90, 240, 0.35);
    }

    .stButton > button p,
    .stButton > button div,
    .stButton > button span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 26px rgba(44, 177, 188, 0.5);
        opacity: 0.95;
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* رسالة النتيجة (Churned / Not Churned) */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Customer Prediction App")
st.write("Fill in the customer details to get the prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    country = st.selectbox("Country", COUNTRIES)
    city = st.selectbox("City", CITIES)
    membership_years = st.number_input("Membership Years", min_value=0.0, value=1.0)
    login_frequency = st.number_input("Login Frequency", min_value=0.0, value=5.0)
    session_duration_avg = st.number_input("Session Duration Avg", min_value=0.0, value=10.0)
    pages_per_session = st.number_input("Pages Per Session", min_value=0.0, value=5.0)
    cart_abandonment_rate = st.number_input("Cart Abandonment Rate", min_value=0.0, max_value=1.0, value=0.3)
    wishlist_items = st.number_input("Wishlist Items", min_value=0, value=2)
    total_purchases = st.number_input("Total Purchases", min_value=0, value=10)
    average_order_value = st.number_input("Average Order Value", min_value=0.0, value=50.0)

with col2:
    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)
    discount_usage_rate = st.number_input("Discount Usage Rate", min_value=0.0, max_value=1.0, value=0.2)
    returns_rate = st.number_input("Returns Rate", min_value=0.0, max_value=1.0, value=0.1)
    email_open_rate = st.number_input("Email Open Rate", min_value=0.0, max_value=1.0, value=0.4)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)
    product_reviews_written = st.number_input("Product Reviews Written", min_value=0, value=1)
    social_media_engagement_score = st.number_input("Social Media Engagement Score", min_value=0.0, value=50.0)
    mobile_app_usage = st.number_input("Mobile App Usage", min_value=0.0, value=10.0)
    payment_method_diversity = st.number_input("Payment Method Diversity", min_value=0, value=2)
    lifetime_value = st.number_input("Lifetime Value", min_value=0.0, value=500.0)
    credit_balance = st.number_input("Credit Balance", min_value=0.0, value=0.0)
    signup_quarter = st.selectbox("Signup Quarter", QUARTERS)

if st.button("Predict"):
    row = pd.DataFrame(np.zeros((1, len(ENCODED_COLUMNS))), columns=ENCODED_COLUMNS)

    row.loc[0, "Age"] = age
    row.loc[0, "Membership_Years"] = membership_years
    row.loc[0, "Login_Frequency"] = login_frequency
    row.loc[0, "Session_Duration_Avg"] = session_duration_avg
    row.loc[0, "Pages_Per_Session"] = pages_per_session
    row.loc[0, "Cart_Abandonment_Rate"] = cart_abandonment_rate
    row.loc[0, "Wishlist_Items"] = wishlist_items
    row.loc[0, "Total_Purchases"] = total_purchases
    row.loc[0, "Average_Order_Value"] = average_order_value
    row.loc[0, "Days_Since_Last_Purchase"] = days_since_last_purchase
    row.loc[0, "Discount_Usage_Rate"] = discount_usage_rate
    row.loc[0, "Returns_Rate"] = returns_rate
    row.loc[0, "Email_Open_Rate"] = email_open_rate
    row.loc[0, "Customer_Service_Calls"] = customer_service_calls
    row.loc[0, "Product_Reviews_Written"] = product_reviews_written
    row.loc[0, "Social_Media_Engagement_Score"] = social_media_engagement_score
    row.loc[0, "Mobile_App_Usage"] = mobile_app_usage
    row.loc[0, "Payment_Method_Diversity"] = payment_method_diversity
    row.loc[0, "Lifetime_Value"] = lifetime_value
    row.loc[0, "Credit_Balance"] = credit_balance

    row.loc[0, "Gender"] = 1 if gender == "Male" else 0

    country_col = f"Country_{country}"
    if country_col in ENCODED_COLUMNS:
        row.loc[0, country_col] = 1

    city_col = f"City_{city}"
    if city_col in ENCODED_COLUMNS:
        row.loc[0, city_col] = 1

    quarter_col = f"Signup_Quarter_{signup_quarter}"
    if quarter_col in ENCODED_COLUMNS:
        row.loc[0, quarter_col] = 1

    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]

    if prediction == 1:
        st.error("🔴 Prediction: Churned")
    else:
        st.success("🟢 Prediction: Not Churned")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        st.write("Probabilities:", proba)
