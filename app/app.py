from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name.lower() == "app" else APP_DIR

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_fraud_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Hour", "Amount_log"]
)

THRESHOLD = 0.50

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# HIGH-CONTRAST PROFESSIONAL THEME
# ================================================================
st.markdown(
    """
    <style>

    /* ---------- MAIN APP ---------- */
    .stApp {
        background-color: #0B1120 !important;
        color: #FFFFFF !important;
    }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #334155 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption {
        color: #FFFFFF !important;
    }

    /* ---------- ALL HEADINGS ---------- */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeader"] {
        color: #FFFFFF !important;
    }

    /* ---------- CUSTOM MAIN TITLE ---------- */
    .title {
        font-size: 40px;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #E2E8F0 !important;
        margin-bottom: 28px;
    }

    /* ---------- GOLD SECTION HEADINGS ---------- */
    .section {
        font-size: 22px;
        font-weight: 800;
        color: #FFD166 !important;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* ---------- STREAMLIT LABELS ---------- */
    label,
    label p,
    label span,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] label * {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    /* ---------- NUMBER INPUT BOXES ---------- */
    [data-testid="stNumberInput"] input {
        color: #FFFFFF !important;
        background-color: #172033 !important;
        border: 1px solid #64748B !important;
        border-radius: 7px !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] input::placeholder {
        color: #CBD5E1 !important;
        opacity: 1 !important;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #FFD166 !important;
        box-shadow: 0 0 0 1px #FFD166 !important;
    }

    /* ---------- RADIO / NAVIGATION ---------- */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stRadio"] > div {
        color: #FFFFFF !important;
    }

    /* ---------- MARKDOWN TEXT ---------- */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    .stMarkdown li,
    .stMarkdown strong,
    .stMarkdown b {
        color: #FFFFFF !important;
    }

    /* Keep intentionally colored custom text */
    .stMarkdown .title,
    .stMarkdown .subtitle,
    .stMarkdown .section,
    .stMarkdown .fraud-title,
    .stMarkdown .legitimate-title {
        color: inherit !important;
    }

    /* ---------- INFO / ALERT BOX ---------- */
    [data-testid="stAlert"] {
        background-color: #172033 !important;
        border: 1px solid #475569 !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* ---------- BUTTON ---------- */
    .stButton > button {
        width: 100%;
        min-height: 45px;
        background-color: #D97706 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFD166 !important;
        border-radius: 8px;
        font-weight: 800;
        font-size: 16px;
    }

    .stButton > button p,
    .stButton > button span {
        color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background-color: #F59E0B !important;
        color: #111827 !important;
    }

    .stButton > button:hover p,
    .stButton > button:hover span {
        color: #111827 !important;
    }

    /* ---------- METRICS ---------- */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* ---------- CAPTION ---------- */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .stCaption {
        color: #E2E8F0 !important;
        opacity: 1 !important;
    }

    /* ---------- GENERAL TEXT ---------- */
    p, li, span, div {
        opacity: 1;
    }

    /* ---------- CARDS ---------- */
    .card {
        background-color: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* ---------- FRAUD RESULT ---------- */
    .fraud {
        background-color: #35171A;
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
    }

    .fraud-title {
        color: #FF7B7B !important;
        font-size: 28px;
        font-weight: 800;
    }

    /* ---------- LEGITIMATE RESULT ---------- */
    .legitimate {
        background-color: #10291F;
        border: 1px solid #22C55E;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
    }

    .legitimate-title {
        color: #5EEB8A !important;
        font-size: 28px;
        font-weight: 800;
    }

    /* ---------- DIVIDER ---------- */
    hr {
        border-color: #334155 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found:\n{SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def make_prediction(time_value, amount_value, feature_values):
    model, scaler = load_model_and_scaler()

    transaction = {
        "Time": float(time_value),
        "Amount": float(amount_value),
    }

    transaction.update(feature_values)

    # Same feature engineering used during training
    transaction["Hour"] = (transaction["Time"] // 3600) % 24
    transaction["Amount_log"] = np.log1p(transaction["Amount"])

    # Exact training feature order
    input_data = pd.DataFrame([transaction])[FEATURE_COLUMNS]

    # Use the saved training scaler
    input_scaled = scaler.transform(input_data)

    probability = float(model.predict_proba(input_scaled)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return prediction, probability


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <div style="font-size:28px; font-weight:800; color:#FFD166;">
                FraudGuard AI
            </div>
            <div style="color:#FFFFFF; font-size:13px; font-weight:600;">
                Credit Card Fraud Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Fraud Detection", "About Model"],
    )

    st.markdown("---")

    st.markdown(
        '<p style="color:#FFD166 !important; font-weight:800;">Model</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#FFFFFF !important; font-weight:600;">Tuned XGBoost</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#FFD166 !important; font-weight:800;">Task</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#FFFFFF !important; font-weight:600;">Binary Classification</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#FFD166 !important; font-weight:800;">Threshold</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#FFFFFF !important; font-weight:600;">0.50</p>',
        unsafe_allow_html=True,
    )


# ================================================================
# FRAUD DETECTION PAGE
# ================================================================
if page == "Fraud Detection":

    st.markdown(
        '<div class="title">Credit Card Fraud Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Analyze a credit card transaction using the trained
            XGBoost fraud detection model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section">Transaction Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        time_value = st.number_input(
            "Transaction Time",
            min_value=0.0,
            value=406.0,
            step=1.0,
        )

    with col2:
        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=149.62,
            step=0.01,
            format="%.2f",
        )

    st.markdown(
        '<div class="section">Transaction Features (V1–V28)</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "V1–V28 are PCA-transformed features from the original credit-card dataset."
    )

    feature_values = {}
    feature_columns = st.columns(4)

    for i in range(1, 29):
        with feature_columns[(i - 1) % 4]:
            feature_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"feature_v{i}",
            )

    st.markdown("---")

    analyze = st.button(
        "Analyze Transaction",
        type="primary",
    )

    if analyze:

        with st.spinner("Loading model and analyzing transaction..."):

            try:
                prediction, probability = make_prediction(
                    time_value,
                    amount_value,
                    feature_values,
                )

                st.markdown(
                    '<div class="section">Prediction Result</div>',
                    unsafe_allow_html=True,
                )

                result_col, probability_col = st.columns([1.4, 1])

                with result_col:

                    if prediction == 1:

                        st.markdown(
                            """
                            <div class="fraud">
                                <div class="fraud-title">
                                    Fraudulent Transaction
                                </div>

                                <p style="color:#FFFFFF !important;">
                                    The model classified this transaction
                                    as potentially fraudulent.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            """
                            <div class="legitimate">
                                <div class="legitimate-title">
                                    Legitimate Transaction
                                </div>

                                <p style="color:#FFFFFF !important;">
                                    The model classified this transaction
                                    as legitimate.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with probability_col:

                    st.metric(
                        "Fraud Probability",
                        f"{probability:.2%}",
                    )

                    st.progress(
                        min(max(probability, 0.0), 1.0)
                    )

                    st.markdown(
                        f"""
                        <div class="card">
                            <b style="color:#FFD166 !important;">
                                Decision Threshold: {THRESHOLD:.0%}
                            </b>

                            <p style="color:#FFFFFF !important;">
                                Probability ≥ {THRESHOLD:.0%} → Fraud<br>
                                Probability &lt; {THRESHOLD:.0%} → Legitimate
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as error:

                st.error("Prediction could not be completed.")
                st.exception(error)


# ================================================================
# ABOUT MODEL PAGE
# ================================================================
else:

    st.markdown(
        '<div class="title">About the Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Machine learning pipeline used for credit card fraud detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model", "Tuned XGBoost")

    with c2:
        st.metric("Problem", "Binary Classification")

    with c3:
        st.metric("Threshold", "0.50")

    st.markdown("---")

    st.markdown(
        '<div class="section">Machine Learning Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">

        <p style="color:#FFD166 !important; font-weight:800;">
            1. Data Preparation
        </p>
        <p style="color:#FFFFFF !important;">
            Duplicate transactions were removed and class imbalance was analyzed.
        </p>

        <p style="color:#FFD166 !important; font-weight:800;">
            2. Feature Engineering
        </p>
        <p style="color:#FFFFFF !important;">
            Hour was extracted from Time and Amount was log-transformed.
        </p>

        <p style="color:#FFD166 !important; font-weight:800;">
            3. Preprocessing
        </p>
        <p style="color:#FFFFFF !important;">
            The training StandardScaler is reused during prediction.
        </p>

        <p style="color:#FFD166 !important; font-weight:800;">
            4. Model Comparison
        </p>
        <p style="color:#FFFFFF !important;">
            Logistic Regression, SMOTE, Decision Tree, Random Forest and XGBoost were evaluated.
        </p>

        <p style="color:#FFD166 !important; font-weight:800;">
            5. Hyperparameter Tuning
        </p>
        <p style="color:#FFFFFF !important;">
            GridSearchCV was used to tune XGBoost using Average Precision.
        </p>

        <p style="color:#FFD166 !important; font-weight:800;">
            6. Deployment
        </p>
        <p style="color:#FFFFFF !important;">
            The saved XGBoost model and scaler are used to generate fraud probabilities and final predictions.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section">Model Features</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#FFFFFF !important;">Time, V1–V28, Amount, Hour, and Amount_log.</p>',
        unsafe_allow_html=True,
    )

    st.info(
        "V1–V28 are PCA-transformed features and do not have direct business "
        "interpretations such as customer age or merchant type."
    )

    st.markdown("---")

    st.markdown(
        """
        <p style="color:#CBD5E1 !important; text-align:center;">
            Credit Card Fraud Detection | Tuned XGBoost Machine Learning Project
        </p>
        """,
        unsafe_allow_html=True,
    )
