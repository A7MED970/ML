import streamlit as st
import pandas as pd
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="NBA Free Throw Predictor",
    page_icon="🏀",
    layout="wide"
)

# =====================================
# CUSTOM STYLE
# =====================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom, #0F172A, #111827);
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(to right, #FF512F, #DD2476);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #DD2476, #FF512F);
}

[data-testid="stMetric"] {
    background-color: #1F2937;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODELS
# =====================================

logistic_model = joblib.load("logistic.pkl")
knn_model = joblib.load("knn.pkl")
svc_model = joblib.load("svc.pkl")
rf_model = joblib.load("rf.pkl")
gb_model = joblib.load("gb.pkl")

# =====================================
# LOAD SCALERS
# =====================================

logistic_scaler = joblib.load("logistic_scaler.pkl")
knn_scaler = joblib.load("knn_scaler.pkl")
svc_scaler = joblib.load("svc_scaler.pkl")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🏀 NBA ML Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Model Comparison"
    ]
)

# =====================================
# HOME PAGE
# =====================================

if page == "Home":

    st.title("🏀 NBA Free Throw Prediction System")

    st.write("""
    Predict whether an NBA player performs
    above or below the league-average free throw percentage
    using machine learning and basketball statistics.
    """)

    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Rows", "26,292")

    with col2:
        st.metric("Features Used", "25")

    with col3:
        st.metric("ML Models", "5")

    st.write("---")

    st.success("🏆 Best Model: Logistic Regression (90.72% Accuracy)")

    st.write("---")

    st.subheader("📌 Machine Learning Models")

    st.write("""
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    - Random Forest
    - Gradient Boosting
    """)

    st.write("---")

    st.subheader("👨‍💻 Team Members")

    st.write("""
    - Habibatallah Mahdi
    - Sama Abdelsadek
    - Mohamed Allam
    - Shaza Hossam
    - Ahmed Shaban
    """)

# =====================================
# PREDICTION PAGE
# =====================================

elif page == "Prediction":

    st.title("🏀 NBA Player Analyzer")

    st.write("""
    Enter basketball statistics below to predict whether
    the player performs above or below the NBA league-average
    free throw percentage.
    """)

    st.write("---")

    model_choice = st.selectbox(
        "🤖 Choose Machine Learning Model",
        [
            "Logistic Regression",
            "KNN",
            "SVC",
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    st.write("---")

    # =====================================
    # PLAYER INFO
    # =====================================

    st.subheader("👤 Player Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 45, 25)

    with col2:
        games = st.slider("Games Played", 0, 82, 50)

    with col3:
        minutes = st.slider("Minutes Played", 0.0, 48.0, 20.0)

    st.write("---")

    # =====================================
    # SHOOTING STATS
    # =====================================

    st.subheader("🎯 Shooting Statistics")

    col1, col2 = st.columns(2)

    with col1:

        fg_percent = st.slider(
            "Field Goal Percentage",
            0.0,
            1.0,
            0.45
        )

        threep_percent = st.slider(
            "3-Point Percentage",
            0.0,
            1.0,
            0.35
        )

        twop_percent = st.slider(
            "2-Point Percentage",
            0.0,
            1.0,
            0.50
        )

    with col2:

        efg = st.slider(
            "Effective FG Percentage",
            0.0,
            1.0,
            0.50
        )

        ft_percent = st.slider(
            "Free Throw Percentage",
            0.0,
            1.0,
            0.75
        )

        ft_attempts = st.slider(
            "Free Throw Attempts",
            0.0,
            15.0,
            4.0
        )

    st.write("---")

    # =====================================
    # PERFORMANCE
    # =====================================

    st.subheader("📊 Player Performance")

    col1, col2 = st.columns(2)

    with col1:

        pts = st.slider(
            "Points Per Game",
            0.0,
            40.0,
            12.0
        )

        ast = st.slider(
            "Assists",
            0.0,
            15.0,
            3.0
        )

    with col2:

        trb = st.slider(
            "Rebounds",
            0.0,
            20.0,
            5.0
        )

        tov = st.slider(
            "Turnovers",
            0.0,
            10.0,
            2.0
        )

    st.write("---")

    # =====================================
    # INPUT DATA
    # =====================================

    input_data = pd.DataFrame({

        'Age': [age],
        'G': [games],
        'GS': [40],
        'MP': [minutes],

        'FG': [5.0],
        'FGA': [10.0],
        'FG%': [fg_percent],

        'FT': [ft_percent * ft_attempts],
        'FTA': [ft_attempts],

        'ORB': [1.0],
        'DRB': [4.0],
        'TRB': [trb],

        'AST': [ast],
        'STL': [1.0],
        'BLK': [0.5],

        'TOV': [tov],
        'PF': [2.0],

        'PTS': [pts],

        '3P': [1.0],
        '3PA': [4.0],
        '3P%': [threep_percent],

        '2P': [4.0],
        '2PA': [8.0],
        '2P%': [twop_percent],

        'eFG%': [efg]
    })

    # =====================================
    # APPLY SCALERS
    # =====================================

    logistic_scaled = logistic_scaler.transform(input_data)

    knn_scaled = knn_scaler.transform(input_data)

    svc_scaled = svc_scaler.transform(input_data)

    # =====================================
    # PREDICTION
    # =====================================

    if st.button("🏀 Analyze Player"):

        if model_choice == "Logistic Regression":

            prediction = logistic_model.predict(logistic_scaled)

        elif model_choice == "KNN":

            prediction = knn_model.predict(knn_scaled)

        elif model_choice == "SVC":

            prediction = svc_model.predict(svc_scaled)

        elif model_choice == "Random Forest":

            prediction = rf_model.predict(logistic_scaled)

        else:

            prediction = gb_model.predict(logistic_scaled)

        st.write("---")

        if prediction[0] == 1:

            st.markdown(
                '''
                <div style="
                    background: linear-gradient(to right, #11998e, #38ef7d);
                    padding:30px;
                    border-radius:15px;
                    text-align:center;
                    font-size:28px;
                    color:white;
                    font-weight:bold;
                ">
                🏀 ABOVE LEAGUE-AVERAGE FREE THROW SHOOTER
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '''
                <div style="
                    background: linear-gradient(to right, #cb2d3e, #ef473a);
                    padding:30px;
                    border-radius:15px;
                    text-align:center;
                    font-size:28px;
                    color:white;
                    font-weight:bold;
                ">
                🏀 BELOW LEAGUE-AVERAGE FREE THROW SHOOTER
                </div>
                ''',
                unsafe_allow_html=True
            )

# =====================================
# MODEL COMPARISON PAGE
# =====================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    data = {
        "Model": [
            "Logistic Regression",
            "KNN",
            "SVC",
            "Random Forest",
            "Gradient Boosting"
        ],

        "Accuracy (%)": [
            90.72,
            71.48,
            82.75,
            77.16,
            85.17
        ]
    }

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    st.write("---")

    st.subheader("📈 Accuracy Comparison")

    st.bar_chart(df.set_index("Model"))

    st.write("---")

    st.subheader("📌 Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.image("images/lr_confusion_matrix.png", caption="Logistic Regression")
        st.image("images/knn_confusion_matrix.png", caption="KNN")
        st.image("images/rf_confusion_matrix.png", caption="Random Forest")

    with col2:
        st.image("images/svc_confusion_matrix.png", caption="SVC")
        st.image("images/gb_confusion_matrix.png", caption="Gradient Boosting")