import streamlit as st
import pandas as pd
import numpy as np
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

logistic_model = joblib.load("models/logistic.pkl")
knn_model      = joblib.load("models/knn.pkl")
svc_model      = joblib.load("models/svc.pkl")
rf_model       = joblib.load("models/rf.pkl")
gb_model       = joblib.load("models/gb.pkl")
ann_model      = joblib.load("models/ann.pkl")

# =====================================
# LOAD SCALERS (only for KNN, SVC, LR, ANN)
# =====================================

logistic_scaler = joblib.load("models/logistic_scaler.pkl")
knn_scaler      = joblib.load("models/knn_scaler.pkl")
svc_scaler      = joblib.load("models/svc_scaler.pkl")
ann_scaler      = joblib.load("models/ann_scaler.pkl")

# =====================================
# LOAD FEATURE COLUMNS
# =====================================

feature_columns = joblib.load("models/feature_columns.pkl")

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
    Predict whether an NBA player is a **good** (FT% >= 75%) or **below-average**
    (FT% < 75%) free-throw shooter using machine learning and basketball statistics.
    The model is trained on 15,020 player-seasons (1975-2025) with no exclusions —
    every player who attempted at least one free throw per game receives a prediction.
    """)

    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Rows", "15,020")

    with col2:
        st.metric("Features Used", "39")

    with col3:
        st.metric("ML Models", "6")

    st.write("---")

    st.success("🏆 Best Model: Gradient Boosting (78.06% Accuracy)")

    st.write("---")

    st.subheader("📌 Machine Learning Models")

    st.write("""
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    - Random Forest
    - Gradient Boosting
    - Artificial Neural Network (ANN)
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
    Enter basketball statistics below to predict whether the player is a
    **good free-throw shooter** (FT% >= 75%) or a **below-average free-throw shooter**
    (FT% < 75%).
    """)

    st.write("---")

    model_choice = st.selectbox(
        "🤖 Choose Machine Learning Model",
        [
            "Logistic Regression",
            "KNN",
            "SVC",
            "Random Forest",
            "Gradient Boosting",
            "ANN",
        ]
    )

    st.write("---")

    # =====================================
    # FREE-THROW HISTORY
    # =====================================

    st.subheader("📊 Free-Throw History")

    col1, col2, col3 = st.columns(3)

    with col1:
        career_ft_pct = st.slider("Career FT% (prior seasons)", 0.0, 1.0, 0.75, step=0.01)

    with col2:
        prev_ft_pct = st.slider("Previous Season FT%", 0.0, 1.0, 0.75, step=0.01)

    with col3:
        career_seasons = st.slider("Career Seasons Played", 0, 20, 3)

    st.write("---")

    # =====================================
    # PLAYER INFO
    # =====================================

    st.subheader("👤 Player Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age    = st.slider("Age", 18, 45, 25)
        pos    = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"])

    with col2:
        games   = st.slider("Games Played", 0, 82, 50)
        gs      = st.slider("Games Started", 0, 82, 30)

    with col3:
        minutes = st.slider("Minutes Per Game", 0.0, 48.0, 20.0)
        height  = st.slider("Height (inches)", 66, 90, 78)
        weight  = st.slider("Weight (lbs)", 150, 320, 215)

    st.write("---")

    # =====================================
    # SHOOTING STATS
    # =====================================

    st.subheader("🎯 Shooting Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        fg_pct  = st.slider("Field Goal %", 0.0, 1.0, 0.45)
        fga     = st.slider("FG Attempts per game", 0.0, 30.0, 10.0)
        threep_pct = st.slider("3-Point %", 0.0, 1.0, 0.35)

    with col2:
        twop_pct = st.slider("2-Point %", 0.0, 1.0, 0.50)
        efg      = st.slider("Effective FG %", 0.0, 1.0, 0.50)
        threep_a = st.slider("3PA per game", 0.0, 15.0, 4.0)

    with col3:
        twop_a   = st.slider("2PA per game", 0.0, 25.0, 6.0)

    st.write("---")

    # =====================================
    # PERFORMANCE
    # =====================================

    st.subheader("📊 Player Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        pts = st.slider("Points Per Game", 0.0, 40.0, 12.0)
        ast = st.slider("Assists Per Game", 0.0, 15.0, 3.0)
        tov = st.slider("Turnovers Per Game", 0.0, 10.0, 2.0)

    with col2:
        trb = st.slider("Total Rebounds Per Game", 0.0, 20.0, 5.0)
        orb = st.slider("Offensive Rebounds", 0.0, 8.0, 1.0)
        drb = st.slider("Defensive Rebounds", 0.0, 15.0, 4.0)

    with col3:
        stl = st.slider("Steals Per Game", 0.0, 5.0, 1.0)
        blk = st.slider("Blocks Per Game", 0.0, 5.0, 0.5)
        pf  = st.slider("Personal Fouls", 0.0, 6.0, 2.0)

    st.write("---")

    # =====================================
    # BUILD INPUT ROW AND ENGINEER FEATURES
    # =====================================

    # Compute derived fields
    fg       = fg_pct * fga
    threep   = threep_pct * threep_a
    twop     = twop_pct * twop_a

    # Engineered ratios - denominator 0 -> ratio 0 (matches notebook 02)
    ast_to_tov     = ast / tov if tov > 0 else 0.0
    pts_per_fga    = pts / fga if fga > 0 else 0.0
    threep_rate    = threep_a / fga if fga > 0 else 0.0
    reb_per_min    = trb / minutes if minutes > 0 else 0.0
    stocks_per_min = (stl + blk) / minutes if minutes > 0 else 0.0
    scoring_load   = fga / minutes if minutes > 0 else 0.0

    # One-hot encode position
    pos_C  = 1 if pos == 'C'  else 0
    pos_PF = 1 if pos == 'PF' else 0
    pos_PG = 1 if pos == 'PG' else 0
    pos_SF = 1 if pos == 'SF' else 0
    pos_SG = 1 if pos == 'SG' else 0

    raw_input = {
        'Age':            age,
        'G':              games,
        'GS':             gs,
        'MP':             minutes,
        'FG':             fg,
        'FGA':            fga,
        'FG%':            fg_pct,
        'ORB':            orb,
        'DRB':            drb,
        'TRB':            trb,
        'AST':            ast,
        'STL':            stl,
        'BLK':            blk,
        'TOV':            tov,
        'PF':             pf,
        'PTS':            pts,
        '3P':             threep,
        '3PA':            threep_a,
        '3P%':            threep_pct,
        '2P':             twop,
        '2PA':            twop_a,
        '2P%':            twop_pct,
        'eFG%':           efg,
        'career_ft_pct':  career_ft_pct,
        'prev_ft_pct':    prev_ft_pct,
        'career_seasons': career_seasons,
        'height_in':      height,
        'weight_lb':      weight,
        'ast_to_tov':     ast_to_tov,
        'pts_per_fga':    pts_per_fga,
        'threep_rate':    threep_rate,
        'reb_per_min':    reb_per_min,
        'stocks_per_min': stocks_per_min,
        'scoring_load':   scoring_load,
        'Pos_C':          pos_C,
        'Pos_PF':         pos_PF,
        'Pos_PG':         pos_PG,
        'Pos_SF':         pos_SF,
        'Pos_SG':         pos_SG,
    }

    # Build dataframe and reindex to exactly match training column order
    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # =====================================
    # APPLY SCALERS WHERE NEEDED
    # =====================================

    input_knn      = knn_scaler.transform(input_df)
    input_svc      = svc_scaler.transform(input_df)
    input_logistic = logistic_scaler.transform(input_df)
    input_ann      = ann_scaler.transform(input_df)
    # RF and GB receive raw (unscaled) input

    # =====================================
    # PREDICTION
    # =====================================

    if st.button("🏀 Analyze Player"):

        if model_choice == "Logistic Regression":
            prediction = logistic_model.predict(input_logistic)

        elif model_choice == "KNN":
            prediction = knn_model.predict(input_knn)

        elif model_choice == "SVC":
            prediction = svc_model.predict(input_svc)

        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_df)

        elif model_choice == "Gradient Boosting":
            prediction = gb_model.predict(input_df)

        else:  # ANN
            prediction = ann_model.predict(input_ann)

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
                🏀 GOOD FREE-THROW SHOOTER (predicted FT% >= 75%)
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
                🏀 BELOW-AVERAGE FREE-THROW SHOOTER (predicted FT% < 75%)
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
            "KNN",
            "ANN",
            "Random Forest",
            "Gradient Boosting",
            "SVC",
            "Logistic Regression",
        ],
        "Accuracy (%)":  [71.64, 75.43, 76.96, 78.06, 77.23, 74.23],
        "Precision (%)": [73.35, 76.71, 79.09, 79.65, 78.31, 75.25],
        "Recall (%)":    [74.52, 78.22, 77.91, 79.70, 79.95, 77.85],
        "F1-Score (%)":  [73.93, 77.46, 78.50, 79.68, 79.12, 76.53],
    }

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    st.write("---")

    st.subheader("📈 Accuracy Comparison")

    st.bar_chart(df.set_index("Model")[["Accuracy (%)"]])

    st.write("---")

    st.subheader("📌 Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.image("images/lr_confusion_matrix.png",   caption="Logistic Regression")
        st.image("images/knn_confusion_matrix.png",  caption="KNN")
        st.image("images/rf_confusion_matrix.png",   caption="Random Forest")

    with col2:
        st.image("images/svc_confusion_matrix.png",  caption="SVC")
        st.image("images/gb_confusion_matrix.png",   caption="Gradient Boosting")
        st.image("images/ann_confusion_matrix.png",  caption="ANN")
