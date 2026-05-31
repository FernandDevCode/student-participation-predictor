import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# Configuration des chemins
# =========================
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "model" / "best_random_forest_model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.joblib"

# Adapter ce chemin selon l'emplacement réel du fichier
CSV_PATH = BASE_DIR / "data" / "raw" / "RIT_Opportunity_Wise_Data_weeek2_cleaned.csv"

# =========================
# Chargement du modèle
# =========================
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# =========================
# Chargement des options
# =========================
@st.cache_data
def load_options():
    try:
        df = pd.read_csv(CSV_PATH)

        options = {
            "Opportunity Name": sorted(df["Opportunity Name"].dropna().unique().tolist()),
            "Opportunity Category": sorted(df["Opportunity Category"].dropna().unique().tolist()),
            "Gender": sorted(df["Gender"].dropna().unique().tolist()),
            "Country": sorted(df["Country"].dropna().unique().tolist()),
            "Major_Cleaned": sorted(df["Major_Cleaned"].dropna().unique().tolist()),
            "Sectors_of_Intended_Major": sorted(
                df["Sectors_of_Intended_Major"].dropna().unique().tolist()
            ),
        }

        return options

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV : {e}")
        st.stop()

options = load_options()

# =========================
# Interface Streamlit
# =========================
st.set_page_config(
    page_title="Student Participation Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Participation Prediction Dashboard")

st.write(
    "Utilisez les paramètres dans la barre latérale puis cliquez sur "
    "**Predict** pour obtenir une prédiction."
)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Input Features")

    Student_Age = st.slider(
        "Student Age",
        min_value=15,
        max_value=60,
        value=25
    )

    Opportunity_Duration = st.slider(
        "Opportunity Duration (days)",
        min_value=0,
        max_value=700,
        value=30
    )

    Opportunity_Quarter = st.selectbox(
        "Opportunity Quarter",
        [1, 2, 3, 4]
    )

    Apply_to_Start_Days = st.slider(
        "Apply to Start Days",
        min_value=-500,
        max_value=500,
        value=0
    )

    Signup_to_Start_Days = st.slider(
        "Signup to Start Days",
        min_value=-500,
        max_value=500,
        value=0
    )

    Opportunity_Name = st.selectbox(
        "Opportunity Name",
        options["Opportunity Name"]
    )

    Opportunity_Category = st.selectbox(
        "Opportunity Category",
        options["Opportunity Category"]
    )

    Gender = st.selectbox(
        "Gender",
        options["Gender"]
    )

    Country = st.selectbox(
        "Country",
        options["Country"]
    )

    Major_Cleaned = st.selectbox(
        "Major Cleaned",
        options["Major_Cleaned"]
    )

    Sectors_of_Intended_Major = st.selectbox(
        "Sectors of Intended Major",
        options["Sectors_of_Intended_Major"]
    )

# =========================
# Données utilisateur
# =========================
input_dict = {
    "Student_Age": Student_Age,
    "Opportunity_Duration": Opportunity_Duration,
    "Opportunity_Quarter": Opportunity_Quarter,
    "Apply_to_Start_Days": Apply_to_Start_Days,
    "Signup_to_Start_Days": Signup_to_Start_Days,
    "Opportunity Name": Opportunity_Name,
    "Opportunity Category": Opportunity_Category,
    "Gender": Gender,
    "Country": Country,
    "Major_Cleaned": Major_Cleaned,
    "Sectors_of_Intended_Major": Sectors_of_Intended_Major
}

# =========================
# Prédiction
# =========================
if st.button("Predict", type="primary"):

    try:
        input_df = pd.DataFrame([input_dict])

        input_transformed = preprocessor.transform(input_df)

        prediction = model.predict(input_transformed)[0]
        prediction_proba = model.predict_proba(input_transformed)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Likely to Participate")
        else:
            st.error("❌ Likely Not to Participate")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Participate (Class 0)",
                f"{prediction_proba[0] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Participate (Class 1)",
                f"{prediction_proba[1] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")