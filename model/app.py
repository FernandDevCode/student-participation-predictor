import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the preprocessor and model
preprocessor = joblib.load('preprocessor.joblib')
best_model = joblib.load('best_random_forest_model.joblib')

# Get feature names from the preprocessor after fitting
# Assuming preprocessor is a ColumnTransformer with named transformers
def get_feature_names(preprocessor, numerical_features, categorical_features):
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    return numerical_features + list(ohe_feature_names)

# Define the features to be used for the model (must match the order used during training)
features_for_model = [
    'Opportunity Name',
    'Opportunity Category',
    'Gender',
    'Country',
    'Student_Age',
    'Opportunity_Duration',
    'Opportunity_Quarter',
    'Apply_to_Start_Days',
    'Major_Cleaned',
    'Sectors_of_Intended_Major',
    'Signup_to_Start_Days'
]

numerical_features = ['Student_Age', 'Opportunity_Duration', 'Opportunity_Quarter', 'Apply_to_Start_Days', 'Signup_to_Start_Days']
categorical_features = [
    'Opportunity Name',
    'Opportunity Category',
    'Gender',
    'Country',
    'Major_Cleaned',
    'Sectors_of_Intended_Major'
]

full_feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)

# Streamlit App Title
st.title('Student Participation Prediction Dashboard')

st.markdown("Enter the student and opportunity details to predict participation status.")

# Input fields for features
with st.sidebar:
    st.header('Student & Opportunity Details')

    # Numerical Inputs
    student_age = st.slider('Student Age', min_value=15, max_value=60, value=25)
    opportunity_duration = st.slider('Opportunity Duration (days)', min_value=0, max_value=700, value=100)
    opportunity_quarter = st.selectbox('Opportunity Quarter', options=[1, 2, 3, 4], index=3) # Assuming 1-4, default to Q4 (index 3 for 0-based)
    apply_to_start_days = st.slider('Days from Application to Start', min_value=-500, max_value=500, value=30)
    signup_to_start_days = st.slider('Days from Signup to Start', min_value=-500, max_value=500, value=45)

    # Categorical Inputs (using unique values from original df if available, or a representative list)
    # For a real app, you'd load these unique values from your data or a config file.
    # For this example, we'll use a sample of values used during training.

    # You would typically load these from your original dataset or store them during preprocessing
    # For demonstration, we'll use a hardcoded small list. For a complete solution, you'd need to adapt.
    unique_opportunity_names = ['Career Essentials: Getting Started with Your Professional Journey', 'Data Visualization', 'Project Management', 'Business Consulting', 'CPR/AED Certification', 'Digital Marketing', 'Data Visualization Associate', 'Digital Strategy Virtual Internship', 'Health Care Management'] # Sample
    opportunity_name = st.selectbox('Opportunity Name', options=unique_opportunity_names)

    unique_opportunity_categories = ['Course', 'Internship', 'Event', 'Competition'] # Sample
    opportunity_category = st.selectbox('Opportunity Category', options=unique_opportunity_categories)

    unique_genders = ['Male', 'Female', 'Other'] # Sample
    gender = st.selectbox('Gender', options=unique_genders)

    unique_countries = ['United States', 'India', 'Pakistan', 'Canada', 'United Kingdom'] # Sample
    country = st.selectbox('Country', options=unique_countries)

    unique_majors = ['computer science', 'information systems', 'radiology', 'business administration', 'electrical engineering'] # Sample
    major_cleaned = st.selectbox('Major Cleaned', options=unique_majors)

    unique_sectors = ['Computer Science & IT', 'Health & Medical Sciences', 'Business & Finance', 'Engineering & Technology'] # Sample
    sectors_of_intended_major = st.selectbox('Sectors of Intended Major', options=unique_sectors)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Student_Age': [student_age],
    'Opportunity_Duration': [opportunity_duration],
    'Opportunity_Quarter': [opportunity_quarter],
    'Apply_to_Start_Days': [apply_to_start_days],
    'Signup_to_Start_Days': [signup_to_start_days],
    'Opportunity Name': [opportunity_name],
    'Opportunity Category': [opportunity_category],
    'Gender': [gender],
    'Country': [country],
    'Major_Cleaned': [major_cleaned],
    'Sectors_of_Intended_Major': [sectors_of_intended_major]
})

# Ensure the input_data has all the features the preprocessor expects, even if they are not directly from input fields.
# This involves creating dummy columns for all possible one-hot encoded categories.
# This is a critical step for robustness in deployment.

# Create a dummy DataFrame with all possible feature columns as expected by the preprocessor
# This is a more robust way to handle unseen categories or missing categories in input

def create_empty_df_with_all_features(preprocessor, numerical_features, categorical_features):
    # Create an empty DataFrame
    dummy_data = pd.DataFrame(columns=numerical_features + categorical_features)
    # Fit transform a dummy row to get all OHE column names, then discard
    # A more robust way is to store the full list of OHE column names during training
    # For this example, we'll recreate them based on the `get_feature_names` function behavior.
    # In a real-world scenario, you would save and load `full_feature_names` directly.
    return pd.DataFrame(columns=get_feature_names(preprocessor, numerical_features, categorical_features))

# Placeholder for a full list of columns expected by the model after preprocessing
# In a real application, this list `full_feature_names` would be saved during training
# and loaded here, or derived more robustly.

# Preprocess the input data
# Create a DataFrame with the same columns as X_train had *before* preprocessing (i.e., 'features_for_model')
# Then fill it with user input.
processed_input = preprocessor.transform(input_data[features_for_model])

# Convert to DataFrame with correct column names
processed_input_df = pd.DataFrame(processed_input, columns=full_feature_names)

# Make prediction
if st.button('Predict Participation'):
    prediction = best_model.predict(processed_input_df)
    prediction_proba = best_model.predict_proba(processed_input_df)[0]

    st.subheader('Prediction Result:')
    if prediction[0] == 1.0:
        st.success(f"The student is likely to Participate (Probability: {prediction_proba[1]:.2f})")
    else:
        st.error(f"The student is likely Not to Participate (Probability: {prediction_proba[0]:.2f})")

    st.subheader('Prediction Probabilities:')
    st.write(f"Not Participate (Class 0): {prediction_proba[0]:.2f}")
    st.write(f"Participate (Class 1): {prediction_proba[1]:.2f}")

