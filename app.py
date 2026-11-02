import os

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Starbucks Beverage Predictor",
    page_icon="☕",
    layout="wide",
)


@st.cache_resource
def load_model():
    # Accept either filename so the app works with the uploaded file name
    # or the cleaner filename recommended for the GitHub repository.
    possible_files = [
        "starbucks_decision_tree_model.joblib",
        "starbucks_decision_tree_model(1).joblib",
    ]

    for file_name in possible_files:
        if os.path.exists(file_name):
            return joblib.load(file_name)

    raise FileNotFoundError(
        "Model file not found. Add "
        "'starbucks_decision_tree_model.joblib' "
        "or 'starbucks_decision_tree_model(1).joblib' "
        "to the same GitHub folder as app.py."
    )


features = [
    "Calories",
    "Total Fat (g)",
    "Trans Fat (g)",
    "Saturated Fat (g)",
    "Sodium (mg)",
    "Total Carbohydrates (g)",
    "Cholesterol (mg)",
    "Dietary Fibre (g)",
    "Sugars (g)",
    "Protein (g)",
    "Vitamin A (% DV)",
    "Vitamin C (% DV)",
    "Calcium (% DV)",
    "Iron (% DV)",
    "Caffeine (mg)",
]

# Median values used by the original notebook to fill missing feature values.
default_values = {
    "Calories": 185,
    "Total Fat (g)": 2.5,
    "Trans Fat (g)": 0.5,
    "Saturated Fat (g)": 0,
    "Sodium (mg)": 5,
    "Total Carbohydrates (g)": 125,
    "Cholesterol (mg)": 34,
    "Dietary Fibre (g)": 0,
    "Sugars (g)": 32,
    "Protein (g)": 6,
    "Vitamin A (% DV)": 8,
    "Vitamin C (% DV)": 0,
    "Calcium (% DV)": 20,
    "Iron (% DV)": 2,
    "Caffeine (mg)": 75,
}


st.title("☕ Starbucks Beverage Category Predictor")
st.write(
    "Enter the nutritional information below. The Decision Tree model "
    "will predict the Starbucks beverage category."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.subheader("Nutritional Information")
st.caption("The starting values are defaults. Change them to match your beverage.")

inputs = {}
columns = st.columns(3)

for index, feature in enumerate(features):
    with columns[index % 3]:
        inputs[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=float(default_values[feature]),
            step=1.0,
        )

if st.button("Predict Beverage Category", type="primary", use_container_width=True):
    input_data = pd.DataFrame([inputs], columns=features)

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.subheader("Predicted Beverage Category")
        st.markdown(f"## {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            results = (
                pd.DataFrame(
                    {
                        "Beverage category": model.classes_,
                        "Model probability": probabilities,
                    }
                )
                .sort_values("Model probability", ascending=False)
                .reset_index(drop=True)
            )
            results["Model probability"] = (
                results["Model probability"] * 100
            ).round(1).astype(str) + "%"

            with st.expander("View model probabilities"):
                st.dataframe(results, use_container_width=True, hide_index=True)

        with st.expander("View entered nutritional information"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
