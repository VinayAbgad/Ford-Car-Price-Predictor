# ============================================================
# FORD CAR PRICE PREDICTION STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_model_objects():

    model = joblib.load("LR_ford_car.pkl")

    scaler = joblib.load("scaler.pkl")

    encoded_columns = joblib.load("columns.pkl")

    return model, scaler, encoded_columns


# Load saved machine learning objects
model, scaler, encoded_columns = load_model_objects()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🚗 About This App")

    st.write(
        "This application predicts the estimated selling price "
        "of a Ford car using a trained Linear Regression model."
    )

    st.markdown("---")

    st.subheader("Technologies Used")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🤖 Scikit-learn")
    st.write("🎈 Streamlit")

    st.markdown("---")

    st.info(
        "Enter the car details and click the prediction button "
        "to estimate the selling price."
    )


# ============================================================
# TITLE AND DESCRIPTION
# ============================================================

st.title("🚗 Ford Car Price Predictor")

st.write(
    "Enter the car details below to predict its estimated selling price."
)


st.markdown("---")


# ============================================================
# CAR DETAILS
# ============================================================

st.subheader("🚘 Car Details")


# Create two columns for better layout
col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1: MODEL AND NUMERICAL FEATURES
# ------------------------------------------------------------

with col1:

    model_name = st.text_input(
        "Car Model",
        placeholder="Example: Fiesta"
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2026,
        value=2018,
        step=1
    )

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=500000,
        value=50000,
        step=1000
    )

    tax = st.number_input(
        "Road Tax",
        min_value=0,
        max_value=1000,
        value=150,
        step=10
    )


# ------------------------------------------------------------
# COLUMN 2: CATEGORICAL AND NUMERICAL FEATURES
# ------------------------------------------------------------

with col2:

    transmission = st.selectbox(
        "Transmission",
        [
            "Automatic",
            "Manual",
            "Semi-Auto"
        ]
    )

    fuelType = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "Hybrid",
            "Electric",
            "Other"
        ]
    )

    mpg = st.number_input(
        "Miles Per Gallon (MPG)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=0.1
    )

    engineSize = st.number_input(
        "Engine Size",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.1
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")


predict_button = st.button(
    "🔍 Predict Car Price",
    use_container_width=True
)


# ============================================================
# COMPLETE PREPROCESSING AND PREDICTION PIPELINE
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # STEP 1: VALIDATE USER INPUT
        # ----------------------------------------------------

        if model_name.strip() == "":

            st.warning(
                "⚠️ Please enter a car model name."
            )

        else:

            # ------------------------------------------------
            # STEP 2: CREATE INPUT DATAFRAME
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "model": [model_name],

                "year": [year],

                "transmission": [transmission],

                "mileage": [mileage],

                "fuelType": [fuelType],

                "tax": [tax],

                "mpg": [mpg],

                "engineSize": [engineSize]

            })

            # ------------------------------------------------
            # STEP 3: ONE-HOT ENCODING
            # ------------------------------------------------

            input_encoded = pd.get_dummies(

                input_data,

                columns=[

                    "model",

                    "transmission",

                    "fuelType"

                ]

            )

            # ------------------------------------------------
            # STEP 4: ALIGN COLUMNS
            # ------------------------------------------------

            input_encoded = input_encoded.reindex(

                columns=encoded_columns,

                fill_value=0

            )

            # ------------------------------------------------
            # STEP 5: SCALE NUMERICAL FEATURES
            # ------------------------------------------------

            numeric_columns = [

                "year",

                "mileage",

                "tax",

                "mpg",

                "engineSize"

            ]

            input_encoded[numeric_columns] = scaler.transform(

                input_encoded[numeric_columns]

            )

            # ------------------------------------------------
            # STEP 6: MAKE PREDICTION
            # ------------------------------------------------

            predicted_price = model.predict(

                input_encoded

            )[0]

            # ------------------------------------------------
            # STEP 7: DISPLAY RESULT
            # ------------------------------------------------

            st.success(
                "✅ Prediction completed successfully!"
            )

            st.metric(
                label="💰 Estimated Ford Car Price",
                value=f"£{predicted_price:,.2f}"
            )

            # Display the processed input data
            with st.expander("View Processed Input Data"):

                st.dataframe(
                    input_encoded,
                    use_container_width=True
                )

    except Exception as error:

        st.error(
            f"❌ An error occurred while making the prediction: "
            f"{error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Ford Car Price Predictor | "
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
