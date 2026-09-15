import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)


WORKCLASSES = [
    "Private",
    "Self-emp-not-inc",
    "Self-emp-inc",
    "Federal-gov",
    "Local-gov",
    "State-gov",
    "Without-pay",
    "Never-worked",
]

EDUCATION_NUM_MAPPING = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

MARITAL_STATUSES = [
    "Never-married",
    "Married-civ-spouse",
    "Divorced",
    "Separated",
    "Widowed",
    "Married-spouse-absent",
    "Married-AF-spouse",
]

OCCUPATIONS = [
    "Tech-support",
    "Craft-repair",
    "Other-service",
    "Sales",
    "Exec-managerial",
    "Prof-specialty",
    "Handlers-cleaners",
    "Machine-op-inspct",
    "Adm-clerical",
    "Farming-fishing",
    "Transport-moving",
    "Priv-house-serv",
    "Protective-serv",
    "Armed-Forces",
]

RELATIONSHIPS = [
    "Wife",
    "Own-child",
    "Husband",
    "Not-in-family",
    "Other-relative",
    "Unmarried",
]

RACES = [
    "White",
    "Asian-Pac-Islander",
    "Amer-Indian-Eskimo",
    "Other",
    "Black",
]

SEXES = [
    "Female",
    "Male",
]

COUNTRIES = [
    "United-States",
    "Mexico",
    "Philippines",
    "Germany",
    "Canada",
    "Puerto-Rico",
    "El-Salvador",
    "India",
    "Cuba",
    "England",
    "Jamaica",
    "South",
    "China",
    "Italy",
    "Dominican-Republic",
    "Vietnam",
    "Guatemala",
    "Japan",
    "Poland",
    "Columbia",
    "Taiwan",
    "Haiti",
    "Iran",
    "Portugal",
    "Nicaragua",
    "Peru",
    "France",
    "Greece",
    "Ecuador",
    "Ireland",
    "Hong",
    "Cambodia",
    "Trinadad&Tobago",
    "Laos",
    "Thailand",
    "Yugoslavia",
    "Outlying-US(Guam-USVI-etc)",
    "Hungary",
    "Honduras",
    "Scotland",
    "Holand-Netherlands",
]

st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="centered",
)

st.title("Adult Income Predictor")

st.write(
    "Enter the person's information and predict "
    "whether their annual income is above $50K."
)



with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=100,
        value=35,
    )

    workclass = st.selectbox(
        "Workclass",
        WORKCLASSES,
    )

    education = st.selectbox(
        "Education",
        list(EDUCATION_NUM_MAPPING.keys()),
        index=list(
            EDUCATION_NUM_MAPPING.keys()
        ).index("Bachelors"),
    )

    marital_status = st.selectbox(
        "Marital status",
        MARITAL_STATUSES,
    )

    occupation = st.selectbox(
        "Occupation",
        OCCUPATIONS,
    )

    relationship = st.selectbox(
        "Relationship",
        RELATIONSHIPS,
    )

    race = st.selectbox(
        "Race",
        RACES,
    )

    sex = st.selectbox(
        "Sex",
        SEXES,
    )

    hours_per_week = st.number_input(
        "Hours per week",
        min_value=1,
        max_value=168,
        value=40,
    )

    native_country = st.selectbox(
        "Native country",
        COUNTRIES,
    )

    st.subheader("Additional census features")

    fnlwgt = st.number_input(
        "Final weight (fnlwgt)",
        min_value=0,
        value=180000,
        help=(
            "Census sampling weight associated "
            "with this observation."
        ),
    )

    capital_gain = st.number_input(
        "Capital gain",
        min_value=0,
        value=0,
    )

    capital_loss = st.number_input(
        "Capital loss",
        min_value=0,
        value=0,
    )

    submitted = st.form_submit_button(
        "Predict income"
    )


if submitted:

    education_num = (
        EDUCATION_NUM_MAPPING[education]
    )

    payload = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        prediction = result["prediction"]

        probability = result.get(
            "probability_over_50k",
            result.get("probability"),
        )

        if prediction == ">50K":
            st.success(
                "Predicted income: **>50K**"
            )
        else:
            st.info(
                "Predicted income: **<=50K**"
            )

        if probability is not None:
            st.metric(
                "Probability of income >50K",
                f"{probability:.1%}",
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        st.error(
            "The prediction API did not respond in time."
        )

    except requests.exceptions.HTTPError:
        st.error(
            f"API error: {response.text}"
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )