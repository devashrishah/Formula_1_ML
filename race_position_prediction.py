import base64
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import datetime

@st.cache_data
def load_and_prepare_data():
    raceresult_df = pd.read_csv("f1db-races-race-results.csv")
    circuits_df = pd.read_csv("f1db-races.csv")
    merge_data = pd.merge(raceresult_df, circuits_df, left_on='raceId', right_on='id')
    merge_data = merge_data[merge_data['year_x'] >= 2013]

    df = merge_data[['positionNumber', 'driverId', 'gridPositionNumber', 'circuitId', 'year_x']].copy()
    df.columns = ['positionNumber', 'driverId', 'gridPosition', 'circuitId', 'year']
    df.dropna(inplace=True)

    def categorize_position(pos):
        try:
            pos = int(pos)
            if 1 <= pos <= 3:
                return 'On Podium'
            elif 4 <= pos <= 10:
                return 'In Points'
            elif 11 <= pos <= 20:
                return 'Finished - No Points'
            else:
                return 'Finished - No Points'
        except:
            return 'Did Not Finish'

    df['resultCategory'] = df['positionNumber'].apply(categorize_position)

    le_driver = LabelEncoder()
    le_circuit = LabelEncoder()
    le_result = LabelEncoder()

    df['driverId'] = le_driver.fit_transform(df['driverId'])
    df['circuitId'] = le_circuit.fit_transform(df['circuitId'])
    df['resultCategoryEncoded'] = le_result.fit_transform(df['resultCategory'])

    return df, le_driver, le_circuit, le_result

def show_race_position_prediction():
    background_image_path = "download.png" 
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpg;base64,{base64.b64encode(open(background_image_path, "rb").read()).decode()}');
                background-size: cover;
                background-position: center center;
                background-repeat: no-repeat;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Formula 1 Race Position Predictor')
    
    df, le_driver, le_circuit, le_result = load_and_prepare_data()

    X = df[['driverId', 'gridPosition', 'circuitId', 'year']]
    y = df['resultCategoryEncoded']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=0)
    xgb_clf.fit(X_train, y_train)

    st.write("### Race Position Predictor")

    driver_input_str = st.selectbox("Select Driver", sorted(le_driver.classes_))
    circuit_input_str = st.selectbox("Select Circuit", sorted(le_circuit.classes_))
    grid_position = st.number_input("Grid Position", min_value=1, max_value=30, step=1)
    year_input = st.slider("Race Year", min_value=2013, max_value=datetime.datetime.now().year, value=2023)

    # Encode inputs
    driver_encoded = le_driver.transform([driver_input_str])[0]
    circuit_encoded = le_circuit.transform([circuit_input_str])[0]

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            'driverId': driver_encoded,
            'gridPosition': grid_position,
            'circuitId': circuit_encoded,
            'year': year_input
        }])
        prediction = xgb_clf.predict(input_df)[0]
        label = le_result.inverse_transform([prediction])[0]
        st.success(f"🏁 Predicted Result Category: **{label}**")
