import base64
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def show_pit_stop_prediction():
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

    # Load the datasets locally
    pit_stops_df = pd.read_csv('f1db-races-pit-stops.csv')
    race_results_df = pd.read_csv('f1db-races-race-results.csv')
    races_df = pd.read_csv('f1db-races.csv')

    # Rename 'id' column to 'raceId' in races_df
    races_df.rename(columns={'id': 'raceId'}, inplace=True)

    # Merging the datasets
    merged_df = pd.merge(race_results_df, pit_stops_df, on=['raceId', 'driverId'], how='inner')
    final_df = pd.merge(merged_df, races_df[['raceId', 'circuitId', 'year']], on='raceId', how='inner')

    # Filter the data to only include races from 2013 or later
    final_df = final_df[final_df['year'] >= 2013]

    # Filter active drivers, constructors, and circuits post-2013
    active_drivers = final_df['driverId'].unique()
    active_constructors = final_df['constructorId_y'].unique()
    active_circuits = final_df['circuitId'].unique()

    # ------------------- PIT STOP LAP PREDICTION -------------------
    # Preprocessing for lap prediction
    pit_stop_lap_data_encoded = final_df[['driverId', 'constructorId_y', 'circuitId', 'lap', 'stop']]

    # Encoding categorical variables
    pit_stop_lap_data_encoded = pd.get_dummies(pit_stop_lap_data_encoded, columns=['driverId', 'constructorId_y', 'circuitId'], drop_first=True)

    # Features and target for lap prediction
    X_lap = pit_stop_lap_data_encoded.drop(columns=['lap', 'stop'])
    y_lap = pit_stop_lap_data_encoded['lap']

    # Normalizing the target variable (lap)
    scaler = MinMaxScaler()
    y_lap_scaled = scaler.fit_transform(y_lap.values.reshape(-1, 1))

    # Splitting the dataset for training and testing
    X_train_lap, X_test_lap, y_train_lap, y_test_lap = train_test_split(X_lap, y_lap_scaled, test_size=0.2, random_state=42)

    # Pre-train the RandomForest model for Lap Prediction
    model_lap = RandomForestRegressor(n_estimators=100, random_state=42)
    model_lap.fit(X_train_lap, y_train_lap)

    # ------------------- PIT STOP COUNT PREDICTION -------------------
    # Preprocessing for pit stop count prediction
    pit_stop_data_encoded = final_df[['driverId', 'constructorId_y', 'circuitId', 'pitStops', 'stop']]

    # Encoding categorical variables
    pit_stop_data_encoded = pd.get_dummies(pit_stop_data_encoded, columns=['driverId', 'constructorId_y', 'circuitId'], drop_first=True)

    # Features and target for pit stop count prediction
    X_count = pit_stop_data_encoded.drop(columns=['pitStops', 'stop'])
    y_count = pit_stop_data_encoded['pitStops']

    # Splitting the dataset for training and testing
    X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(X_count, y_count, test_size=0.2, random_state=42)

    # Pre-train the RandomForest model for Pit Stop Count Prediction
    model_count = RandomForestRegressor(n_estimators=100, random_state=42)
    model_count.fit(X_train_count, y_train_count)

    # Save the models after training (for later use)
    joblib.dump(model_lap, 'lap_prediction_model.pkl')
    joblib.dump(model_count, 'pit_stop_count_model.pkl')

    # Evaluate the models
    y_pred_count = model_count.predict(X_test_count)
    mse_count = mean_squared_error(y_test_count, y_pred_count)

    y_pred_lap = model_lap.predict(X_test_lap)
    y_pred_lap_original = scaler.inverse_transform(y_pred_lap.reshape(-1, 1))
    mse_lap = mean_squared_error(scaler.inverse_transform(y_test_lap.reshape(-1, 1)), y_pred_lap_original)

    # Create Streamlit UI
    st.title('Formula 1 Pit Stop Predictor')
    st.write('Input driver, constructor, circuit, and year to predict the number of pit stops and lap for pit stop.')

    # Filtered dropdowns for driver, constructor, and circuit
    driver = st.selectbox('Select Driver', active_drivers)
    constructor = st.selectbox('Select Constructor', active_constructors)
    circuit = st.selectbox('Select Circuit', active_circuits)

    # Year input field with restriction to post-2013
    year = st.number_input('Enter Year', min_value=2013, max_value=2025, value=2020)

    # Prepare the input data for both models
    input_data = pd.DataFrame([[driver, constructor, circuit, year]], columns=['driverId', 'constructorId', 'circuitId', 'year'])

    # Preprocess the input data (encode categorical features)
    input_data_encoded = pd.get_dummies(input_data, columns=['driverId', 'constructorId', 'circuitId'], drop_first=True)

    # Ensure that input_data_encoded has the same columns as X_train_count and X_train_lap
    input_data_encoded = input_data_encoded.reindex(columns=X_count.columns, fill_value=0)

    # Load the pre-trained models
    model_lap = joblib.load('lap_prediction_model.pkl')
    model_count = joblib.load('pit_stop_count_model.pkl')

    # Predict the number of pit stops
    if st.button('Predict Pit Stops'):
        predicted_pit_stops = model_count.predict(input_data_encoded)
        st.write(f'The predicted number of pit stops is: {int(predicted_pit_stops[0])}')
        st.write(f'Mean Squared Error (Pit Stops Model): {mse_count:.2f}')

        # Predict the lap for the first pit stop (if applicable)
        if predicted_pit_stops[0] > 0:
            predicted_lap = model_lap.predict(input_data_encoded)
            predicted_lap_original = scaler.inverse_transform(predicted_lap.reshape(-1, 1))
            st.write(f'The predicted lap for the first pit stop is: {int(predicted_lap_original[0][0])}')
        else:
            st.write("No pit stops predicted for this race.")

        st.write(f'Mean Squared Error (Lap Prediction Model): {mse_lap:.2f}')
