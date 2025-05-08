# main.py
import streamlit as st
from home import show_home
from race_position_prediction import show_race_position_prediction
from driver_performance_analysis import show_driver_performance_analysis
from pit_stop_prediction import show_pit_stop_prediction

# Sidebar navigation
st.sidebar.title("Formula 1 Dashboard")
page = st.sidebar.radio("Go to", [
    "Home",
    "Race Position Predictor",
    "Driver Performance Analysis",
    "Pit Stop Predictor" ,
])
# Route to selected page
try:
    if page == "Home":
        show_home()
    elif page == "Race Position Predictor":
        show_race_position_prediction()
    elif page == "Driver Performance Analysis":
        show_driver_performance_analysis() 
    elif page == "Pit Stop Predictor":
        show_pit_stop_prediction()

except Exception as e:
    st.error("Oops! Something went wrong while loading this page. Please try adjusting your filters or refresh the app.")
    st.exception(e)