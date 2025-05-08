import base64
import streamlit as st

def show_home():

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

    st.title("Formula 1 Dashboard")
    st.markdown("""
    Welcome to the **Formula 1 Dashboard** — your interactive platform for predicting and analysing the sport through data.

    **Formula One (F1)** is the highest class of worldwide racing for open-wheel single-seater formula racing cars sanctioned by the Fédération Internationale de l'Automobile (FIA). 
    
    The FIA Formula One World Championship has been one of the world's premier forms of motorsport since its inaugural running in 1950 and is often considered to be the pinnacle of motorsport.
    
    The word formula in the name refers to the set of rules all participant cars must follow. 
    
    A Formula One season consists of a series of races, known as Grands Prix. Grands Prix take place in multiple countries and continents on either purpose-built circuits or closed roads.
    """)