import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/disaster.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Spaceship Titanic",
        page_icon=image,

    )

    st.write(
        """
        # Классификация пассажиров космического корабля "Титаник"
        Определяем, кто из пассажиров переместился в другое измерение из-за столкновения с аномалией, а кто – нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    home = st.sidebar.selectbox("Родная планета", ("Европа", "Земля", "Марс"))
    destination = st.sidebar.selectbox("Планета назначения", (
    "TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"))
    
    is_cryo_sleep = st.sidebar.checkbox("Находились в криогенном сне")
    is_vip = st.sidebar.checkbox("Заплатили за VIP-обслуживании")

    age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)

    translatetion = {
        "Европа": "Europa",
        "Земля": "Earth",
        "Марс": "Mars",
        "TRAPPIST-1e": "TRAPPIST-1e",
        "PSO J318.5-22": "PSO J318.5-22",
        "55 Cancri e": "55 Cancri e"
    }

    data = {
        "HomePlanet": translatetion[home],
        "CryoSleep": is_cryo_sleep,
        "Age": age,
        "VIP": is_vip,
        "Destination": translatetion[destination]
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()