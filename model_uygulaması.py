
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import streamlit as st
from analysis_functions import *
from hotel_prediction_pipeline import data_preprocessing, df_load
from datetime import datetime

new_model = joblib.load("voting_clf.pkl")
df = df_load()

no_of_adults = st.number_input("Yetişkin Sayısı",step = 1)
no_of_children = st.number_input("Çocuk Sayısı", step = 1)
no_of_weekend_nights = st.number_input("Haftasonu Kaç Akşam Kalacaksınız?",step=1)
no_of_week_nights = st.number_input("Haftaiçi Kaç Akşam Kalacaksınız?",step=1)
type_of_meal_plan = st.selectbox("Yemek Planınız?", ["Just Room", "Breakfast", "Half Board", "Full Board"])
required_car_parking_space = st.selectbox("Araba için Otopark Kullanacak mısınız?", ["Yes", "No"])
if required_car_parking_space == "Yes":
    required_car_parking_space = 1
else:
    required_car_parking_space = 0
room_type_reserved = st.selectbox("Oda Tipini Seçiniz", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                                                                     'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
current_date =datetime(2018,1,1)
arrival_year = st.number_input("Varış Tarihi Yıl?",step = 1)
arrival_month = st.number_input("Varış Tarihi Ay?", step = 1)
arrival_date = st.number_input("Varış Tarihi Gün?",step = 1)
if arrival_year > 0 and arrival_month > 0 and arrival_date > 0 :
    date = datetime(arrival_year, arrival_month, arrival_date)
    lead_time = (date - current_date).days
    st.write("Your lead time is: ", lead_time)

market_segment_type = st.selectbox("Rezervasyonu Nasıl Yapacaksınız", ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.selectbox("Daha Önce Otelimizde Kaldınız mı?", ["Yes", "No"])
if repeated_guest == "Yes":
    repeated_guest = 1
else:
    repeated_guest = 0
no_of_previous_cancellations = st.number_input("Otelimizde Kaç Defa Rezervasyon İptali Yaptınız?",step=1)
no_of_previous_bookings_not_canceled = st.number_input("Otelimizde Daha önce kaç kere kaldınız?", step=1)
if room_type_reserved == "Room_Type 1" and lead_time < 50:
    avg_price_per_room = int(st.selectbox("Type 1 odasını seçtiniz ve gelmenize 50 günden az kaldığı için oda fiyatımız:",["120"]))
elif room_type_reserved == "Room_Type 1" and 50 <= lead_time < 100:
    avg_price_per_room = int(st.selectbox("Type 1 odasını seçtiniz ve gelmenize 50 günden fazla olduğu için oda fiyatımız:",["100"]))
elif room_type_reserved == "Room_Type 1" and lead_time >= 100:
    avg_price_per_room = int(st.selectbox("Type 1 odasını seçtiniz ve gelmenize 100 günden fazla olduğu için oda fiyatımız:",["90"]))
elif room_type_reserved == "Room_Type 2" and lead_time < 50:
    avg_price_per_room = int(st.selectbox("Type 2 odasını seçtiniz ve gelmenize 50 günden az kaldığı için oda fiyatımız:",["110"]))
elif room_type_reserved == "Room_Type 2" and 50 <= lead_time < 100:
    avg_price_per_room = int(st.selectbox("Type 2 odasını seçtiniz ve gelmenize 50 günden fazla olduğu için oda fiyatımız:",["90"]))
elif room_type_reserved == "Room_Type 2" and lead_time >= 100:
    avg_price_per_room = int(st.selectbox("Type 2 odasını seçtiniz ve gelmenize 100 günden fazla olduğu için oda fiyatımız:",["80"]))
elif room_type_reserved == "Room_Type 3" and lead_time < 50:
    avg_price_per_room = int(st.selectbox("Type 3 odasını seçtiniz ve gelmenize 50 günden az kaldığı için oda fiyatımız:",["140"]))
elif room_type_reserved == "Room_Type 3" and 50 <= lead_time < 100:
    avg_price_per_room = int(st.selectbox("Type 3 odasını seçtiniz ve gelmenize 50 günden fazla olduğu için oda fiyatımız:",["120"]))
elif room_type_reserved == "Room_Type 3" and lead_time >= 100:
    avg_price_per_room = int(st.selectbox("Type 3 odasını seçtiniz ve gelmenize 100 günden fazla olduğu için oda fiyatımız:",["100"]))
else:
    avg_price_per_room = int(
        st.selectbox("Kampanyalı bir odamızdan kullanmadığınız için oda fiyatımız:", ["200"]))


no_of_special_requests = 0


if st.button("Onayla"):
    data = {'no_of_adults': [no_of_adults],
            'no_of_children': [no_of_children],
            'no_of_weekend_nights': [no_of_weekend_nights],
            'no_of_week_nights': [no_of_week_nights],
            'type_of_meal_plan': [type_of_meal_plan],
            'required_car_parking_space': [required_car_parking_space],
            'room_type_reserved': [room_type_reserved],
            'lead_time': [lead_time],
            'arrival_year': [arrival_year],
            'arrival_month': [arrival_month],
            'arrival_date': [arrival_date],
            'market_segment_type': [market_segment_type],
            'repeated_guest': [repeated_guest],
            'no_of_previous_cancellations': [no_of_previous_cancellations],
            'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
            'avg_price_per_room': [avg_price_per_room],
            'no_of_special_requests': [no_of_special_requests],
            'booking_status': 0 }
    df1 = pd.DataFrame(data)
    df = pd.concat([df,df1],ignore_index=True)
    X, y = data_preprocessing(df)
    sonuc = new_model.predict(X.tail(1))
    a = sonuc[0]
    if a == 1:
        a = "Bu müşteri rezervasyonun 89% olasılıkla iptal edecektir"
    else:
        a = "Bu müşteri rezervasyonunu 89% olasılıkla iptal etmeyecektir"
    st.write("Tahminleme Sonucu:", a)
# prediction_proba