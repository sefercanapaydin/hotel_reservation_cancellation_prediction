# Booking_ID	Unique identifier of the booking.
# no_of_adults	The number of adults.
# no_of_children	The number of children.
# no_of_weekend_nights	Number of weekend nights (Saturday or Sunday).
# no_of_week_nights	Number of week nights (Monday to Friday).
# type_of_meal_plan	Type of meal plan included in the booking.
# required_car_parking_space	Whether a car parking space is required.
# room_type_reserved	The type of room reserved.
# lead_time	Number of days before the arrival date the booking was made.
# arrival_year	Year of arrival.
# arrival_month	Month of arrival.
# arrival_date	Date of the month for arrival.
# market_segment_type	How the booking was made.
# repeated_guest	Whether the guest has previously stayed at the hotel.
# no_of_previous_cancellations	Number of previous cancellations.
# no_of_previous_bookings_not_canceled	Number of previous bookings that were canceled.
# avg_price_per_room	Average price per day of the booking.
# no_of_special_requests	Count of special requests made as part of the booking.
# booking_status	Whether the booking was cancelled or not.

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import datetime as dt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from datetime import date
from analysis_functions import *

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title = "Miuul Resort Hotel Sunar!", page_icon = ":sunglasses:", layout = "wide")


def df_load():
    global df
    df = pd.read_csv("hotel_bookings.csv")
    df["booking_status"] = np.where(df["booking_status"] == "Not_Canceled", "0", "1").astype("int64")
    df.drop("Booking_ID", axis = 1, inplace = True)
    df.loc[df['type_of_meal_plan'] == 'Not Selected', 'type_of_meal_plan'] = 'Just Room'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 1', 'type_of_meal_plan'] = 'Breakfast'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 2', 'type_of_meal_plan'] = 'Half Board'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 3', 'type_of_meal_plan'] = 'Full Board'


df_load()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 8, car_th = 20)
cat_cols = [col for col in cat_cols if "booking_status" not in col]

for col in num_cols:
    df[col].fillna(df.groupby("booking_status")[col].transform("median"), inplace = True)

for col in cat_cols:
    df[col].fillna(df.groupby("booking_status")[col].transform(lambda x: x.fillna(x.mode().iloc[0])), inplace = True)

for col in df.columns:
    if col not in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'avg_price_per_room']:
        df[col] = df[col].astype("int64")


def time_features(df):
    temp = df.rename(columns = {
        'arrival_year': 'year',
        'arrival_month': 'month',
        'arrival_date': 'day'
    })
    df['date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors = 'coerce')
    df['date'].fillna('2018-02-28', inplace = True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['booking_date'] = df['date'] - pd.to_timedelta(df['lead_time'], unit = 'D')
    df['booking_year'] = df['booking_date'].dt.year
    df['booking_month'] = df['booking_date'].dt.month
    df['booking_week'] = df['booking_date'].dt.isocalendar().week.astype(float)
    df['booking_day'] = df['booking_date'].dt.day
    df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    df['booking_quarter'] = df['booking_date'].dt.quarter
    df['booking_dayofyear'] = df['booking_date'].dt.dayofyear
    df.drop(['booking_date', 'date', 'arrival_month', 'arrival_date', 'arrival_year'], axis = 1, inplace = True)
    return df


df = time_features(df)

df["no_of_guest_NEW"] = df["no_of_adults"].astype('int64') + df["no_of_children"].astype('int64')
df["no_of_nights_NEW"] = df["no_of_weekend_nights"].astype('int64') + df["no_of_week_nights"].astype('int64')

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th = 8, car_th = 20)
cat_cols = [col for col in cat_cols if "booking_status" not in col]
for col in cat_cols:
    df[col] = df[col].astype("object")

df = rare_encoder(df, 0.01)

df3 = df.copy()

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1501426026826-31c667bdf23d");
background-size: cover%;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html = True)
st.title(":orange[Tatil Zamanı!] :sunglasses:")

st.title("**Bodrum, Miuul Resort Hotel'e Hoşgeldiniz** 🌅")
st.subheader("*:blue[Vahit Keskin]: Son zamanlarda otelimizde yer ayırtıp birkaç gün kala iptal eden müşterilerimiz var. Bu durum otel kaynaklarının kullanımı açısından bizi "
             "oldukça zor durumda bırakıyor. Hangi müşterilerimizin rezervasyonunu iptal edeceğini önceden :red[tahminleyip] aksiyonlar almanızı ve otelimizin verimini "
             "artırmanızı bekliyorum.*")

col1, col2, col3 = st.columns(3)

with col2:
    image_v1 = Image.open('miuul_hotel.jpg')
    st.image(image_v1, caption = 'Miuul Resort Hotel Bodrum', width = 666)


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


blue_hotel = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_LYavbtkrBH.json")
man_in_island = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_vdqgavca.json")
front_desk = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5ohCYt.json")
recommended = load_lottieurl("https://assets1.lottiefiles.com/temp/lf20_uCb4xZ.json")
request = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_xme5jqhd.json")
telephone = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_E0NT3v.json")
room = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_yvM0Ga2Wah.json")

with st.expander("1 - AMAÇ HAKKINDA"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>AMACIMIZ</h2>", unsafe_allow_html = True)
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("**Rezervasyon iptali, gelir üzerinde önemli bir etkiye sahiptir ve otel sektöründe yönetim kararlarını etkilemektedir. İptal etmenin etkisini azaltmak ve "
                     "sorunu çözmek için makine öğrenimine dayalı bir iptal model sistemi geliştirilmiştir. Veri bilimi araçları, insan yargısı ve davranışlarıyla birleştirilerek,"
                     "modelin tahmin analizinin, rezervasyon iptali tahminleri konusunda öngörülere nasıl katkıda bulunabileceği gösterilecektir.**")
        with col3:
            st_lottie(man_in_island, height = 400)

with st.expander("2 - VERİ HAKKINDA"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>MÜŞTERİLER ve REZARVASYONLAR</h2>", unsafe_allow_html = True)
        st.write("---")
        st.dataframe(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1, col2:
            image_v2 = Image.open('columns.jpg')
            st.image(image_v2, caption = 'Miuul Resort Hotel Degişkenler', width = 1200)

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col3:
            image_v3 = Image.open("shape.JPG")
            st.image(image_v3, caption = 'Gözlem Sayısı')
        with col4:
            image_v4 = Image.open("missing.JPG")
            st.image(image_v4, caption = 'Eksik Değer Sayısı')
        with col5:
            image_v5 = Image.open("dropna.JPG")
            st.image(image_v5, caption = "Eksik Değerlerden Sonra Kalan Gözlem Sayısı")

ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

with st.expander("3 - GRAFİKLER HAKKINDA"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>GRAFİKLER ve YORUMLAMALAR</h2>", unsafe_allow_html = True)
        st.write("---")

        col1, col2 = st.columns(2)
        with col1:
            bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 443]

            df["lead_time_bin"] = pd.cut(df["lead_time"], bins = bins)
            lead_time_counts = df.groupby(["lead_time_bin", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["lead_time_bin"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "lead_time_bin", y = "count", hue = "booking_status", data = lead_time_counts, palette = 'viridis')
            sns.pointplot(x = "lead_time_bin", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Lead Time")
            plt.xlabel("Lead Time")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        with col2:
            no_of_special_requests = df.groupby(["no_of_special_requests", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["no_of_special_requests"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "no_of_special_requests", y = "count", hue = "booking_status", data = no_of_special_requests, palette = 'viridis')
            sns.pointplot(x = "no_of_special_requests", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Special Requests")
            plt.xlabel("Number of Special Requests")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        col3, col4 = st.columns(2)
        with col3:
            type_of_meal_plan = df.groupby(["type_of_meal_plan", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["type_of_meal_plan"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "type_of_meal_plan", y = "count", hue = "booking_status", data = type_of_meal_plan, palette = 'viridis')
            sns.pointplot(x = "type_of_meal_plan", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Meal Plan ")
            plt.xlabel("Meal Plans")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        with col4:
            market_segment_type = df.groupby(["market_segment_type", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["market_segment_type"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "market_segment_type", y = "count", hue = "booking_status", data = market_segment_type, palette = 'viridis')
            sns.pointplot(x = "market_segment_type", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Market Segment Type ")
            plt.xlabel("Market  Types")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        col5, col6 = st.columns(2)
        with col5:
            month = df.groupby(["month", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["month"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "month", y = "count", hue = "booking_status", data = month, palette = 'viridis')
            sns.pointplot(x = "month", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Months ")
            plt.xlabel("Months")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        with col6:
            bins = [0, 50, 100, 150, 200, 250]

            df["avg_price_per_room_bin"] = pd.cut(df["avg_price_per_room"], bins = bins)
            avg_price_per_room = df.groupby(["avg_price_per_room_bin", "booking_status"]).size().reset_index(name = "count")
            total_count = df.groupby(["avg_price_per_room_bin"]).size().reset_index(name = "total")

            plt.figure(figsize = (12, 6))
            sns.barplot(x = "avg_price_per_room_bin", y = "count", hue = "booking_status", data = avg_price_per_room, palette = 'viridis')
            sns.pointplot(x = "avg_price_per_room_bin", y = "total", data = total_count, color = "k")

            plt.title("Count of Bookings by Average Room Price ")
            plt.xlabel("Room Price")
            plt.ylabel("Count of Bookings")

            plt.tight_layout()
            st.pyplot()

        col7, col8 = st.columns(2)
        with col7:
            plt.figure(figsize = (15, 8))
            sns.scatterplot(x = 'avg_price_per_room', y = 'lead_time', data = df, hue = 'booking_status', palette = 'viridis')
            st.pyplot()

with st.expander("4 - ÖNERİLER HAKKINDA"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>FAYDALI NELER YAPILABİLİR?</h2>", unsafe_allow_html = True)
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader("Lead Time süresi 5 aydan az olanlarda iptal olma oranı %20’iken, 5 aydan daha fazla zaman geçen rezervasyonların %70’i iptal edilmektedir. Bu iptal "
                         "oranının önüne geçmek için şu yollar izlenebilir;")
            st.write(
                """1.	5 ay ve daha önceden yapılan rezervasyon fiyatları minimum tutulur ancak iptal durumunda para iadesi yapılmaz. Böylelikle caydırıcı bir önlem alınmış 
                olunur ancak bu yüzden bizi tercih etmeyen müşteriler olabilir. 2.	Rezervasyon yapılırken iptal durumuna karşı “iptal etme durumunda para iadesi” adı altında 
                ayrı bir sigorta satılabilir. Örneğin toplam rezervasyon bedelinin %10’u kadar sigorta bedeli alınabilir. 3.	Rezervasyonun iptal tarihi baz alınarak para 
                iadesi kademeli olarak azalan bir şekilde yapılabilir. Böylece varış süresi ile rezervasyon iptali arasındaki süre artılır, boş odalara yeni rezervasyonlar 
                yapılması sağlanabilir."""
            )
        with col2:
            st_lottie(recommended, height = 400)

        col3, col4 = st.columns(2)
        with col3:
            st_lottie(request, height = 400)

        with col4:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader("Müşterilerin özel talepleri artıkça, rezervasyonun iptal edilme olasılığı azalmaktadır. Hiç talepte bulunmayan müşterilerin iptal oranı 57%'iken, "
                         "1 ya da daha fazla özel talep tercih eden müşterilerin iptal oranı 20%'dir. Dolayısıyla iptal oranını azaltmak için şunlar yapılabilir;")
            st.write(
                """
            1. Özel istekte bulunmayan müşterilerin, özel istekte bulunmalarını sağlayacak, "ilk talebe özel % indirim" ya da müşterilere otelin herhangi 
            bir aktivitesinden ya da imkanından faydalanma fırsatı sağlanabilir.
            2. En az bir özel talepte bulunan müşterilerin talep sayısını artırmak adına belirli bir talep sayısından sonra ek talep hakkı verilebilir.
                """
            )

        col5, col6 = st.columns(2)
        with col6:
            st_lottie(telephone, height = 400)

        with col5:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader("Otel rezervasyonlarını online platformlar üzerinden gerçekleştiren müşterilerin iptal oranı 36%’iken, otele gelerek ya da telefon yolu ile otel ile "
                         "doğrudan iletişim kuran müşterilerinm iptal oranı 26%’dır. Online platformlarda birçok aracı tatil sitesi olması otel ile müşteri arasında zayıf bir bağ "
                         "kurulmasına neden oluyor olabilir. Müşterilerin doğrudan otel ile daha kuvvetli bağ kurmalarını sağlayacak offline kanalların tercih edilebilirliğini "
                         "artıracak kampanyalar düzenlenebilir.")

        col7, col8 = st.columns(2)
        with col7:
            st_lottie(room, height = 400)

        with col8:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader("En fazla iptal 43% oranıyla, 100 ile 150 dolar arasında rezervasyon yapılan odalarda gerçekleşmektedir. Bu odaları rezerve eden müşterilerimizin "
                         "iptal oranlarını düşürmek adına şu adımlar izlenebilir;")
            st.write(
                """
                        1.	Bu fiyat aralığında odaları rezerve eden müşterilere ek hizmetler sunulabilir. (Spa, hamam, masaj vb.) Erken check-in/geç check-out ayrıcalıkları sunulabilir. Bu müşterilere zaman konusunda esneklik sağlar.
                        2.	Rezervasyon süresi yaklaştıkça bu odaları tercih eden müşterilere kendilerini özel hissettirecek mailler gönderilebilir.
                        3.	Bir defaya mahsus müşteri aranarak özel bir isteği olup olmadığı sorularak, müşteri üzerinde samimiyet duygusu yaratılabilir ve otel ile arasında bağ kurulabilir.
                        """
            )

from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import streamlit as st
from analysis_functions import *
from hotel_prediction_pipeline import data_preprocessing, df_load

new_model = joblib.load("voting_clf.pkl")
df = df_load()

no_of_adults = st.number_input("Yetişkin Sayısı", step = 1)
no_of_children = st.number_input("Çocuk Sayısı", step = 1)
no_of_weekend_nights = st.number_input("Haftasonu Kaç Akşam Kalacaksınız?", step = 1)
no_of_week_nights = st.number_input("Haftaiçi Kaç Akşam Kalacaksınız?", step = 1)
type_of_meal_plan = st.selectbox("Yemek Planınız?", ["Just Room", "Breakfast", "Half Board", "Full Board"])
required_car_parking_space = st.selectbox("Araba için Otopark Kullanacak mısınız?", ["Yes", "No"])
if required_car_parking_space == "Yes":
    required_car_parking_space = 1
else:
    required_car_parking_space = 0
room_type_reserved = st.selectbox("Oda Tipini Seçiniz", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4',
                                                         'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = 500
arrival_year = st.number_input("Varış Tarihi Yıl?", step = 1)
arrival_month = st.number_input("Varış Tarihi Ay?", step = 1)
arrival_date = st.number_input("Varış Tarihi Gün?", step = 1)
market_segment_type = st.selectbox("Rezervasyonu Nasıl Yapacaksınız", ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.selectbox("Daha Önce Otelimizde Kaldınız mı?", ["Yes", "No"])
if repeated_guest == "Yes":
    repeated_guest = 1
else:
    repeated_guest = 0
no_of_previous_cancellations = st.number_input("Otelimizde Kaç Defa Rezervasyon İptali Yaptınız?", step = 1)
no_of_previous_bookings_not_canceled = st.number_input("Otelimizde Daha önce kaç kere kaldınız?", step = 1)
avg_price_per_room = 50
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
            'booking_status': 0}
    df1 = pd.DataFrame(data)
    df = pd.concat([df, df1], ignore_index = True)
    X, y = data_preprocessing(df)
    sonuc = new_model.predict(X.tail(1))
    a = sonuc[0]
    if a == 1:
        a = "Bu müşteri rezervasyonun 0.8925 olasılıkla iptal etmiştir"
    else:
        a = "Bu müşteri rezervasyonunu 0.8925 olasılıkla iptal etmeyecektir"
    st.write("Tahminleme Sonucu:", a)
