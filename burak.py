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
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title = "Miuul Resort Hotel Sunar!", page_icon = ":sunglasses:", layout = "wide")

df = pd.read_csv("hotel_bookings.csv")

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
    st.image(image_v1, caption = 'Miuul Resort Hotel Bodrum', width = 616)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


blue_hotel = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_LYavbtkrBH.json")
man_in_island = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_vdqgavca.json")
front_desk = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_5ohCYt.json")

with st.expander("1 - Amaç Hakkında"):
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

with st.expander("2 - Veri Hakkında"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>MÜŞTERİLER ve REZARVASYONLAR</h2>", unsafe_allow_html = True)
        st.write("---")
        st.dataframe(df)

        col1, col2, col3,col4 = st.columns(4)
        with col1,col2:
            image_v2 = Image.open('columns.jpg')
            st.image(image_v2, caption = 'Miuul Resort Hotel Degişkenler', width = 1200)


with st.expander("3 - Grafikler Hakkında"):
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
            plt.show(block = True)

            st.pyplot()


