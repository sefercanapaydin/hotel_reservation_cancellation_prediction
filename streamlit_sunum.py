
from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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

st.title("**Miuul Resort Hotel'e Hoşgeldiniz** 🌅")


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

st.subheader("*:blue[Vahit Keskin]: Son zamanlarda otelimizdeki iptal olan rezervasyon sayıları artmıştır. Bu durum otel kaynaklarının kullanımı açısından bizi "
             "oldukça zor durumda bırakıyor. Hangi müşterilerimizin rezervasyonunu iptal edeceğini önceden :red[tahminleyip] aksiyonlar almanızı ve otelimizin verimini "
             "arttırmanızı bekliyorum.*")
col1, col2, col3, col4,col5, col6 = st.columns(6)
with col2:
    image_vk = Image.open('Vahit_Keskin.jpg')
    st.image(image_vk, caption = 'Miuul Resort Hotel Müdürü',width=1200)




with st.expander("1 - AMAÇ HAKKINDA"):

    with st.container():
        st.markdown("<h2 style='text-align: center;'>AMACIMIZ</h2>", unsafe_allow_html = True)
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            st.write("##")
            st.write("##")
            st.write("**Rezervasyon iptali, gelir üzerinde önemli bir etkiye sahiptir ve otel sektöründe yönetim kararlarını etkilemektedir. İptal etmenin etkisini azaltmak ve "
                     "sorunu çözmek için makine öğrenimine dayalı bir iptal model sistemi geliştirilmiştir. Veri bilimi araçları, insan yargısı ve davranışlarıyla birleştirilerek,"
                     "modelin tahmin analizinin, rezervasyon iptali tahminleri konusunda öngörülere nasıl katkıda bulunabileceği gösterilecektir.**" )
            st.write("**Ayrıca elimizdeki verisetine keşifçi veri analizi uygulayarak rezervasyonların iptal olma sebepleri saptanıp, bunları engellemek için otele tavsiyeler verilecektir.**" )
        with col3:
            st_lottie(man_in_island, height = 400)
        col1, col2, col3, col4,col5, col6 = st.columns(6)
        with col2:
            image_res = Image.open('Resepsiyon.jpeg')
            st.image(image_res, caption='Miuul Resort Hotel Çalışanları', width=1200)

with st.expander("2 - VERİ HAKKINDA"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>MÜŞTERİLER ve REZARVASYONLAR</h2>", unsafe_allow_html = True)
        st.write("---")
        st.dataframe(df)

        col1, col2, col3, col4, col5,col6 = st.columns(6)
        with col1, col2:
            image_v2 = Image.open('columns.jpg')
            st.image(image_v2, caption = 'Miuul resort hotel degişkenler', width = 1200)

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            image_v3 = Image.open("shape.JPG")
            st.image(image_v3, caption = 'Gözlem sayısı', width = 400)
        with col3:
            image_v4 = Image.open("missing.JPG")
            st.image(image_v4, caption = 'Eksik değer sayısı',width = 500)
        with col6:
            image_v5 = Image.open("dropna.JPG")
            st.image(image_v5, caption = "Eğer eksik değerler silinir ise kalan gözlem sayısı")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sns.countplot(x=df["booking_status"], data=df)
            plt.show(block=True)
            st.pyplot()
        with col3:
            st.write("**Değişkenlerin tiplerinin belirlenmesi**")
            st.write("**Eksik değerlerin doldurulması**")
            st.write("**Anlamlı yeni değişkenlerin oluşturulması**")
            st.write("**Nadir değerlerin birleştirilmesi**")

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
            ort1 = round(df.loc[df["lead_time"] <= 150, "booking_status"].mean(),2)
            st.write("Varış zamanları 5 aydan kısa olan rezervasyonların iptal olma durumlarının ortalaması",
                     f"<span>{ort1}</span>", unsafe_allow_html=True)
            ort2 = round(df.loc[df["lead_time"] > 150, "booking_status"].mean(), 2 )
            st.write("Varış zamanları 5 aydan fazla olan rezervasyonların iptal olma durumlarının ortalaması",
                     f"<span>{ort2}</span>", unsafe_allow_html=True)
            df.drop("lead_time_bin",axis=1,inplace=True)
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
            ort1 = round(df.loc[df["no_of_special_requests"] == 0, "booking_status"].mean(),2)
            st.write("Hiç özel isteği olmayan rezervasyonların iptal olma durumlarının ortalaması:",
                     f"<span>{ort1}</span>", unsafe_allow_html=True)
            ort2 = round(df.loc[df["no_of_special_requests"] != 0, "booking_status"].mean(),2)
            st.write("Bir ya da daha fazla özel isteği olan rezervasyonların iptal olma durumlarının ortalaması",
                     f"<span>{ort2}</span>", unsafe_allow_html=True)

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
            ort1 = round(df.loc[df["type_of_meal_plan"] == 'Breakfast', "booking_status"].mean(), 2)
            st.write("Rezervasyon tipi sadece kahvaltı olan rezervasyonların iptal olma durumlarının ortalaması",
                     f"<span>{ort1}</span>", unsafe_allow_html=True )
            ort2 = round(df.loc[df["type_of_meal_plan"] != 'Breakfast', "booking_status"].mean(), 2)
            st.write("Rezervasyon tipi kahvaltı dışındaki olan rezervasyonların iptal olma durumlarının ortalaması",
                     f"<span>{ort2}</span>", unsafe_allow_html=True )

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

            ort1 = round(df.loc[df["market_segment_type"] == 'Online', "booking_status"].mean(),2)
            st.write('Rezervasyonun alındığı platform online ise iptal olma durumlarının ortalaması:',
                     f"<span>{ort1}</span>", unsafe_allow_html=True)
            ort2 = round(df.loc[df["market_segment_type"] != 'Online', "booking_status"].mean(), 2)
            st.write('Rezervasyonun alındığı platform online değil ise iptal olma durumlarının ortalaması:',
                     f"<span>{ort2}</span>", unsafe_allow_html=True)


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
            df.drop("avg_price_per_room_bin", axis=1, inplace=True)

            st.write(df.groupby("booking_status")[['avg_price_per_room']].mean())
        col7, col8 = st.columns(2)
        with col7:
            plt.figure(figsize = (15, 8))
            sns.scatterplot(x = 'avg_price_per_room', y = 'lead_time', data = df, hue = 'booking_status', palette = 'viridis')
            st.pyplot()

with st.expander("4 - KEŞİFÇİ VERİ ANALİZİNE DAYALI ÖNERİLER"):
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

with st.expander("5 -MAKINE ÖĞRENMESİ MODELLEME"):
    with st.container():
        st.markdown("<h2 style='text-align: center;'>MODELE SOKMADAN ÖNCEKİ YAPILAN DÜZENLEMELER</h2>", unsafe_allow_html = True)
        st.write("--------------------------------------")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Korelasyon Grafiği**")
            fig = plt.gcf()
            fig.set_size_inches(10, 8)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            fig = sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                              cmap='RdBu')
            plt.show(block=True)
            st.pyplot()

        with col2:
            st.write("**Aykırı değer analizi**")
            fig = plt.figure(figsize=(30, 150))
            for index, col in enumerate(num_cols):
                plt.subplot(26, 3, index + 1)
                sns.boxplot(x=col, data=df, color='navy')
                plt.ylabel('COUNT', size=25, color="black")
                plt.xlabel(col, fontsize=25, color="black")
                plt.xticks(size=20, color="black", rotation=45)
                plt.yticks(size=20, color="black")
            fig.tight_layout(pad=1.0)
            st.pyplot()

            for col in num_cols:
                replace_with_thresholds(df, col)
            drop_list = high_correlated_cols(df)
            df.drop(drop_list, axis=1, inplace=True)
            cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
            cat_cols = [col for col in cat_cols if "booking_status" not in col]
        col1, col2 = st.columns(2)
        with col1:
            st.write("Encoding işlemleri")
            df = one_hot_encoder(df, cat_cols, drop_first=True)
            st.write("Scaling işlemleri")
            X_scaled = StandardScaler().fit_transform(df[num_cols])
            df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


        y = df["booking_status"]
        X = df.drop(["booking_status"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

        st.markdown("<h2 style='text-align: center;'>ÖRNEK MODELLER</h2>", unsafe_allow_html = True)
        st.write("-----------------")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**XGBOOST - MODEL**")
            xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False).fit(X_train, y_train)
            y_pred = xgboost_model.predict(X_test)
            acc_score = accuracy_score(y_pred, y_test)
            st.write("Accuracy Score:", f"<span>{acc_score}</span>", unsafe_allow_html=True)
            plot_importance(xgboost_model, X, num=len(X))
        with col2:
            st.write("**Random Forest - MODEL**")
            rf_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            acc_score = accuracy_score(y_pred, y_test)
            st.write("Accuracy Score:", f"<span>{acc_score}</span>", unsafe_allow_html=True)
            plot_importance(rf_model, X, num=len(X))
        with col3:
            st.write("**CatBoost - MODEL**")
            catboost_model = CatBoostClassifier(random_state=17, verbose=False).fit(X_train, y_train)
            y_pred = catboost_model.predict(X_test)
            acc_score = accuracy_score(y_pred, y_test)
            st.write("Accuracy Score:", f"<span>{acc_score}</span>", unsafe_allow_html=True)
            plot_importance(catboost_model, X, num=len(X))




        st.write("Hiperparametre optimizasyonu, modelin çıkartılması, pipeline ile oluşturulmuştur.")
        pipeline = Image.open('pipeline.png')
        st.image(pipeline, caption='Pipeline Sonuçları')
        # st.write("İşlem başladı \n"
        #          "Base Models....\n"
        #          "accuracy: 0.8033 (LR) \n"
        #          "accuracy: 0.8284 (KNN) \n"
        #          "accuracy: 0.8395 (SVC) "
        #          "accuracy: 0.8542 (CART) "
        #          "accuracy: 0.8897 (RF) "
        #          "accuracy: 0.8141 (Adaboost) "
        #          "accuracy: 0.8404 (GBM) "
        #          "accuracy: 0.8844 (XGBoost) "
        #          "accuracy: 0.8768 (LightGBM) "
        #          "Hyperparameter Optimization...."
        #          "########## CART ##########"
        #          "accuracy (Before): 0.8546"
        #          "accuracy (After): 0.8649"
        #          "CART best params: {'max_depth': 12, 'min_samples_split': 9}"
        #          "########## RF ##########"
        #          "accuracy (Before): 0.8892"
        #          "accuracy (After): 0.8869"
        #          "RF best params: {'max_depth': None, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 500}"
        #          "########## XGBoost ##########"
        #          "accuracy (Before): 0.8844"
        #          "accuracy (After): 0.8944"
        #          "XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 500}"
        #          "########## LightGBM ##########"
        #          "accuracy (Before): 0.8768"
        #          "accuracy (After): 0.8956"
        #          "LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 10000}"
        #          "Voting Classifier..."
        #          "Accuracy: 0.8924879505883266"
        #          "F1Score: 0.8288231161713395"
        #          "ROC_AUC: 0.9515720334609802"
        #          "Process finished with exit code 0")

col1, col2, col3, col4,col5, col6 = st.columns(6)
with col2:
    buraksefo = Image.open('buraksefo.jpeg')
    st.image(buraksefo, caption = 'Miuul Resort Hotel Şekerleri',width=1200)

