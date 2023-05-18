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








from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from datetime import date
from analysis_functions import *






pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def df_load():
    global df
    df = pd.read_csv("hotel_bookings.csv")
    df["booking_status"] = np.where(df["booking_status"] == "Not_Canceled", "0", "1").astype("int64")
    df.drop("Booking_ID", axis=1, inplace=True)



df_load()




check_df(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
df.nunique()
cat_cols = [col for col in cat_cols if "booking_status" not in col]
for col in cat_cols:
    if col not in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']:
        df[col] = df[col].astype("object")




for col in cat_cols:
    cat_summary(df, col, "booking_status", plot= True)
for col in num_cols:
    num_summary(df, col, plot= True)
for col in num_cols:
    target_summary_with_num(df,"booking_status", col)
for col in cat_cols:
    target_summary_with_cat(df,"booking_status", col)
rare_analyser(df, "booking_status", cat_cols)


new_df = rare_encoder(df, 0.01)

######## missing values

missing_values_table(df, True)
na_cols = missing_values_table(df, True)
missing_vs_target(df, "booking_status", na_cols)

# eksik gözlem tablolaları
msno.bar(df)
plt.show(block=True)
msno.matrix(df)
plt.show(block=True)
msno.heatmap(df)
plt.show(block=True)

df.dropna().shape

df.dtypes

df.head()

for col in num_cols:
     df[col].fillna(df.groupby("booking_status")[col].transform("median"), inplace = True)

for col in cat_cols:
     df[col].fillna(df.groupby("booking_status")[col].transform(lambda x: x.fillna(x.mode().iloc[0])), inplace= True)

df_load()
df.describe().T
df.isnull().sum()




# df.arrival_year = pd.to_datetime(df.arrival_year, format='%Y')





################ correlation

correlation_matrix(df, num_cols)

df.head()

correlation_matrix(df, num_cols)

######################### outlier

check_outlier(df, num_cols)
for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    sns.boxplot(x=df[col], whis = 1.5)
    plt.show(block = True)

# ohe_cols = [col for col in df.cat_cols if dataframe[col].dtypes == "O"]

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape

df_load()
# Standartlaştırma

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["booking_status"]
X = df.drop(["booking_status"], axis=1)



base_models(X, y, scoring="accuracy")






df.arrival_year.value_counts()

df.dtypes
df.describe().T
df.nunique()


df["no_of_guest_NEW"] = df["no_of_adults"]+df["no_of_children"]

df["no_of_nights_NEW"] = df["no_of_weekend_nights"]+df["no_of_week_nights"]

df['no_of_previous_cancellations'].value_counts()
df['no_of_previous_bookings_not_canceled'].value_counts()