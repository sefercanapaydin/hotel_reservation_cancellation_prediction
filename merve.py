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


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime as dt


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("hotel_bookings.csv")


################################### EDA ###################################

df.isnull().sum()
df.shape
df.describe().T
df.dtypes

########## astype

df.loc[:, 'arrival_year'] = df.loc[:, 'arrival_year'].apply(pd.to_datetime)
df['arrival_year'] = df['arrival_year'].dt.year

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)