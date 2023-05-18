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



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("hotel_bookings.csv")