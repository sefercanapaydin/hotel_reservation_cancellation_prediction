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
import plotly.express as px

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("hotel_bookings.csv")


######## TOTAL BOOKINGS MARKET SEGMENT (PIE)


segments=df["market_segment_type"].value_counts()
fig = px.pie(segments,
             values=segments.values,
             names=segments.index,
             title="Bookings per market segment")
fig.update_traces(rotation=-90, textinfo="percent+label")
fig.show()


######## TOTAL BOOKINGS MEAL PLAN  (PIE)



segments=df["type_of_meal_plan"].value_counts()
fig = px.pie(segments,
             values=segments.values,
             names=segments.index,
             title="Bookings per meal type")
fig.update_traces(rotation=-90, textinfo="percent+label")
fig.show()



######## CANCELLATION BY LEAD TIME


plt.figure(figsize=(10, 7))
labels = ['1-30\n days', '31-60\n days', '61-90\n days', '91-120\n days', '121-150\n days',
          '151-180\n days', '181-210\n days', '211-240\n days', '241-270\n days',
          '271-300\n days', '300+\n days']
sns.countplot(x=pd.cut(df["lead_time"], bins=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, np.inf], labels=labels ),hue="booking_status", data=df, width=0.7, palette='viridis')
plt.title("Cancellation by Lead Time")
plt.xlabel("Lead Time\n (days)")
plt.ylabel("Count")
plt.xticks(fontsize=8)
plt.legend(title="Booking Status", labels=['Not Cancelled', 'Cancelled'])
plt.subplots_adjust(bottom=0.05)
plt.show(block=True)


######## CANCELLATION BY SEGMENT TYPE


sns.set(style="whitegrid", palette="viridis", color_codes=True)
plt.figure(figsize=(10,6))
ax = sns.countplot(y="market_segment_type", hue="booking_status", data=df, order=df["market_segment_type"].value_counts().index)
ax.set(xlabel='Count', ylabel='Market Segment')
plt.legend(title="Booking Status", labels=['Not Cancelled', 'Cancelled'], loc='lower right')
plt.title('Booking Status by Market Segment Type')
plt.show()


######## MOST BUSIEST MONTH



import calendar

# Example dataset with numerical values for months
df1 = pd.DataFrame({'arrival_month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'count': [1000, 1668, 2328, 2700, 2563, 3162, 2887, 3761, 4550, 5742, 2937, 2977]})

# Map numerical month values to month names
df1['arrival_month'] = df1['arrival_month'].apply(lambda x: calendar.month_name[x])

# Specify the order of months
ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'month' column to categorical with specified order of months
df1['arrival_month'] = pd.Categorical(df1['arrival_month'], categories=ordered_months, ordered=True)

# Plot the data
sns.barplot(x='arrival_month', y='count', data=df1, palette='viridis')
plt.title('Counts by Month', weight='bold')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()


######## BOOKING STATUS BY MONTH


ordered_months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
#df['arrival_month'] = pd.Categorical(df['arrival_month'], categories=ordered_months, ordered=True)

plt.figure(figsize=(12,6))
sns.countplot(x='arrival_month', hue='booking_status', data = df, palette= 'viridis')
plt.title('Booking Status by Month', weight='bold')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.legend(title="Booking Status", labels=['Not Cancelled', 'Cancelled'])
# Manually renaming x-tick labels
plt.xticks(range(0,12), ordered_months, rotation=45)
plt.show()


######## BOOKING STATUS BY AVERAGE PRICE


plt.figure(figsize=(10, 7))
labels = ['0-50', '51-100', '101-150', '151-200', '201+']
sns.countplot(x=pd.cut(df["avg_price_per_room"], bins=[0, 50, 100, 150, 200, np.inf], labels=labels ),hue="booking_status", data=df, width=0.7, palette='viridis')

plt.title("Cancellation by Average Price Per Room")
plt.xlabel("Average price\n (USD)")
plt.ylabel("Count")
plt.xticks(fontsize=8)
plt.legend(title="Booking Status", labels=['Not Cancelled', 'Cancelled'])
plt.subplots_adjust(bottom=0.05)
plt.show()


######## CHI-SQUARE TEST


from scipy.stats import chi2_contingency

# create contingency table
contingency_table = pd.crosstab(df['booking_status'], df['avg_price_per_room'])

# conduct chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# print results
print("Chi-square statistic:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)


######## Relationship between Avg Price and Arrival Month by Booking cancellation status


plt.figure(figsize=(12,6))
sns.lineplot(x = "arrival_month", y = "avg_price_per_room", hue="booking_status",hue_order= [1,0],data=df, palette= 'Set1')
plt.title("Relationship between Avg Price and Arrival Month by Booking cancellation status", weight = 'bold')
plt.xlabel("Arrival Month")
plt.xticks(rotation=45)
plt.ylabel("Average Price")
# Manually renaming x-tick labels
plt.xticks(range(1,13), ordered_months, rotation=45)
plt.legend(loc="upper right")
plt.grid(False)
plt.show()


######## Total Nights Spent by Guests by Market Segment & Hotel Type


df['total_stay'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
plt.figure(figsize=(12,6))
sns.barplot(x = "market_segment_type", y = "total_stay", data = df, palette = 'rocket')
plt.title('Total Nights Spent by Guests by Market Segment & Hotel Type', weight='bold')
plt.xlabel('Market Segment')
plt.ylabel('Number of Days');


######## Arrival Year vs Lead Time By Cancellation Status


plt.figure(figsize=(12,6))
sns.barplot(x='arrival_year', y ='lead_time', hue="booking_status", data=df, palette="viridis")
plt.title('Arrival Year vs Lead Time By Cancellation Status', weight='bold')
plt.xlabel(' Arrival Year')
plt.ylabel('Lead Time')
plt.legend(loc = "upper right")

######## ONCEKI REZ GORE IPTAL EDIP ETMEME


bookings = df[df["no_of_previous_bookings_not_canceled"] != 0]["no_of_previous_bookings_not_canceled"].value_counts()
cancellations = df[df["no_of_previous_cancellations"] != 0]["no_of_previous_cancellations"].value_counts()

df_stacked = pd.DataFrame({"bookings": bookings, "cancellations": cancellations})
df_stacked.plot(kind="bar", stacked=True)

plt.xlabel("Number of Previous Bookings/Cancellations")
plt.ylabel("Count", fontsize = 14,)
plt.title("Comparison of Bookings and Cancellations\n(cancellations 0 = 35441 entries)\n(prev_bookings 0 = 34923 entries)")
plt.xlim(-1,14)
plt.xticks(range(0,15))
plt.show()


######## ANALYSIS OF LEAD TIME


bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 443]

lead_time["lead_time_bin"] = pd.cut(lead_time["lead_time"], bins=bins)
lead_time_counts = lead_time.groupby(["lead_time_bin", "booking_status"]).size().reset_index(name="count")
total_count = lead_time.groupby(["lead_time_bin"]).size().reset_index(name="total")

plt.figure(figsize=(12, 6))
sns.barplot(x="lead_time_bin", y="count", hue="booking_status", data=lead_time_counts)
sns.pointplot(x="lead_time_bin", y="total", data=total_count, color="k")

plt.title("Count of Bookings by Lead Time")
plt.xlabel("Lead Time (in 30-day intervals)")
plt.ylabel("Count of Bookings")
plt.ylim([0,6000])  # to zoom in

plt.tight_layout()
plt.show(block=True)


######## CORR MAP


plt.figure(figsize = (15,10))
sns.heatmap(hotels.corr(),annot=True)


######## HEDEF DEGISKEN ANALIZI (DALGA GRAFIK)


for col in numerical_col:
    plt.figure(figsize=(10,8))
    #sns.distplot(df.loc[df.booking_status==1][col],kde_kws={'label':'Not Canceled'},color='green')
    #sns.distplot(df.loc[df.booking_status==0][col],kde_kws={'label':'Canceled'},color='red')
    sns.kdeplot(x=col,hue='booking_status',shade=True,data=hotels,)

plt.legend(['Booking','No Booking'])


######## DEGISKENLERIN UNIQUE DEGERLERI COUNTLARI


plt.figure(figsize = (20,25))

plt.subplot(4,2,1)
plt.gca().set_title('Variable no_of_adults')
sns.countplot(x = 'no_of_adults', palette = 'Set2', data = hotels)

plt.subplot(4,2,2)
plt.gca().set_title('Variable no_of_children')
sns.countplot(x = 'no_of_children', palette = 'Set2', data = hotels)

plt.subplot(4,2,3)
plt.gca().set_title('Variable no_of_weekend_nights')
sns.countplot(x = 'no_of_weekend_nights', palette = 'Set2', data = hotels)

plt.subplot(4,2,4)
plt.gca().set_title('Variable no_of_week_nights')
sns.countplot(x = 'no_of_week_nights', palette = 'Set2', data = hotels)

plt.subplot(4,2,5)
plt.gca().set_title('Variable type_of_meal_plan')
sns.countplot(x = 'type_of_meal_plan', palette = 'Set2', data = hotels)

plt.subplot(4,2,6)
plt.gca().set_title('Variable required_car_parking_space')
sns.countplot(x = 'required_car_parking_space', palette = 'Set2', data = hotels)

plt.subplot(4,2,7)
plt.gca().set_title('Variable room_type_reserved')
sns.countplot(x = 'room_type_reserved', palette = 'Set2', data = hotels)

plt.subplot(4,2,8)
plt.gca().set_title('Variable arrival_year')
sns.countplot(x = 'arrival_year', palette = 'Set2', data = hotels)


######## ODA PARASI VE ZAMANA GORE IPTAL ANALIZI (BONCUK GRAFIGI)


plt.figure(figsize = (15,8))
sns.scatterplot(x='avg_price_per_room',y='lead_time',data=hotels,hue='booking_status')


######## HAFTASONU-HAFTAICI IPTAL SAYILARI

val=[6140,10060]
val1=['no_of_week_nights cancelled','no_of_weekend_nights cancelled']
plt.title('Weekdays Vs Weekend Cancellation')
plt.bar(val1,val)
plt.show(block=True)


######## VERIABLES- TARGET VERIABLE


fig = px.histogram(hotels1, x=hotels1["no_of_special_requests"],
             color='booking_status', barmode='group',width=700,title="Special Request Count Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["required_car_parking_space"],
             color='booking_status', barmode='group',width=700,title="Required Car Parking Space Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["no_of_special_requests"],
             color='booking_status', barmode='group',width=700,title="Special Request Count Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["required_car_parking_space"],
             color='booking_status', barmode='group',width=700,title="Required Car Parking Space Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["arrival_year"],
             color='booking_status', barmode='group',width=700,title="Year Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["type_of_meal_plan"],
             color='booking_status', barmode='group',width=700,title="Meal Plan type Vs Booking Status",
             height=500)
fig.show()

fig = px.histogram(hotels1, x=hotels1["room_type_reserved"],
             color='booking_status', barmode='group',width=700,title="Room Type Vs Booking Status",
             height=500)
fig.show()


######## PIE


fig = plt.figure(figsize=(5,5), dpi=80)
hotels1['booking_status'].value_counts().plot(kind='pie',  autopct='%1.0f%%', startangle=360, fontsize=13)