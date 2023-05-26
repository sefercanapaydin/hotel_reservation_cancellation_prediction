 Booking_ID	Unique identifier of the booking.
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
import missingno as msno
from sklearn.preprocessing import StandardScaler
from datetime import date
from analysis_functions import *


from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def df_load():
    global df
    df = pd.read_csv("hotel_bookings.csv")
    df["booking_status"] = np.where(df["booking_status"] == "Not_Canceled", "0", "1").astype("int64")
    df.drop("Booking_ID", axis=1, inplace=True)
    df.loc[df['type_of_meal_plan'] == 'Not Selected', 'type_of_meal_plan'] = 'Just Room'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 1', 'type_of_meal_plan'] = 'Breakfast'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 2', 'type_of_meal_plan'] = 'Half Board'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 3', 'type_of_meal_plan'] = 'Full Board'


df_load()
check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
cat_cols = [col for col in cat_cols if "booking_status" not in col]

######## missing values

df.isnull().sum()
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

########## filling missing values
for col in num_cols:
     df[col].fillna(df.groupby("booking_status")[col].transform("median"), inplace = True)

for col in cat_cols:
     df[col].fillna(df.groupby("booking_status")[col].transform(lambda x: x.fillna(x.mode().iloc[0])), inplace= True)

#########################EDA



for col in df.columns:
    if col not in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type','avg_price_per_room']:
        df[col] = df[col].astype("int64")








df.dtypes

for col in cat_cols:
    cat_summary(df, col, plot= True)
for col in num_cols:
    num_summary(df, col, plot= True)
for col in num_cols:
    target_summary_with_num(df,"booking_status", col)
for col in cat_cols:
    target_summary_with_cat(df,"booking_status", col)



### feature engıneerıngten sonra tekrar bak


######################### outlier
df2 = df.copy()
check_outlier(df, num_cols)

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    sns.boxplot(x=df2[col], whis = 1.5)
    plt.show(block = True)

df2.describe().T


#Before treshold
#
#                                          count    mean    std   min    25%    50%     75%     max
# no_of_weekend_nights                 36275.000   0.812  0.867 0.000  0.000  1.000   2.000   7.000
# no_of_week_nights                    36275.000   2.199  1.394 0.000  1.000  2.000   3.000  17.000
# lead_time                            36275.000  85.052 85.579 0.000 17.000 57.000 125.000 443.000
# arrival_month                        36275.000   7.432  3.048 1.000  5.000  8.000  10.000  12.000
# arrival_date                         36275.000  15.616  8.625 1.000  8.000 16.000  23.000  31.000
# no_of_previous_cancellations         36275.000   0.023  0.368 0.000  0.000  0.000   0.000  13.000
# no_of_previous_bookings_not_canceled 36275.000   0.152  1.751 0.000  0.000  0.000   0.000  58.000
# avg_price_per_room                   36275.000 103.372 34.844 0.000 80.750 99.450 120.000 540.000
# booking_status                       36275.000   0.328  0.469 0.000  0.000  0.000   1.000   1.000
# 0.1- 0.9 treshold
#                                          count    mean    std   min    25%    50%     75%     max
# no_of_weekend_nights                 36275.000   0.812  0.863 0.000  0.000  1.000   2.000   5.000
# no_of_week_nights                    36275.000   2.189  1.336 0.000  1.000  2.000   3.000   8.500 -
# lead_time                            36275.000  85.052 85.579 0.000 17.000 57.000 125.000 443.000
# arrival_month                        36275.000   7.432  3.048 1.000  5.000  8.000  10.000  12.000
# arrival_date                         36275.000  15.616  8.625 1.000  8.000 16.000  23.000  31.000
# no_of_previous_cancellations         36275.000   0.000  0.000 0.000  0.000  0.000   0.000   0.000 -
# no_of_previous_bookings_not_canceled 36275.000   0.000  0.000 0.000  0.000  0.000   0.000   0.000 -
# avg_price_per_room                   36275.000 103.338 34.636 0.000 80.750 99.450 120.000 266.200 -
# booking_status                       36275.000   0.328  0.469 0.000  0.000  0.000   1.000   1.000
# 0.01 - 0.99 treshold
#                                          count     mean    std      min      25%      50%      75%      max
# no_of_adults                         36275.000    1.847  0.516    0.000    2.000    2.000    2.000    4.000
# no_of_children                       36275.000    0.104  0.401    0.000    0.000    0.000    0.000   10.000-
# no_of_weekend_nights                 36275.000    0.812  0.863    0.000    0.000    1.000    2.000    5.000
# no_of_week_nights                    36275.000    2.199  1.392    0.000    1.000    2.000    3.000   15.000
# required_car_parking_space           36275.000    0.029  0.166    0.000    0.000    0.000    0.000    1.000
# lead_time                            36275.000   85.052 85.579    0.000   17.000   57.000  125.000  443.000
# arrival_year                         36275.000 2017.822  0.382 2017.000 2018.000 2018.000 2018.000 2018.000
# arrival_month                        36275.000    7.432  3.048    1.000    5.000    8.000   10.000   12.000
# arrival_date                         36275.000   15.616  8.625    1.000    8.000   16.000   23.000   31.000
# repeated_guest                       36275.000    0.025  0.157    0.000    0.000    0.000    0.000    1.000
# no_of_previous_cancellations         36275.000    0.000  0.000    0.000    0.000    0.000    0.000    0.000 -
# no_of_previous_bookings_not_canceled 36275.000    0.099  0.835    0.000    0.000    0.000    0.000   10.000 -
# avg_price_per_room                   36275.000  103.371 34.837    0.000   80.750   99.450  120.000  519.750
# no_of_special_requests               36275.000    0.606  0.782    0.000    0.000    0.000    1.000    5.000
# booking_status                       36275.000    0.328  0.469    0.000    0.000    0.000    1.000    1.000

df2.loc[df2['no_of_previous_bookings_not_canceled']>0].count()/len(df2)
df2.loc[df2['no_of_previous_bookings_not_canceled']==0].count()/len(df2)

df2.loc[df2['no_of_previous_cancellations']>0].count()/len(df2)
df2.loc[df2['no_of_previous_cancellations']==0].count()/len(df2)

df2 = df.copy()

############################# feature_extraction

# ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'type_of_meal_plan',
#  'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
#  'arrival_date', 'market_segment_type', 'repeated_guest', 'no_of_previous_cancellations',
#  'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests', 'booking_status']


df['avg_price_per_room'].describe()

df.groupby("room_type_reserved")["avg_price_per_room"].agg(["mean","count"])
def time_features (df):
    temp = df.rename(columns={
        'arrival_year': 'year',
        'arrival_month': 'month',
        'arrival_date': 'day'
    })
    df['date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors='coerce')
    df['date'].fillna('2018-02-28', inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['booking_date'] = df['date'] - pd.to_timedelta(df['lead_time'], unit='D')
    df['booking_year'] = df['booking_date'].dt.year
    df['booking_month'] = df['booking_date'].dt.month
    df['booking_week'] = df['booking_date'].dt.isocalendar().week.astype(float)
    df['booking_day'] = df['booking_date'].dt.day
    df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    df['booking_quarter'] = df['booking_date'].dt.quarter
    df['booking_dayofyear'] = df['booking_date'].dt.dayofyear
    df.drop(['booking_date','date','arrival_month','arrival_date','arrival_year'], axis=1, inplace=True)
    return df


df = time_features(df)

df.head()

df['room_segment'] = pd.qcut(df['avg_price_per_room'], 4, labels=['cheap','normal','expensive','suite'] )

df["no_of_guest_NEW"] = df["no_of_adults"].astype('int64')+df["no_of_children"].astype('int64')
df["no_of_nights_NEW"] = df["no_of_weekend_nights"].astype('int64')+df["no_of_week_nights"].astype('int64')



cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
cat_cols = [col for col in cat_cols if "booking_status" not in col]
for col in cat_cols:
    df[col] = df[col].astype("object")


############# Rare Analyser
rare_analyser(df, "booking_status", cat_cols)

df = rare_encoder(df, 0.01)

df.isnull().sum()



################ correlation


correlation_matrix(df, num_cols)

df.head()

correlation_matrix(df, num_cols)

plt.figure(dpi = 120,figsize= (5,4))
mask = np.triu(np.ones_like(df.corr(),dtype = bool))
sns.heatmap(df.corr(),mask = mask, fmt = ".2f",annot=True,lw=1,cmap = 'plasma')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')
plt.show(block=True)



high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1, inplace=True)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)
################# Encoding

df = one_hot_encoder(df, cat_cols, drop_first=True)



# Standartlaştırma

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)


##### Modelleme
y = df["booking_status"]
X = df.drop(["booking_status"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

X.head()

##### XGBOOST MODEL

xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False).fit(X_train,y_train)
y_pred = xgboost_model.predict(X_test)
accuracy_score(y_pred, y_test)
plot_importance(xgboost_model, X, num=len(X))

xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


##### RANDOMFOREST MODEL
rf_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


rf_model.get_params()
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
plot_importance(rf_model, X, num=len(X))

##### CATBOOST MODEL
catboost_model = CatBoostClassifier(random_state=17, verbose=False).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
accuracy_score(y_pred, y_test)

catboost_model.get_params()
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
plot_importance(catboost_model, X, num=len(X))

############# LGBM MODEL
lgbm_model = LGBMClassifier(random_state=17).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_pred, y_test)

lgbm_model.get_params()
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



base_models(X, y, scoring="accuracy")

best_models = hyperparameter_optimization(X, y)

voting_clf = voting_classifier(best_models, X, y)



joblib.dump(voting_clf, "voting_clf1.pkl")

new_model = joblib.load("voting_clf2.pkl")








# for col in num_cols:
#     plt.figure(figsize=(10,8))
#     #sns.distplot(df.loc[df.booking_status==1][col],kde_kws={'label':'Not Canceled'},color='green')
#     #sns.distplot(df.loc[df.booking_status==0][col],kde_kws={'label':'Canceled'},color='red')
#     sns.kdeplot(x=col,hue='booking_status',shade=True,data=df,)
# plt.legend(['Booking','No Booking'])
# plt.show(block=True)





df_load()


df.head()
df.type_of_meal_plan.unique()

cm = ConfusionMatrix(decision_tree)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)