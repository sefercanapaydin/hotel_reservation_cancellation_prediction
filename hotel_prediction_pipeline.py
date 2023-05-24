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
    df = pd.read_csv("hotel_bookings.csv")
    df["booking_status"] = np.where(df["booking_status"] == "Not_Canceled", "0", "1").astype("int64")
    df.drop("Booking_ID", axis=1, inplace=True)
    df.loc[df['type_of_meal_plan'] == 'Not Selected', 'type_of_meal_plan'] = 'Just Room'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 1', 'type_of_meal_plan'] = 'Breakfast'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 2', 'type_of_meal_plan'] = 'Half Board'
    df.loc[df['type_of_meal_plan'] == 'Meal Plan 3', 'type_of_meal_plan'] = 'Full Board'
    return df
def data_preprocessing(df):

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
    cat_cols = [col for col in cat_cols if "booking_status" not in col]

    for col in num_cols:
        df[col].fillna(df.groupby("booking_status")[col].transform("median"), inplace=True)
    for col in cat_cols:
        df[col].fillna(df.groupby("booking_status")[col].transform(lambda x: x.fillna(x.mode().iloc[0])), inplace=True)

    for col in df.columns:
        if col not in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'avg_price_per_room']:
            df[col] = df[col].astype("int64")
    # def time_features(df):
    #     temp = df.rename(columns={
    #         'arrival_year': 'year',
    #         'arrival_month': 'month',
    #         'arrival_date': 'day'
    #     })
    #     df['date'] = pd.to_datetime(temp[['year', 'month', 'day']], errors='coerce')
    #     df['date'].fillna('2018-02-28', inplace=True)
    #     df['year'] = df['date'].dt.year
    #     df['month'] = df['date'].dt.month
    #     df['day'] = df['date'].dt.day
    #     df['dayofweek'] = df['date'].dt.dayofweek
    #     df['quarter'] = df['date'].dt.quarter
    #     df['dayofyear'] = df['date'].dt.dayofyear
    #     df['booking_date'] = df['date'] - pd.to_timedelta(df['lead_time'], unit='D')
    #     df['booking_year'] = df['booking_date'].dt.year
    #     df['booking_month'] = df['booking_date'].dt.month
    #     df['booking_week'] = df['booking_date'].dt.isocalendar().week.astype(float)
    #     df['booking_day'] = df['booking_date'].dt.day
    #     df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    #     df['booking_quarter'] = df['booking_date'].dt.quarter
    #     df['booking_dayofyear'] = df['booking_date'].dt.dayofyear
    #     df.drop(['booking_date', 'date', 'arrival_month', 'arrival_date', 'arrival_year'], axis=1, inplace=True)
    #     return df
    #
    # df = time_features(df)
    df["no_of_guest_NEW"] = df["no_of_adults"].astype('int64') + df["no_of_children"].astype('int64')
    df["no_of_nights_NEW"] = df["no_of_weekend_nights"].astype('int64') + df["no_of_week_nights"].astype('int64')

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=8, car_th=20)
    cat_cols = [col for col in cat_cols if "booking_status" not in col]

    for col in cat_cols:
        df[col] = df[col].astype("object")

    df = rare_encoder(df, 0.01)

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["booking_status"]
    X = df.drop(["booking_status"], axis=1)

    return X, y



def main():
    df = df_load()
    X, y = data_preprocessing(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf
    print(X.head(1))

if __name__ == "__main__":
    print("İşlem başladı")
    main()

# Voting Classifier...
# Accuracy: 0.8925706476137388
# F1Score: 0.8289764318866539
# ROC_AUC: 0.9515292269251306