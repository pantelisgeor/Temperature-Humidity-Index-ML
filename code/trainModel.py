import os
import pandas as pd
import xarray as xr
from calendar import monthrange
import gc
import numpy as np
from tqdm.contrib.concurrent import process_map
import json
import joblib
import xgboost as xgb
from sklearn import metrics
import warnings
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to directory where ERA5 data are stored")
parser.add_argument("--root", type=str,
                    help="Path to root directory for training.")
parser.add_argument("--model_name", type=str,
                    help="Name of the model to be trained.")
parser.add_argument("--scaler_feats", type=str,
                    help="Path to feature scaler.")
parser.add_argument("--scaler_target", type=str,
                    help="Path to target variable scaler.")
args = parser.parse_args()


# =========================================================================== 
def daylength(dat_) -> float:
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    date, lat = dat_
    # Convert to pandas datetime object to get day of year
    dayOfYear = pd.to_datetime(date).day_of_year
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(
        np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) *
                                         np.tan(np.deg2rad(declinationOfEarth))
                                         ))
        return 2.0*hourAngle/15.0


def getTime(time) -> pd.DataFrame:
    """
    Function to get the "dayOfYear", "hourOfDay" features from the hourly date
    """
    time = pd.to_datetime(time)
    return pd.DataFrame({"time": [time],
                         "date": [str(time).split(" ")[0]],
                         "dayOfYear": [time.dayofyear],
                         "hourOfDay": [time.hour]})


def getMonth(month, year, path_era) -> pd.DataFrame:
    # Read the temperature
    dst2m = xr.open_dataset(
        f"{path_era}/2t/ERA5_{year}_MET-VARIABLES_global_var_2m_temperature.nc")\
        .sel(time=slice(f"{year}-{month}-01",
                        f"{year}-{month}-{monthrange(year, month)[1]}"))
    # Read the relative humidity and merge
    dsRH = xr.open_dataset(
        f"{path_era}/rh/ERA5_{year}_relative_humidity.nc")\
        .sel(time=slice(f"{year}-{month}-01",
                        f"{year}-{month}-{monthrange(year, month)[1]}"))
    dst2m = dst2m.merge(dsRH).merge(dsLSM)
    # Calculate the hourly THI index from the temperature and rel. hum.
    # THI = (1.8 * T + 32) - (0.55 - (0.0055 * RH)) * (1.8 * T - 26)
    # Convert mean temperature to Celcius
    dst2m["t2mC"] = dst2m["t2m"] - 273.15
    dst2m["THI"] = (1.8 * dst2m["t2mC"] + 32) -\
        (0.55 - (0.0055 * dst2m["rh"])) *\
        (1.8 * dst2m["t2mC"] - 26)
    dst2m = dst2m.drop_vars("t2mC")
    gc.collect()

    # Calculate the mean, min and max daily values
    ds_ = dst2m["t2m"].resample(time="1D").mean().to_dataset(name="t2m").merge(
        dst2m["t2m"].resample(time="1D").min().to_dataset(name="t2mmin").merge(
            dst2m["t2m"].resample(time="1D").max().to_dataset(name="t2mmax")))\
        .merge(dsLSM)\
        .merge(dst2m["rh"].resample(time="1d").mean()
               .to_dataset(name="rhmean"))
    gc.collect()
    # Calculate the daily THI index from the mean temperature and mean rh.
    # THI = (1.8 * T + 32) - (0.55 - (0.0055 * RH)) * (1.8 * T - 26)
    # Convert mean temperature to Celcius
    ds_["t2mC"] = ds_["t2m"] - 273.15
    ds_["THI_"] = (1.8 * ds_["t2mC"] + 32) - (0.55 - (0.0055 * ds_["rhmean"])) *\
        (1.8 * ds_["t2mC"] - 26)
    ds_ = ds_.drop_vars("t2mC")

    # Convert the hourly data to pandas dataframe and drop the sea
    df = dst2m.to_dataframe().reset_index(drop=False)
    df = df.loc[df.lsm > 0].reset_index(drop=True)
    gc.collect()
    # Calculate the day of the year and hour of the day variables
    dates = pd.concat(getTime(x) for x in df.time.unique())
    df = df.merge(dates, on="time", how="left")
    del dates
    gc.collect()

    # Convert the daily data to pandas dataframe and drop the sea
    df_ = ds_.to_dataframe().reset_index(drop=False)
    df_ = df_.loc[df_.lsm > 0].reset_index(drop=True)
    gc.collect()
    # Calculate the daylength for the daily data
    timeLat = df_.groupby(["time", "latitude"], as_index=False)\
        .size().drop("size", axis=1)
    timeLat = timeLat.assign(
        dayLength=process_map(daylength, timeLat[["time", "latitude"]].values,
                            max_workers=20, chunksize=len(timeLat)//80))
    # Merge them
    df_ = df_.merge(timeLat, on=["time", "latitude"], how="left")
    del dst2m, dsRH, timeLat
    gc.collect()
    # Convert the date to string to merge with df
    times = df_.time.unique()
    times = pd.DataFrame({"time": times,
                        "date": [str(x).split(" ")[0] for x in times]})
    df_ = df_.merge(times, on="time", how="left")
    del times
    gc.collect()

    # Merge with df (drop the hourly temperature and rel humidity first)
    df_2 = df.drop(["time", "t2m", "lsm", "rh"], axis=1)\
        .merge(df_, on=["longitude", "latitude", "date"], how="left")
    del df, df_
    gc.collect()
    return df_2


def getDatTrain(month, year, path_era, scaler_target, scaler_feats):
    # Read the data for the specified month and year
    df = getMonth(month=month, year=year, path_era=path_era)
    # Split it into feature set and target
    feats = ["THI_", "dayLength", "t2m", "t2mmax",
             "lsm", "t2mmin", "rhmean", "dayOfYear", "hourOfDay"]
    target = ["THI"]
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    X_train[feats] = scaler_feats.transform(df[feats])
    y_train[target] = scaler_target.transform(df[target])
    return X_train, y_train


# =========================================================================== #
# Load the scalers
scaler_feats = joblib.load(args.scaler_feats)
scaler_target = joblib.load(args.scaler_target)

# Global parameters
warnings.filterwarnings("ignore")
path_era = args.path_data
path = args.root
os.chdir(path)

# Model name etc
os.makedirs(f"{path}/models", exist_ok=True)
model_name = args.model_name
path_model = f"models/{model_name}"
os.makedirs(path_model, exist_ok=True)

# Input features and target variable names
feats = ["THI_", "dayLength", "t2m", "t2mmax",
         "lsm", "t2mmin", "rhmean", "dayOfYear", "hourOfDay"]
target = ["THI"]

# Read the era5 land-sea mask
dsLSM = xr.open_dataset(f"{path}/ERA5_lsm.nc")\
    .squeeze().drop_vars("time")
# =========================================================================== #

# =========================================================================== #
# If start from the beginning
print("\t-----------------\t")
if not os.path.isfile(f"{path_model}/training_history.csv"):
    df_hist = pd.DataFrame()
    year = 1980
    month = 1
    # define the model
    params = {"reg_lambda": 1,
              "alpha": 0,
              "num_parallel_tree": 10,
              "max_depth": 5,
              "verbosity": 1,
              "learning_rate": 0.1,
              "tree_method": "approx",
              "n_jobs": 128,
              "objective": "reg:squarederror",
              "eval_metric": "rmse"}
    # Save the parameters json
    with open(f"{path_model}/params.json", "w") as f:
        json.dump(params, f)
else:
    # Read the parameters set
    # Opening JSON file
    with open(f"{path_model}/params.json") as f:
        params = json.load(f)
    # Read the training history dataframe
    df_hist = pd.read_csv(f"{path_model}/training_history.csv")
    # Get the last year it completed training for
    year_tr = df_hist.year.max()
    month_tr = df_hist.loc[df_hist.year == year_tr]["month"].max().item()
    # If the year is 2017 and month 12 stop training
    if year_tr == 2017 and month_tr == 12:
        quit()
    # Load the model
    modelML = xgb.Booster()
    modelML.load_model(f"{path_model}/{model_name}_{year_tr}_{month_tr}.model")
    if month_tr < 12:
        month, year = month_tr + 1, year_tr
    elif month_tr == 12:
        month, year = 1, year_tr + 1
    print(f"Training for year: {year} - month: {month}")


# Get the training data
print(f"Loading data for model {model_name}. . .")
X, y = getDatTrain(month=1, year=1980,
                   path_era=path_era,
                   scaler_target=scaler_target,
                   scaler_feats=scaler_feats)
# Use the first month of 2018 as validation 
if not os.path.isfile(f"{path}/X_val.parquet"):
    X_val, y_val = getDatTrain(month=1, year=2018,
                               path_era=path_era,
                               scaler_target=scaler_target,
                               scaler_feats=scaler_feats)
    gc.collect()
    X_val.to_parquet(f"{path}/X_val.parquet", compression="gzip")
    y_val.to_parquet(f"{path}/y_val.parquet", compression="gzip")
else:
    X_val = pd.read_parquet(f"{path}/X_val.parquet")
    y_val = pd.read_parquet(f"{path}/y_val.parquet")


# Data in XG Train object
xgTrain = xgb.DMatrix(X, label=y)
xgEval = xgb.DMatrix(X_val, label=y_val)
del X
gc.collect()

# Model watchlist
watchlist = [(xgEval, 'eval'), (xgTrain, 'train')]

# Training parameters
num_boost_round = 10
early_stopping_round = 2

# Train the model on these data
if year == 1980 and month == 1:
    print("Starting training from the beginning!")
    # Train the XGBoost model
    modelML = xgb.train(params, dtrain=xgTrain,
                        evals=[(xgTrain, 'train'), (xgEval, 'valid')],
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_round,
                        verbose_eval=1)
    # Save the model
    modelML.save_model(
        f"{path_model}/{model_name}_{year}_{month}.model")
else:
    print("Continuing training. . .")
    print(f"At year: {year} - Month: {month}")
    modelML_ = xgb.train(params, xgTrain,
                         evals=[(xgTrain, 'train'), (xgEval, 'valid')],
                         xgb_model=modelML,
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_round,
                         verbose_eval=1)
    # Save the model
    modelML_.save_model(
        f"{path_model}/{model_name}_{year}_{month}.model")

# Append to the training history dataframe the year completed
pred = modelML.predict(xgTrain)
pred = scaler_target.inverse_transform(pred.reshape(-1, 1))
y[target] = scaler_target.inverse_transform(y)

print(f"MSE: {metrics.mean_squared_error(pred, y)}")
print(f"MAE: {metrics.mean_absolute_error(pred, y)}")

predEval = modelML.predict(xgEval)
predEval = scaler_target.inverse_transform(predEval.reshape(-1, 1))
y_val = scaler_target.inverse_transform(y_val)

print(f"MSE Eval.: {metrics.mean_squared_error(predEval, y_val)}")
print(f"MAE Eval.: {metrics.mean_absolute_error(predEval, y_val)}")

df_hist = pd.concat([
    df_hist,
    pd.DataFrame({"year": [year],
                  "month": [month],
                  "MSE": [metrics.mean_squared_error(pred,
                                                     y)],
                  "MAE": [metrics.mean_absolute_error(pred,
                                                      y)],
                  "MSE_Eval": [metrics.mean_squared_error(predEval,
                                                          y_val)],
                  "MAE_Eval": [metrics.mean_absolute_error(predEval,
                                                           y_val)]})])
# Save it
df_hist.to_csv(f"{path_model}/training_history.csv",
               index=False)

print("\t-----------------\t")
