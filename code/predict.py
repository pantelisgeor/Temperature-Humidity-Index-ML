import os
import pandas as pd
import xarray as xr
import glob
import gc
import warnings
from tqdm import tqdm
from calendar import monthrange
import datetime
import argparse
import joblib
import xgboost as xgb
import numpy as np


# =========================== Argument parser ========================= #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Climate model",
                    choices=['GFDL-ESM4', 'CMCC-CM2-SR5', 'NorESM2-MM',
                             'FGOALS-g3', 'ACCESS-ESM1-5', 'INM-CM4-8',
                             'GFDL-CM4', 'MRI-ESM2-0', 'MIROC6',
                             'EC-Earth3-Veg-LR', 'NESM3', 'INM-CM5-0',
                             'BCC-CSM2-MR', 'IITM-ESM', 'CESM2-WACCM',
                             'EC-Earth3'])
parser.add_argument("--scenario", type=str, choices=["ssp245", "ssp585"],
                    help="Climate scenario (ssp245 or ssp585")
parser.add_argument("--startYear", type=int,
                    help="Start of prediction window")
parser.add_argument("--endYear", type=int,
                    help="End of prediction window")
parser.add_argument("--pathSave", type=str,
                    help="Path to directory to save predictions.")
parser.add_argument("--pathModel", type=str,
                    help="Path to directory of model to be used.")
parser.add_argument("--path_data", type=str,
                    help="Path to directory where CMIP6 data are stored.")
parser.add_argument("--root", type=str,
                    help="Path to root directory for training.")
parser.add_argument("--scaler_feats", type=str,
                    help="Path to feature scaler.")
parser.add_argument("--scaler_target", type=str,
                    help="Path to target variable scaler.")
args = parser.parse_args()
# ======================================================================= #


# -------------- FUNCTIONS ------------- #
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


def loadModel(pathModel) -> xgb.Booster:
    """
    Lists the trained model iterations in the directory and
    loads the latest one.
    Args:
        pathModel: Path to model's directory
    Returns:
        model: XGB model
    """
    # List the models
    models = glob.glob(f"{pathModel}/*.model")
    # Put them in a dataframe and get the details from the filename
    models_ = pd.DataFrame({
        "year": [int(x.split("_")[-2]) for x in models],
        "month": [int(x.split("_")[-1].replace(".model", "")) for
                  x in models],
        "path": models
    })
    models = models_.sort_values(by=["year", "month"], ascending=True)\
        .reset_index(drop=True)
    model = xgb.Booster()
    model.load_model(models.path.values[-1])
    return model


def modelDat(model, path_) -> pd.DataFrame:
    # Dataset directory
    path_dat = f"{path_}/{model}"
    # List all the netcdf files in the path_dat directory
    files = glob.glob1(path_dat, "*.nc")
    # Put the data in a pandas dataframe
    df_ = pd.DataFrame({
        "variable": [x.split("_")[0] for x in files],
        "model": [x.split("_")[2] for x in files],
        "scenario": [x.split("_")[3] for x in files],
        "year": [int(x.split("_")[-1].split(".")[0]) for x in files],
        "filename": files
    })
    return df_.sort_values(by=["year", "scenario"], ascending=True)


def getYear(model, year, scenario, df_dat, path_, ds_lsm,
            vars_=["tas", "tasmin", "tasmax", "hurs"]) -> xr.Dataset:
    # Read all the variables and combine them
    for v_ in vars_:
        filename = df_dat.loc[(df_dat.model == model) &
                              (df_dat.year == year) &
                              (df_dat.scenario == scenario) &
                              (df_dat.variable == v_)]\
            .filename.item()
        if v_ == vars_[0]:
            ds = xr.open_dataset(f"{path_}/{model}/{filename}")
        else:
            ds = ds.merge(xr.open_dataset(f"{path_}/{model}/{filename}"))

    # Convert mean temperature to Celcius
    ds = ds.rename({"tas": "t2m", "tasmax": "t2mmax",
                    "tasmin": "t2mmin", "hurs": "rhmean"})
    ds["t2mC"] = ds.t2m - 273.15
    # Merge with land sea mask
    ds = ds.merge(ds_lsm)
    # Convert time index to datetime index
    try:
        ds["time"] = ds.indexes['time'].to_datetimeindex()
    except AttributeError:
        ds["time"] = pd.to_datetime(ds["time"].values)
    # Get unique dates and latitutdes in the netcdf
    dates_lats = [[date, lat] for date in ds.time.values
                  for lat in ds.lat.values]
    dayLengths = [daylength(x) for x in tqdm(dates_lats)]
    # Convert to pandas dataframe
    df_ = pd.DataFrame({"time": [x[0] for x in dates_lats],
                        "lat": [x[1]for x in dates_lats],
                        "dayLength": dayLengths})
    # Convert to xarray
    ds_ = df_.set_index(["time", "lat"]).to_xarray()
    # Merge with ds
    ds = ds.merge(ds_)
    # Tidy up
    del ds_, df_, dates_lats, dayLengths
    gc.collect()
    # Calculate daily THI index from mena Relative Humidity
    # and mean temperature
    ds["THI_"] = (1.8 * ds["t2mC"] + 32) - (0.55 - (0.0055 * ds["rhmean"])) *\
        (1.8 * ds["t2mC"] - 26)
    # Drop temperature in celcius
    ds = ds.drop_vars("t2mC")
    return ds


def getMonth(ds, month, year, feats, scaler_feats) -> pd.DataFrame:
    # Subset the data for the specified month and convert to a pandas dataframe
    ds_ = ds.sel(time=slice(f"{year}-{month}-01",
                            f"{year}-{month}-{monthrange(year, month)[1]}"))
    # Convert to pandas dataframe
    df = ds_.to_dataframe().reset_index(drop=False)
    # Drop missing values and convert time to str
    df = df.dropna()
    df = df.assign(date=[str(x).split("T")[0] for x in df.time.values])
    gc.collect()
    # Create an array with all the hours in a year
    df_ = pd.DataFrame({
        "time": pd.Series(pd.date_range(start=f'{year}-01-01',
                                        end=f'{year}-12-31 23:00:00',
                                        freq='H'))})
    # Assign the hour of day and day of year columns and date
    # in str format
    df_ = df_.assign(dayOfYear=[pd.Timestamp(x).day_of_year for
                                x in df_.time.values],
                     hourOfDay=[pd.Timestamp(x).hour for
                                x in df_.time.values],
                     date=[str(x).split("T")[0] for x
                           in df_.time.values])
    # Merge the two to construct the dataset for predictions
    df_feats = pd.merge(df_, df.drop("time", axis=1), on="date",
                        how="left")
    df_feats = df_feats.dropna().reset_index(drop=True)
    del df_, df
    gc.collect()
    # Normalise it
    df_feats[feats] = scaler_feats.transform(df_feats[feats])
    return df_feats


def predictMonth(ds, month, year, feats, coords, modelML,
                 scaler_feats, scaler) -> xr.Dataset:
    # Get dataset for month
    df = getMonth(ds=ds, month=month, year=year, feats=feats,
                  scaler_feats=scaler_feats)
    # Make predictions
    preds = modelML.predict(xgb.DMatrix(df[feats]))
    gc.collect()
    # add to initial dataframe
    df = df.drop(feats + ["date"], axis=1)
    # Scale it back
    df["THI"] = scaler.inverse_transform(preds.reshape(-1, 1))
    del preds
    gc.collect()
    # Merge with coords to get xarray
    df2 = pd.merge(coords, df, on=["lat", "lon"], how="left")
    ds2 = df2.set_index(["time", "lat", "lon"]).to_xarray()
    del df, df2, ds
    gc.collect()
    return ds2


def predictYear(model, year, scenario, df_dat, path_,
                feats, scaler_feats, scaler, ds_lsm) -> xr.Dataset:
    # Construct the xarray dataset
    print("Getting data for the specified model, scenario and year")
    ds = getYear(model=model, year=year, scenario=scenario,
                 ds_lsm=ds_lsm, df_dat=df_dat, path_=path_)
    # Create a dataframe with the coordinates of the xarray
    print("Creating coordinates dataframe.")
    coords = pd.concat([pd.DataFrame({"lat": [lat], "lon": [lon]}) for
                        lat in tqdm(ds.lat.values) for lon in ds.lon.values])
    # Make the prediction for all months and combine them
    t1 = datetime.datetime.now()
    for month in range(1, 13):
        print(f"Running prediction for {model}_{scenario}: {month}-{year}")
        if month == 1:
            dsPred = predictMonth(
                ds, month, year, feats, coords, modelML,
                scaler_feats, scaler)
        else:
            dsPred = dsPred.merge(
                predictMonth(ds, month, year, feats, coords, modelML,
                             scaler_feats, scaler))
    t2 = datetime.datetime.now()
    print(f"{(t2-t1).seconds/60} min.")
    return dsPred


# -------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Read the scaler
    scaler_feats = joblib.load(args.scaler_feats)
    scaler = joblib.load(args.scaler_target)
    
    path_ = args.path_data
    os.chdir(path_)
    # Available CMIP6 model datasets
    # (only models which are complete are listed here)
    models = ['ACCESS-ESM1-5', 'CMCC-CM2-SR5', 'EC-Earth3',
              'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-CM4',
              'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0',
              'MIROC6', 'MRI-ESM2-0', 'NorESM2-MM']
    # Read the land sea mask
    ds_lsm = xr.open_dataset(f"{args.root}/CMIP6_lsm.nc")
    # Load a model
    modelML = loadModel(args.pathModel)
    # List the datasets from the CMIP6 NASA-NEX directories
    df_dat = pd.concat([modelDat(x, path_=path_) for x in models])
    # Drop historical datasets
    df_dat = df_dat.loc[df_dat.scenario != "historical"]
    df_dat = df_dat.reset_index(drop=True)
    # Count the available datasets for each model
    dfCounts = df_dat.groupby(["model", "variable", "scenario"],
                              as_index=False).size()
    # Feature set
    feats = ["THI_", "dayLength", "t2m", "t2mmax", "lsm", "t2mmin",
             "rhmean", "dayOfYear", "hourOfDay"]
    # Path to directory where predictions are to be saved
    path_save = f"{args.pathSave}/{args.model}"
    os.makedirs(args.pathSave, exist_ok=True)
    os.makedirs(path_save, exist_ok=True)
    # Loop through the specified model, scenario and years
    # and perform the predictions
    for year in range(args.startYear, args.endYear+1):
        if os.path.isfile(f"{path_save}/{year}_{args.scenario}.nc"):
            print(f"{args.model} - {args.scenario} - {year}",
                  " exists. Continuing. . .")
            continue
        print(f"{args.model} - {args.scenario} - {year}. Processing. . .")
        ds = predictYear(model=args.model, year=year, scenario=args.scenario,
                         df_dat=df_dat, path_=path_, feats=feats,
                         scaler_feats=scaler_feats, scaler=scaler,
                         ds_lsm=ds_lsm)
        # Add attributes
        ds = ds.assign_attrs(
            creation_date=str(datetime.datetime.now()),
            description="Hourly Temperature Humidity Index data generated" +
            " using machine learning to temporally downscale daily NASA" +
            " NEX-GDDP-CMIP6 climate projections.",
            long_name="temperature humidity index",
            title="THI projections based on NASA NEX-GDDP CMIP6 data",
            institution="The Cyprus Institute",
            institution_id="CyI",
            frequency="hourly",
            history="none",
            grid="Follows the NASA NEX-GDDP-CMIP6 grid (0.25deg)",
            nominal_resolution="25 km")
        # Save it
        ds.to_netcdf(f"{path_save}/{year}_{args.scenario}.nc",
                     encoding={"THI": {"dtype": "float32",
                                       "zlib": True,
                                       "complevel": 5}})
        del ds
        gc.collect()
