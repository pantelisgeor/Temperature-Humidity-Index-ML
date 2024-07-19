import os
import re
import argparse
import pandas as pd
from glob import glob1
import numpy as np
import gc
import xarray as xr


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to directory where ERA5 data will be saved")
parser.add_argument("--year", type=int,
                    help="Year to process (calculate Relative Humidity)")
args = parser.parse_args()
# Switch to working directory
os.chdir(args.path_data)


def getYear(s) -> int:
    return int(re.search(r"ERA5_(\d{4})_", s).group(1))


def listDat(path=args.path_data) -> pd.DataFrame:
    # List all the netcdf datasets in the t2 (2m temperature) and
    # rh (relative humidity) directories
    dat2t = glob1(f"{path}/2t/", "ERA5_*.nc")
    datrh = glob1(f"{path}/rh/", "ERA5_*.nc")
    # Create a pandas dataframe to hold the info for the netcdf datasets
    dat_ = pd.DataFrame({
        "variable": "t2m",
        "year": [getYear(x) for x in dat2t],
        "path": [f"2t/{x}" for x in dat2t]})
    dat_ = pd.concat([dat_, pd.DataFrame({
        "variable": "rh",
        "year": [getYear(x) for x in datrh],
        "path": [f"rh/{x}" for x in datrh]})])
    return dat_.sort_values(by=["variable", "year"]).reset_index(drop=True)


def THICalc(dfDat, pathDat=args.path_data, year=args.year) -> None:
    # Retrieve the path to the t2m and rh datasets corresponding to the
    # specified year
    t2m = dfDat.loc[(dfDat.variable == "t2m") &
                    (dfDat.year == year)].path.item()
    rh = dfDat.loc[(dfDat.variable == "rh") &
                   (dfDat.year == year)].path.item()
    # First merge the 2m temperature and relative humidity datasets
    ds = xr.open_dataset(t2m).merge(xr.open_dataset(rh))
    # Convert temperature to degrees Celcius
    ds["t2m"] = ds["t2m"] - 273.15
    gc.collect()
    # Calculate the Temperature Humidity Index
    # THI = (1.8 × T + 32) − (0.55 − 0.0055 × RH) × (1.8 × T − 26)
    thi = ((1.8*ds["t2m"]) + 32) - ((0.55 - (0.0055 * ds["rh"])) *
                                    ((1.8*ds["t2m"]) - 26))
    del ds
    gc.collect()
    pathSave = f"{pathDat}/thi"
    os.makedirs(pathSave, exist_ok=True)
    # Save it
    thi.to_dataset(name="THI")\
        .to_netcdf(f"{pathSave}/ERA5_{year}_THI.nc",
                   encoding={"THI": {"zlib": True,
                                     "complevel": 6,
                                     "dtype": np.float32}})
    return None


if __name__ == "__main__":
    dfDat = listDat()
    THICalc(dfDat)
