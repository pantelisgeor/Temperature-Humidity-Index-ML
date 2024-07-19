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
    # 2d (2m dewpoint temperature) directories
    dat2t = glob1(f"{path}/2t/", "ERA5_*.nc")
    dat2d = glob1(f"{path}/2d/", "ERA5_*.nc")
    # Create a pandas dataframe to hold the info for the netcdf datasets
    dat_ = pd.DataFrame({
        "variable": "t2m",
        "year": [getYear(x) for x in dat2t],
        "path": [f"2t/{x}" for x in dat2t]})
    dat_ = pd.concat([dat_, pd.DataFrame({
        "variable": "d2m",
        "year": [getYear(x) for x in dat2d],
        "path": [f"2d/{x}" for x in dat2d]})])
    return dat_.sort_values(by=["variable", "year"]).reset_index(drop=True)


def RHCalc(dfDat, pathDat=args.path_data, year=args.year) -> None:
    # Retrieve the path to the t2m and d2m datasets corresponding to the
    # specified year
    t2m = dfDat.loc[(dfDat.variable == "t2m") &
                    (dfDat.year == year)].path.item()
    d2m = dfDat.loc[(dfDat.variable == "d2m") &
                    (dfDat.year == year)].path.item()
    # First merge the 2m temperature and 2m dewpoint temperature
    ds = xr.open_dataset(t2m).merge(xr.open_dataset(d2m))
    gc.collect()
    # Calculate the relative humidity
    e_Td = 6.1078 * np.exp(
        (17.1 * (ds["d2m"]-273.15)) / (235 + (ds["d2m"]-273.15)))
    es_T = 6.1078 * np.exp(
        (17.1 * (ds["t2m"]-273.15)) / (235 + (ds["t2m"]-273.15)))
    del ds
    gc.collect()
    rh = 100 * (e_Td/es_T)
    del e_Td, es_T
    gc.collect()
    pathSave = f"{pathDat}/rh"
    os.makedirs(pathSave, exist_ok=True)
    # Save it
    rh.to_dataset(name="rh")\
        .to_netcdf(f"{pathSave}/ERA5_{year}_relative_humidity.nc",
                   encoding={"rh": {"zlib": True,
                                    "complevel": 5,
                                    "dtype": np.float32}})
    return None


if __name__ == "__main__":
    dfDat = listDat()
    RHCalc(dfDat)
