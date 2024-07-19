import os
import pandas as pd
import argparse
import warnings
import re
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from glob import glob1
import wget

# -------------------------------------------------------------------------- #
warnings.filterwarnings("ignore")
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to directory where ERA5 data will be saved")
args = parser.parse_args()
# Switch to working directory
os.chdir(args.path_data)
# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
# Function definitions
def parseName(x) -> pd.DataFrame:
    # AWS services first part of url
    url_ = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/"
    # Model details
    model = x.split("/")[1]
    scenario = x.split("/")[2]
    variable = x.split("/")[4]
    year = int(re.search(r"_(\d{4})", x).group(1))
    return pd.DataFrame({"model": [model], "scenario": [scenario],
                         "variable": [variable], "year": [year],
                         "url": [f"{url_}{x}"]})
    
    
def downloadSingle(url) -> None:
    if os.path.isfile(url.split("/")[-1]):
        return None
    wget.download(url)
    return None


def downloadCMIP6(datList, modelName, path_data, parallel=True) -> None:
    # Construct a new directory for the CMIP6 model if it doesn't exist
    os.makedirs(f"{path_data}/{modelName}", exist_ok=True)
    os.chdir(f"{path_data}/{modelName}")
    # Download the data
    if parallel:
        process_map(downloadSingle, datList,
                    max_workers=5, chunksize=1)
    else:
        [downloadSingle(x) for x in tqdm(datList)]
    if len(glob1(f"{path_data}/{modelName}", ".tmp")) > 0:
        os.system(f"cd {path_data}/{modelName} && rm *.tmp")
    return None


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    # Read the NEX-GDDP data index
    df = pd.read_csv("index_md5.txt", sep="  ", header=None)\
        .rename(columns={0: "MD5", 1: "filename"})
    df_ = pd.concat([parseName(x) for x in tqdm(df.filename.values)])
    df_ = df_.sort_values(by=["model", "scenario", "year"])\
        .reset_index(drop=True)
    del df

    # NEX-GDDP models used in this study
    models = ["ACCESS-ESM1-5", "CMCC-CM2-SR5", "EC-Earth3", "EC-Earth3-Veg-LR",
              "FGOALS-g3", "GFDL-CM4", "GFDL-ESM4", "INM-CM4-8", "INM-CM5-0",
              "MIROC6", "MRI-ESM2-0", "NorESM2-MM"]
    # Variables needed
    vars_ = ["hurs", "tasmax", "tasmin", "tas"]
    # Scenarios
    scenarios = ["ssp245", "ssp585"]

    # Subset the dataframe for the models defined above and years 2020-2100
    df_ = df_.loc[(df_.model.isin(models)) & (df_.year >= 2020) &
                (df_.year <= 2100) & (df_.variable.isin(vars_)) &
                (df_.scenario.isin(scenarios))]\
        .reset_index(drop=False)

    # Loop through the models and download the associated netcdf files for eac
    for model in models:
        datList = list(df_.loc[df_.model == model]["url"].values)
        downloadCMIP6(datList=datList, modelName=model, path_data=path_data,
                    parallel=True)
