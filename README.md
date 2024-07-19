# The Application of Machine Learning Algorithms to the Global Forecast of Temperature-Humidity Index with High Temporal Resolution

Pantelis Georgiades, Theo Economou, Yiannis Proestos, Jose Araya, Jos Lelieveld and Marco Neira

**Correspondence:** Pantelis Georgiades (p.georgiades@cyi.ac.cy)

This repository contains the data and source code to produce the results presented in:

# ADD PAPER CITATION

## Abstract

Climate change poses a significant threat to agriculture, with potential impacts on food security, economic stability, and human livelihoods. Dairy cattle, a crucial component of the livestock sector, are particularly vulnerable to heat stress, which can adversely affect milk production, immune function, feed intake, and in extreme cases, lead to mortality. The Temperature Humidity Index (THI) is a widely used metric to quantify the combined effects of temperature and humidity on cattle. However, most studies estimate THI using daily-level data, which fails to capture the full extent of daily thermal load and cumulative heat stress, especially during nights when cooling is inadequate. To address this limitation, we developed a machine learning approach to temporally downscale daily climate data to hourly THI values. Utilizing historical ERA5 reanalysis data, we trained an XGBoost model and generated hourly THI datasets for 12 NEX-GDDP-CMIP6 climate models under two emission scenarios (SSP2-4.5 and SSP5-8.5) extending to the end of the century. This high-resolution THI data provides a more accurate assessment of heat stress in dairy cattle, enabling better predictions and management strategies to mitigate the impacts of climate change on this vital agricultural sector.

## Instructions:

You can download a copy of all the files in this repository by cloning the git repository:

```
git clone github.com/pantelisgeor/Temperature-Humidity-Index-ML
```

**Note** The code provided was developed and tested on a node equipped node with 256 GB of RAM and 2 AMD EPYC Milan 64 core CPUs, running Linux.

### Setting up your environment

You'll need a working Python 3 environment with the following libraries:
1. Standard Scientific libraries: numpy, pandas, scipy, matplotlib, cartopy, scikit-learn, pyarrow.
2. XGBoost library: xgboost
3. Spatial data: xarray, netcdf4, rasterio
4. Other libraries: tqdm, python-wget, cdsapi

#### Data Retrieval

The code is written exclusively in Python and uses a number of bash scripts to execute the workload. First, to retrieve the data needed *data.sh* is called. The bash script takes two positional arguments:
1. The first argument corresponds to the directory where the ERA5 datasets are to be stored.
2. The second argument corresponds to the directory where the CMIP6 datasets are to be stored.

After the script retrieves the ERA5 datasets for temperature and dewpoint temperature, it calls the rhCalc.py and calcTHI.py python scripts. These will calculate the *Relative Humidity* (RH) and *Temperature Humidity Index* (THI) from the ERA5 datasets for the time period 1980-2020.

To execute the bash script, run the following commands:

```
cd code
chmod +x download_data.sh
./download_data.sh ~/Data_ERA5 ~/Data_CMIP6
```

**Note** The bash script downloads and processes several GBs of data. Make sure you have the appropriate compute and storage capabilities!

#### Training the model

After the data retrieval and processing procedure are finished, to train the XGBoost model the *train_model.sh* bash script is used. The bash script calls the python script to train the model and takes two positional arguments:

1. The path to the ERA5 data directory.
2. The path to the directory where the model states are stored.

To execute the bash script, run the following commands:

```
chmod +x train_model.sh
./train_model.sh ~/Data_ERA5 ~/models
```

#### Inference

To use the trained model for temporally downscaling the GDDP-NEX-CMIP6 datasaets to hourly *Temperature Humidity Index* values, the *predict.sh* bash script is called. The bash script takes four positional arguments:

1. Path to directory where the THI predictions are to be stored.
2. Path to the directory where the trained model is stored (the last checkpoint will be automatically loaded from there).
3. Path to the directory where CMIP6 data are stored.
4. Path to directory used during training.

To run the bash script execute the following commands:

```
chmod +x predich.sh
./predict.sh ~/predictions ~/models/models/xgb_incremental ~/Data_CMIP6 ~/models
```

The bash script will loop through the 12 models used in the paper and two emission scenarios (SSP2-45 and SSP5-85) and make predictions in 5 year batches.

## License

All Python source code is made available under the MIT license. You can freely use and modify the code, without warranty, so long as you provide attributions to the authors. See 'LICENSE-MIT.txt' for the full license text.

The manuscript text, figures and data/models produced as part of this research are available under the [Creative Commons Attribution 4.0 License (CC-BY)][cc-by]. See `LICENSE-CC-BY.txt` for the full license text.

[cc-by]: https://creativecommons.org/licenses/by/4.0/
