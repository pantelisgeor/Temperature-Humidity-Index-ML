#!/bin/bash

Help()
{
   # Display Help
   echo
   echo "This bash file downloads the ERA5 and CMIP6 datasets needed."
   echo
   echo "-- The first argument for the script corresponds to the directory where the ERA5 datasets are stored"
   echo "   Make sure you have an active Copernicus Data Store (CDS) account set up."
   echo "-- The second argument to the directory where the CMIP6 datasets are stored."
   echo "Example use: ./download_data.sh ~/ERA5 ~/CMIP6"
   echo
}

while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
   esac
done

# First download the ERA5 data for the years 1980-2020
python download_ERA.py --path_data=$1

# The calculate the Relative Humidity and THI index variables for all the years
for (( year=1980; year<=2020; year++ ))
do
    python rhCalc.py --path_data=$1 --year=$year
    python calcTHI.py --path_data=$1 --year=$year
done

# Download the CMIP6 datasets
python CMIP6.sh --path_data=$2