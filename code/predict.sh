#!/bin/bash

# Declare an array of strings with the models
declare -a StringArray=("ACCESS-ESM1-5" "CMCC-CM2-SR5" "EC-Earth3" 
                        "EC-Earth3-Veg-LR" "FGOALS-g3" "GFDL-CM4" 
                        "GFDL-ESM4" "INM-CM4-8" "INM-CM5-0"
                        "MIROC6" "MRI-ESM2-0" "NorESM2-MM")

for model in ${StringArray[@]};
do
    for scenario in "ssp245" "ssp585"
    do
        for (( year=2020; year<2100; year+=4 ))
        do
            python predict.py \
                --model=$model \
                --scenario=$scenario \
                --startYear=$year \
                --endYear=$(($year+4)) \
                --pathSave=$1 \
                --pathModel=$2 \
                --path_data=$3 \
                --root=$4 \
                --scaler_feats=scaler_feats.joblib \
                --scaler_target=scaler_target.joblib  
        done
    done
done
