#!/bin/bash

for (( year=1980; year<=2020; year++ ))
do
    for (( month=1; month<=12; month++))
    do
      # Make sure to fill in the correct paths below.
      python trainModel.py \
         --path_data=$1 \
         --root=$2 \
         --model_name=xgb_incremental_1 \
         --scaler_feats=scaler_feats.joblib \
         --scaler_target=scaler_target.joblib
    done
done