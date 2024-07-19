import cdsapi
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_dat", type=str,
                    help="Path to directory where ERA5 data will be saved")
args = parser.parse_args()

# Initiate the cds client
c = cdsapi.Client()

names = {"2m_dewpoint_temperature": "2d",
         "2m_temperature": "2t"}

for i, v_ in enumerate(names.keys()):
    path_save = f"{args.path_dat}{names[v_]}"
    os.makedirs(path_save, exist_ok=True)
    for year in range(1980, 2021, 1):
        print(f"\nDownloading {names[v_]} for year {year}.\n\n")
        if os.path.isfile(f"{path_save}/ERA5_{year}_MET-VARIABLES_global_var_{v_}.nc"):
            continue
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': v_,
                    'year': str(year),
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31', ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',],
                },
                f"{path_save}/ERA5_{year}_MET-VARIABLES_global_var_{v_}.nc")
        except Exception as e:
            print(e)
            continue
