import os
import pandas as pd
import numpy as np
import re
import resampy
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

def process_timeseries(cfg):
    """
    Processes timeseries data based on specified sensor setups and saves the modified data.

    Parameters:
    - cfg: Configuration object with paths and other settings.
    """
    columns = [
        'wrist_r__acc_x', 'wrist_r__acc_y', 'wrist_r__acc_z',
        'wrist_r__gyro_x', 'wrist_r__gyro_y', 'wrist_r__gyro_z',
        'wrist_l__acc_x', 'wrist_l__acc_y', 'wrist_l__acc_z',
        'wrist_l__gyro_x', 'wrist_l__gyro_y', 'wrist_l__gyro_z'
    ]
    measurement_to_day = {
        'm1': 'day3',
        'm2': 'day9',
        'm3': 'day28',
        'm4': 'day90',
        'm5': 'day365'
    }
    
    # Open HDFStore
    # hdf = pd.HDFStore(cfg.hdf_path)
    
    # Read FMA clinical data
    df_fma = pd.read_csv(cfg.fma_clinical_path)
    
    # Calculate total number of files to process for the progress bar
    total_files = 0
    for setup in cfg.sensor_setups:
        input_folder = os.path.join(cfg.timeseries_path, setup)
        total_files += len([
            filename for filename in os.listdir(input_folder)
            if filename.endswith('.csv') and not filename.endswith('_AC.csv') and 
               os.path.exists(os.path.join(input_folder, filename[:-4] + "_AC.csv"))
        ])
    
    # Initialize the progress bar
    with tqdm(total=total_files, desc="Processing Files", unit="file") as pbar:
        for setup in cfg.sensor_setups:
            input_folder = os.path.join(cfg.timeseries_path, setup)
            print('Start setup: ', setup)
            # List all relevant files in the current setup
            files = [
                filename for filename in os.listdir(input_folder)
                if filename.endswith('.csv') and not filename.endswith('_AC.csv') and 
                   os.path.exists(os.path.join(input_folder, filename[:-4] + "_AC.csv"))
            ]

            for filename in files:
                file_path = os.path.join(input_folder, filename)
                
                # Parse filename
                parts = filename.split('_')
                offset = 0
                if setup in ['all', 'no_chest']:
                    offset = 1
                
                patient_id = 'RU' + parts[0]
                visit_day = measurement_to_day[parts[1]]
                measurement = parts[2]
                affected_side = parts[4 + offset]  # 'right' or 'left'
                dom_affected = parts[5 + offset]

                # Create the HDF5 key
                # hdfKey = f'/{visit_day}/{patient_id}_{measurement}'
                # if hdfKey in hdf.keys():
                #     pbar.update(1)
                #     continue
                
                # # Check if FMA score is available
                # row = df_fma[(df_fma['Pat_id'] == patient_id) & (df_fma['visit'] == visit_day)]
                # if row.empty:
                #     pbar.update(1)
                #     continue  # Skip processing if no FMA score available

                # # Attributes
                # attrs = {
                #     'Pat_id': patient_id,
                #     'Visit': visit_day,
                #     'Visit_day': float(re.findall(r'\d+', visit_day)[0]),
                #     'Measurement': measurement,
                #     'Setup': setup,
                #     'Affected_Side': affected_side,
                #     'Gender': row['gender'].values[0],
                #     'Age': row['age'].values[0],
                #     'Dom_affected': row['dom_aff'].values[0],
                #     'FMA_score': row['fma_ue_aff_total'].values[0]
                # }

                # # Read the _AC.csv file
                # df_AC = pd.read_csv(file_path[:-4] + "_AC.csv")
                # df = pd.DataFrame()

                # # Determine functional slices
                # df['functional'] = (df_AC['AC_aff'] > 2) | (df_AC['AC_nonaff'] > 2)
                # del df_AC

                # # Read the main CSV file
                # df_ts = pd.read_csv(file_path)
                # for col in columns:
                #     resampled_data = resampy.resample(np.asarray(df_ts[col]), 50, 1)
                    
                #     # Adjust the length if necessary
                #     if len(resampled_data) > len(df):
                #         resampled_data = resampled_data[:len(df)]
                #     elif len(resampled_data) < len(df):
                #         resampled_data = np.pad(resampled_data, (0, len(df) - len(resampled_data)), 'constant', constant_values=np.nan)
                    
                #     df[col] = resampled_data  # Assign safely
                # del df_ts

                # Rename columns based on affectedness
                if affected_side == 'right':
                    print(f"Checkpoint A {affected_side}")
                    # df.rename(columns={
                    #     'wrist_r__acc_x': 'wrist_aff__acc_x',
                    #     'wrist_r__acc_y': 'wrist_aff__acc_y',
                    #     'wrist_r__acc_z': 'wrist_aff__acc_z',
                    #     'wrist_r__gyro_x': 'wrist_aff__gyro_x',
                    #     'wrist_r__gyro_y': 'wrist_aff__gyro_y',
                    #     'wrist_r__gyro_z': 'wrist_aff__gyro_z',
                    #     'wrist_l__acc_x': 'wrist_nonaff__acc_x',
                    #     'wrist_l__acc_y': 'wrist_nonaff__acc_y',
                    #     'wrist_l__acc_z': 'wrist_nonaff__acc_z',
                    #     'wrist_l__gyro_x': 'wrist_nonaff__gyro_x',
                    #     'wrist_l__gyro_y': 'wrist_nonaff__gyro_y',
                    #     'wrist_l__gyro_z': 'wrist_nonaff__gyro_z'
                    # }, inplace=True)
                elif affected_side == 'left':
                    print(f"Checkpoint B {affected_side}")
                    # df.rename(columns={
                    #     'wrist_l__acc_x': 'wrist_aff__acc_x',
                    #     'wrist_l__acc_y': 'wrist_aff__acc_y',
                    #     'wrist_l__acc_z': 'wrist_aff__acc_z',
                    #     'wrist_l__gyro_x': 'wrist_aff__gyro_x',
                    #     'wrist_l__gyro_y': 'wrist_aff__gyro_y',
                    #     'wrist_l__gyro_z': 'wrist_aff__gyro_z',
                    #     'wrist_r__acc_x': 'wrist_nonaff__acc_x',
                    #     'wrist_r__acc_y': 'wrist_nonaff__acc_y',
                    #     'wrist_r__acc_z': 'wrist_nonaff__acc_z',
                    #     'wrist_r__gyro_x': 'wrist_nonaff__gyro_x',
                    #     'wrist_r__gyro_y': 'wrist_nonaff__gyro_y',
                    #     'wrist_r__gyro_z': 'wrist_nonaff__gyro_z',
                    # }, inplace=True)
                else:
                    raise ValueError('Affected side missing or invalid in filename.')

                # Save the modified dataset in the HDF5 store
                #hdf.put(hdfKey, df, format='fixed', data_columns=True)
                #hdf.get_storer(hdfKey).attrs.my_attribute = attrs

                # Update the progress bar
                pbar.update(1)
    
    # Final information about the HDFStore
    # print(hdf.info())
    # hdf.close()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    process_timeseries(cfg)

if __name__ == "__main__":
    main()
