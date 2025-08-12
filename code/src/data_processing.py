import pandas as pd;

import numpy as np

"""
Full processing pipeline
"""
def process(file_path):
    data = __load_data(file_path); 
    
    # Drop useless error column
    data = data.drop(columns = ["error"]);
    
    X_label = "wave_obs_AA";
    y_label = "residual";
    training_data = data.dropna(subset = ["wave_obs_AA", "residual"]);
    
    return (
        data,
        np.asarray(training_data[X_label], dtype = float).reshape(-1, 1),
        np.asarray(training_data[y_label], dtype = float).reshape(-1, 1)
    )

"""
Load and fix data
"""
def __load_data(file_path):
    data = pd.read_csv(file_path);
    
    # All values apart from last are mashed into multi-column index
    data = data.reset_index();

    # All column names are mashed into last column name
    last_col_name = data.columns.values[-1];
    all_col_names = last_col_name.split(" ")[1:];
    data.columns = all_col_names; 
    
    return data;
