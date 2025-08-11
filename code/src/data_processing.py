import pandas as pd;
"""
Full processing pipeline
"""
def process(file_path):
    data = __load_data(file_path); 

    return data;

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
