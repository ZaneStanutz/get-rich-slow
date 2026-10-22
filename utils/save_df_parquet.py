import os

def save_dataframe_to_parquet(df, folder_path, file_name, engine='auto', compression='snappy', index=True, partition_cols=None):
    """
    Save a pandas DataFrame to a parquet file at the specified path.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to save
    folder_path : str
        folder path where the paquet should be saved.
    file_name : str
        file name of the desired parquet
    engine : str, default 'auto'
        Parquet library to use ('auto', 'pyarrow', 'fastparquet')
    compression : str or dict, default 'snappy'
        Compression to use ('snappy', 'gzip', 'brotli', None)
    index : bool, default True
        Whether to include the DataFrame index in the parquet file
    partition_cols : list, default None
        Column names to partition by
    
    Returns:
    --------
    None
    """
    # Create the directory if it doesn't exist
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    full_path = os.path.join(folder_path, file_name)

    # Save the DataFrame to parquet
    df.to_parquet(
        full_path,
        engine=engine,
        compression=compression,
        index=index,
        partition_cols=partition_cols
    )
    
    print(f"DataFrame successfully saved to {full_path}")