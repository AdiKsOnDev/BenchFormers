import torch

def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
        exit()

def limit_dataset(df, size):
    """
    Returns a dataset with a specified size after
    shuffling it using df.sample()
    random_state is set to 42

    Args:
        df (pd.Dataframe): Dataframe to be modified
        size (int): Size to be limited to

    Return:
        pd.Dataframe
    """
    return df.sample(frac=1, random_state=42).reset_index(drop=True)[:size]
