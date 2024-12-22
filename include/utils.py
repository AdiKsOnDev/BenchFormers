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
