import logging
import argparse
import torch

include_logger = logging.getLogger('include')

def check_cuda():
    if torch.cuda.is_available():
        include_logger.info(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
    else:
        include_logger.error("CUDA is not available.")
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
    include_logger.debug(f"Limiting dataset dow to {size} samples")
    return df.sample(frac=1, random_state=42).reset_index(drop=True)[:size]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for a data processing task.")
    parser.add_argument(
        "--dataset_size",
        type=int,
        required=True,
        help="The size of the dataset."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="The directory to save the models."
    )
    args = parser.parse_args()

    return args

def parse_validation_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for a data processing task.")

    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="The directory of the fine-tuned models."
    )

    args = parser.parse_args()
    return args
