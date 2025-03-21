import logging
import argparse
import torch

from include.models.roformer import RoformerModel
from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel
from include.models.modernbert import ModernBERT

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


def models(choice, num_labels):
    models = [
        RoformerModel(model_name="junnyu/roformer_chinese_base",
                      num_labels=num_labels),
        LongformerModel(model_name="allenai/longformer-base-4096",
                        num_labels=num_labels, max_length=4096),
        BigBirdModel(model_name="google/bigbird-roberta-base",
                     num_labels=num_labels, max_length=4096),
        LegalBERTModel(model_name="nlpaueb/legal-bert-base-uncased",
                       num_labels=num_labels),
        ModernBERT(model_name="answerdotai/ModernBERT-base",
                       num_labels=num_labels, max_length=4096)
    ]

    if choice.lower() == "all":
        return models
    elif choice.lower() == "roformer":
        return [models[0]]
    elif choice.lower() == "longformer":
        return [models[1]]
    elif choice.lower() == "bigbird":
        return [models[2]]
    elif choice.lower() == "legalbert":
        return [models[3]]
    elif choice.lower() == "modernbert":
        return [models[4]]
    else:
        raise ValueError(f"{choice} is not a valid choice of models!")


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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Can be: 1. roformer 2. longformer 3. bigbird 4. legalbert"
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
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Can be: 1. roformer 2. longformer 3. bigbird 4. legalbert 5. all"
    )
    args = parser.parse_args()

    return args
