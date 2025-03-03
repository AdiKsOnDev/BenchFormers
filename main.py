import os
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from include.preprocessing import preprocess_text
from include.fine_tuning import fine_tune
from include.utils import limit_dataset, check_cuda, parse_arguments, models
from include.Dataset import Dataset

args = parse_arguments()
main_logger = logging.getLogger('main')
include_logger = logging.getLogger('include')
models_logger = logging.getLogger('models')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
tqdm.pandas()

preprocessed_file = "./data/preprocessed.csv"
validation_file = "./data/validation.csv"
dataset_file = "./data/dataset.csv"

label_encoder = LabelEncoder()
dataset_size = args.dataset_size
results_dir = args.results_dir
choice = args.model
log_level = logging.INFO

logging.basicConfig(
    level=logging.WARNING,
    format="%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
main_logger.setLevel(
    log_level
)
include_logger.setLevel(
    log_level
)
models_logger.setLevel(
    log_level
)

check_cuda()

if not os.path.exists(preprocessed_file):
    main_logger.warning(
        f"{preprocessed_file} not found. Preprocessing dataset...")

    df = pd.read_csv(dataset_file)
    main_logger.info(f"Dataset loaded from {dataset_file}")
    main_logger.info(f"Preprocessing dataset...")
    df["text"] = df["text"].apply(preprocess_text)
    main_logger.info(f"Dataset preprocessed.")
    os.makedirs("./data", exist_ok=True)
    df.to_csv(preprocessed_file, index=False)

    main_logger.info(f"Preprocessed dataset saved to {preprocessed_file}.")
else:
    main_logger.warning(f"Preprocessed file {preprocessed_file} already exists. Skipping preprocessing.")
    df = pd.read_csv(preprocessed_file)

df = limit_dataset(df, dataset_size)
num_labels = len(df["label"].unique())

df, validation_df = train_test_split(
    df, test_size=0.25, random_state=42, stratify=df["label"]
)

validation_df.to_csv(validation_file)

models = models(choice, num_labels)

for model in models:
    main_logger.debug(f"Started the pipeline for {model.model_name}")

    df["label"] = label_encoder.fit_transform(df["label"])

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    train_X, test_X, train_y, test_y = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    main_logger.info(f"Tokenizing for {model.model_name}")

    train_X = model.tokenize(train_X)
    test_X = model.tokenize(test_X)
    train_dataset = Dataset(train_X, train_y)
    test_dataset = Dataset(test_X, test_y)

    main_logger.debug(f"About to start fine-tuning {model.model_name}")
    fine_tune(model, train_dataset, test_dataset, results_dir)
