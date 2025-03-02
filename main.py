import os
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from include.preprocessing import preprocess_text
from include.models.roformer import RoformerModel
from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel
from include.fine_tuning import fine_tune
from include.utils import limit_dataset, check_cuda, parse_arguments
from include.Dataset import Dataset

args = parse_arguments()
main_logger = logging.getLogger('main')
include_logger = logging.getLogger('include')
models_logger = logging.getLogger('models')
tqdm.pandas()

preprocessed_file = "./data/preprocessed.csv"
validation_file = "./data/validation.csv"
dataset_file = "./data/dataset.csv"

label_encoder = LabelEncoder()
dataset_size = args.dataset_size
results_dir = args.results_dir
log_level = logging.DEBUG

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

    df["text"] = df["text"].apply(preprocess_text)

    os.makedirs("./data", exist_ok=True)
    df.to_csv(preprocessed_file, index=False)

    main_logger.info(f"Preprocessed dataset saved to {preprocessed_file}.")
else:
    main_logger.warning(f"Preprocessed file {
                        preprocessed_file} already exists. Skipping preprocessing.")
    df = pd.read_csv(preprocessed_file)

df = limit_dataset(df, dataset_size)
num_labels = len(df["label"].unique())

df, validation_df = train_test_split(
    df, test_size=0.25, random_state=42
)

validation_df.to_csv(validation_file)

models = [
    RoformerModel(model_name="junnyu/roformer_chinese_base",
                  num_labels=num_labels),
    LongformerModel(model_name="allenai/longformer-base-4096",
                    num_labels=num_labels, max_length=4096),
    BigBirdModel(model_name="google/bigbird-roberta-base",
                 num_labels=num_labels, max_length=4096),
    LegalBERTModel(model_name="nlpaueb/legal-bert-base-uncased",
                   num_labels=num_labels)
]

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
