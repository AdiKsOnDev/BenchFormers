import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import main, tqdm

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

train_file = "./data/train.csv"
test_file = "./data/test.csv"
validation_file = "./data/validation.csv"

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

if not os.path.exists('./data/preprocessed/'):
    main_logger.warning(
        f"Preprocessed files not found. Preprocessing dataset...")

    for dataset_file in [train_file, test_file, validation_file]:
        df = pd.read_csv(dataset_file)
        main_logger.info(f"Dataset loaded from {dataset_file}")
        main_logger.info(f"Preprocessing dataset...")
        df["text"] = df["text"].apply(preprocess_text)

        df["label"] = label_encoder.fit_transform(df['label'])
        main_logger.info(f"Dataset preprocessed.")

        os.makedirs("./data/preprocessed/", exist_ok=True)
        df.to_csv(f"./data/preprocessed/p_{dataset_file.split('/')[2]}", index=False)
        main_logger.info(f"Preprocessed dataset saved to ./data/preprocessed/p_{dataset_file.split('/')[2]}.")
else:
    main_logger.warning(f"Preprocessed files already exists. Skipping preprocessing.")

train_df = limit_dataset(pd.read_csv(f"./data/preprocessed/p_{train_file.split('/')[2]}"), dataset_size)
test_df = pd.read_csv(f"./data/preprocessed/p_{test_file.split('/')[2]}")

num_labels = len(train_df['label'].unique())
models = models(choice, num_labels)

for model in models:
    main_logger.debug(f"Started the pipeline for {model.model_name}")

    train_y = train_df['label'].to_list()
    test_y = test_df['label'].to_list()

    main_logger.info(f"Tokenizing for {model.model_name}")

    train_X = model.tokenize(train_df['text'].to_list())
    test_X = model.tokenize(test_df['text'].to_list())
    train_dataset = Dataset(train_X, train_y)
    test_dataset = Dataset(test_X, test_y)

    main_logger.debug(f"About to start fine-tuning {model.model_name}")
    fine_tune(model, train_dataset, test_dataset, results_dir)
