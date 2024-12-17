import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from include.preprocessing import preprocess_text
from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel
from include.fine_tuning import fine_tune
from include.Dataset import Dataset

tqdm.pandas()
preprocessed_file = "./data/preprocessed.csv"
dataset_file = "./data/dataset.csv"
label_encoder = LabelEncoder()

if torch.cuda.is_available():
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

if not os.path.exists(preprocessed_file):
    print(f"{preprocessed_file} not found. Preprocessing dataset...")

    df = pd.read_csv(dataset_file)

    df["text"] = df["text"].apply(preprocess_text)

    os.makedirs("./data", exist_ok=True)
    df.to_csv(preprocessed_file, index=False)

    print(f"Preprocessed dataset saved to {preprocessed_file}.")
else:
    print(f"Preprocessed file {
          preprocessed_file} already exists. Skipping preprocessing.")
    df = pd.read_csv(preprocessed_file)

df = pd.read_csv(dataset_file)
df = df[:10]
num_labels = len(df["label"].unique())

models = [
    LongformerModel(model_name="allenai/longformer-base-4096",
                    num_labels=num_labels, max_length=4096),
    BigBirdModel(model_name="google/bigbird-roberta-base",
                 num_labels=num_labels, max_length=4096),
    LegalBERTModel(model_name="nlpaueb/legal-bert-base-uncased",
                   num_labels=num_labels)
]

for model in models:
    print(f"Tokenizing for {model.model_name}")

    df["label"] = label_encoder.fit_transform(df["label"])

    texts = df["text"].tolist()
    labels = df["label"].tolist()


    train_X, test_X, train_y, test_y = train_test_split(
        texts, labels, test_size=0.8, random_state=42, stratify=labels
    )

    train_X = model.tokenize(train_X)
    test_X = model.tokenize(test_X)
    train_dataset = Dataset(train_X, train_y)
    test_dataset = Dataset(test_X, test_y)

    fine_tune(model, train_dataset, test_dataset)
