import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import torch

from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel

label_encoder = LabelEncoder()
device = torch.device("cuda")

df = pd.read_csv("./data/validation.csv")
num_labels = len(df["label"].unique())

models = [
    LongformerModel(model_name="./results/allenai/longformer-base-4096/fine_tuned_allenai/longformer-base-4096/",
                    num_labels=num_labels, max_length=4096),
    BigBirdModel(model_name="./results/google/bigbird-roberta-base/fine_tuned_google/bigbird-roberta-base/",
                 num_labels=num_labels, max_length=4096),
    LegalBERTModel(model_name="./results/nlpaueb/legal-bert-base-uncased/fine_tuned_nlpaueb/legal-bert-base-uncased/",
                   num_labels=num_labels)
]
df["label"] = label_encoder.fit_transform(df["label"])

texts = df["text"].tolist()
labels = df["label"].tolist()

for model in models:
    model.model = model.model.to(device)
    predictions = model.predict(texts)

    decoded_predictions = label_encoder.inverse_transform(predictions)
    decoded_true_labels = label_encoder.inverse_transform(labels)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
