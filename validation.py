import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset

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

""" Evaluation of each model

    TODO: Move this (or part of this) to BaseModel
"""
for model in models:
    tokenized_data = model.tokenize(texts)

    input_ids = torch.tensor(tokenized_data["input_ids"])
    attention_mask = torch.tensor(tokenized_data["attention_mask"])

    dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=4)

    model = model.model.to(device)
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        print("Predicting")
        for batch in dataloader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(batch_predictions)
            all_true_labels.extend(batch_labels.cpu().numpy())

    decoded_predictions = label_encoder.inverse_transform(all_predictions)
    decoded_true_labels = label_encoder.inverse_transform(all_true_labels)

    print(classification_report(decoded_true_labels, decoded_predictions))
