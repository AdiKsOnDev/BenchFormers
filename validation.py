import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from include.models.roformer import RoformerModel
from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel
from include.utils import check_cuda, parse_validation_arguments

check_cuda()
device = torch.device("cuda")
args = parse_validation_arguments()

directory = args.dir
choice = args.model
df = pd.read_csv("./data/preprocessed/p_validation.csv")
num_labels = len(df["label"].unique())

models = []

if choice.lower() == "roformer":
    models.append(
        RoformerModel(model_name=f"{directory}/junnyu/roformer_chinese_base/fine_tuned_junnyu/roformer_chinese_base/",
                        num_labels=num_labels),
    )
elif choice.lower() == "longformer":
    models.append(
        LongformerModel(model_name=f"{directory}/allenai/longformer-base-4096/fine_tuned_allenai/longformer-base-4096/",
                        num_labels=num_labels, max_length=4096),
    )
elif choice.lower() == "bigbird":
    models.append(
        BigBirdModel(model_name=f"{directory}/google/bigbird-roberta-base/fine_tuned_google/bigbird-roberta-base/",
                     num_labels=num_labels, max_length=4096),
    )
elif choice.lower() == "legalbert":
    models.append(
        LegalBERTModel(model_name=f"{directory}/nlpaueb/legal-bert-base-uncased/fine_tuned_nlpaueb/legal-bert-base-uncased/",
                       num_labels=num_labels)
    )


texts = df["text"].tolist()
labels = df["label"].tolist()

for model in models:
    model.model = model.model.to(device)
    predictions = model.predict(texts)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    f1= f1_score(labels, predictions, average="macro")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1_Score": f1,
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
