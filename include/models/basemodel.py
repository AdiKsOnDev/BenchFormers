from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseModel:
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self._load_model(model_name, num_labels)

    def _load_model(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

    def tokenize(self, texts, max_length):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def forward(self, inputs):
        return self.model(**inputs)
