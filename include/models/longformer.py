from transformers import LongformerTokenizer, LongformerForSequenceClassification

from include.models.basemodel import BaseModel


class LongformerModel(BaseModel):
    def _load_model(self, model_name, num_labels):
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
