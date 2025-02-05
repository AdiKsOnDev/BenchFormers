import logging
from transformers import LongformerTokenizer, LongformerForSequenceClassification

from include.models.basemodel import BaseModel

models_logger = logging.getLogger('models')

class LongformerModel(BaseModel):
    def _load_model(self, model_name, num_labels):
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        models_logger.debug(f"Tokenizer and the model for {self.model_name} are initialised")
