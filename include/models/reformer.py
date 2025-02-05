import logging
from transformers import AutoTokenizer, ReformerForSequenceClassification

from include.models.basemodel import BaseModel

models_logger = logging.getLogger('models')

class ReformerModel(BaseModel):
    def _load_model(self, model_name, num_labels):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ReformerForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        models_logger.debug(f"Tokenizer and the model for {self.model_name} are initialised")
