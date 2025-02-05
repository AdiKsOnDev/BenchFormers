import logging
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification

from include.models.basemodel import BaseModel

models_logger = logging.getLogger('models')

class BigBirdModel(BaseModel):
    def _load_model(self, model_name, num_labels):
        self.tokenizer = BigBirdTokenizer.from_pretrained(model_name)
        self.model = BigBirdForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        models_logger.debug(f"Tokenizer and the model for {self.model_name} are initialised")
