import unittest
import torch
from include.models.longformer import LongformerModel
from include.models.bigbird import BigBirdModel
from include.models.legalbert import LegalBERTModel


class TestModels(unittest.TestCase):
    def setUp(self):
        self.models = {
            "longformer": LongformerModel(model_name="allenai/longformer-base-4096", num_labels=2),
            "bigbird": BigBirdModel(model_name="google/bigbird-roberta-base", num_labels=2),
            "legalbert": LegalBERTModel(model_name="nlpaueb/legal-bert-base-uncased", num_labels=2),
        }

    def test_tokenization(self):
        text = "This is a sample text for testing tokenization."
        max_length = 128

        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                inputs = model.tokenize([text], max_length=max_length)
                self.assertIn("input_ids", inputs, f"{
                              model_name} missing input_ids")
                self.assertEqual(
                    inputs["input_ids"].shape[1],
                    max_length,
                    f"{model_name} tokenization length mismatch",
                )

    def test_forward_pass(self):
        text = "This is another test input for the forward pass."
        max_length = 128
        for model_name, model in self.models.items():
            with self.subTest(model=model_name):
                inputs = model.tokenize([text], max_length=max_length)
                inputs = {key: torch.tensor(val)
                          for key, val in inputs.items()}
                outputs = model.forward(inputs)
                self.assertEqual(
                    outputs.logits.shape,
                    (1, 2),
                    f"{model_name} forward pass output shape mismatch",
                )


if __name__ == "__main__":
    unittest.main()
