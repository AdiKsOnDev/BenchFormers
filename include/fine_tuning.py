from transformers import Trainer, TrainingArguments


def fine_tune(model, training_data, testing_data):
    training_args = TrainingArguments(
        output_dir=f"./results/{model.model_name}",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./results/{model.model_name}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    trainer.train()

    model.model.save_pretrained(
        f"results/{model.model_name}/fine_tuned_{model.model_name}")
    model.tokenizer.save_pretrained(
        f"results/{model.model_name}/fine_tuned_{model.model_name}")
