from transformers import Trainer, TrainingArguments, DataCollatorWithPadding


def fine_tune(model, training_data, testing_data):
    training_args = TrainingArguments(
        output_dir=f"./results/{model.model_name}/",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        logging_dir=f"./results/{model.model_name}/logs",
        save_steps=500,
        gradient_checkpointing=True,
        logging_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=True,
        disable_tqdm=False,
        report_to="none",
        eval_strategy="epoch"
    )

    data_collator = DataCollatorWithPadding(model.tokenizer, padding='longest')

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=testing_data,
        data_collator=data_collator,
    )

    trainer.train()

    model.model.save_pretrained(
        f"results/{model.model_name}/fine_tuned_{model.model_name}")
    model.tokenizer.save_pretrained(
        f"results/{model.model_name}/fine_tuned_{model.model_name}")
