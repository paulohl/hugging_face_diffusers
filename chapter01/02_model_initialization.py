from transformers import BertForSequenceClassification, Trainer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
