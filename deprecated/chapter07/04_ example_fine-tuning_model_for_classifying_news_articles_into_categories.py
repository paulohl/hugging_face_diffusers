from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
# Preparing the dataset
from datasets import load_dataset
dataset = load_dataset('ag_news')
dataset = dataset.map(lambda e: {'labels': e['label'], **tokenizer(e['text'], padding='max_length', truncation=True)}, batched=True)
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    warmup_steps=100,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=10
)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)
# Start training
trainer.train()
