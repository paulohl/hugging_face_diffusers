from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Load and preprocess the dataset
dataset = load_dataset('glue', 'sst2', split='train')
dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding="max_length", truncation=True), batched=True)
dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)
# Start fine-tuning
trainer.train()
