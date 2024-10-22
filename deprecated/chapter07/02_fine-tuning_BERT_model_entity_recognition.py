from transformers import BertForTokenClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)  # Adjust num_labels according to your dataset
# Load and preprocess the dataset
dataset = load_dataset("conll2003")
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)
# Start fine-tuning
trainer.train()
