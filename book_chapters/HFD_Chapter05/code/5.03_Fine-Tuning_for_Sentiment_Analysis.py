# The following script demonstrates the fine-tuning process using the widely recognized CoNLL-2003 dataset.  
# This dataset includes annotated text for entities, making it ideal for training and evaluating NER models. 
# The Hugging Face transformers library provides robust tools to streamline the fine-tuning process 

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load and preprocess dataset
dataset = load_dataset('glue', 'sst2')
dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length'), batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./model_save',
    num_train_epochs=3,
    per_device_train_batch_size=16
)
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)
# Fine-tune the model
trainer.train()
