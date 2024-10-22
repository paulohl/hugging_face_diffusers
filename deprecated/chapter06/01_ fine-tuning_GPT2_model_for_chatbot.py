from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Prepare dataset
train_path = 'path_to_train_data.txt'
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)
