from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
# Sample dataset
data = {'review': ['I loved the movie!', 'That was the worst movie ever...'],
        'sentiment': [1, 0]}  # 1 for positive, 0 for negative
df = pd.DataFrame(data)
# Splitting the dataset
train_df, test_df = train_test_split(df, test_size=0.25)
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, sentiments):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def __len__(self):
        return len(self.reviews)
def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(sentiment)
        }
# Prepare the dataset
train_dataset = MovieReviewDataset(train_df['review'].tolist(), train_df['sentiment'].tolist())
test_dataset = MovieReviewDataset(test_df['review'].tolist(), test_df['sentiment'].tolist())
# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
# Start training
trainer.train()
