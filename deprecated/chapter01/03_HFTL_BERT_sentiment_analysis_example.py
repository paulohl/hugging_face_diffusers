# Import necessary libraries
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
# Initialize tokenizer and model from pretrained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Setup the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
# Example text
example_text = "Hugging Face Transformers is incredibly simple to use. What an amazing library!"
# Perform sentiment analysis
result = nlp(example_text)
# Print the result
print(f"Sentiment: {result[0]['label']}, with a confidence of {result[0]['score']:.4f}")
