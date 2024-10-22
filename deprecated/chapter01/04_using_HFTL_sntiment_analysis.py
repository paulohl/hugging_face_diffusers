# Import necessary libraries
from transformers import pipeline

# Initialize a pipeline for sentiment analysis with a pretrained model
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

# Example texts
texts = ["I love using Hugging Face Transformers, they make NLP easy!", 
         "The movie was terrible and I was disappointed by the plot."]

# Perform sentiment analysis
results = classifier(texts)

# Print results
for i, text in enumerate(texts):
    print(f"Text: {text}\nSentiment: {results[i]['label']} with a confidence of {results[i]['score']:.4f}\n")
