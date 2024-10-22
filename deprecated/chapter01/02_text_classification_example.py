from transformers import pipeline
# Initialize sentiment analysis pipeline with pre-trained model
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")
texts = ["I love using Hugging Face Transformers, they make NLP easy!", 
         "The movie was terrible and I was disappointed by the plot."]
results = classifier(texts)
for i, text in enumerate(texts):
    print(f"Text: {text}\nSentiment: {results[i]['label']} with a confidence of {results[i]['score']:.4f}\n")
