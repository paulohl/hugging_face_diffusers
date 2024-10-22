from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# Create a custom pipeline
def custom_sentiment_analysis(texts):
    # Preprocess the texts
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True)
    # Model inference
    outputs = model(**encoded_input)
    # Post-processing the output to convert logits to sentiment labels
    labels = ['negative', 'neutral', 'positive']
    results = [{'label': labels[x.argmax()], 'score': x.max().item()} for x in outputs.logits]
    return results
# Example usage
texts = ["I love this product!", "This is the worst experience ever."]
print(custom_sentiment_analysis(texts))

# Detailed Explanation
#	• Library Import: The script begins by importing necessary classes from the 
#   Hugging Face transformers library.
#	• Custom Pipeline Function: Defines a function custom_sentiment_analysis that 
#   encapsulates preprocessing, model inference, and custom post-processing.
#	• Model Inference: Executes sentiment analysis using pre-trained BERT-based 
#   models specifically fine-tuned for multilingual sentiment analysis.
