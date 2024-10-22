from transformers import pipeline
# Initialize a pipeline for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")
# Example texts
texts = ["I love my phone, it's the best!", "This is the worst laptop I have ever bought."]
# Process texts and print the results
results = sentiment_pipeline(texts)
for result in results:
    print(f"Text: {result['label']} with a confidence of {result['score']:.2f}")

# explaning
#	• Library Import: The pipeline function from the Hugging Face's transformers library 
#   simplifies the creation of NLP pipelines for various tasks.
#	• Initializing the Pipeline: The sentiment analysis pipeline is initialized, 
#   which internally handles tokenization, model inference, and output generation.
#	• Processing Texts: The example texts are fed into the pipeline, 
#   which uses a pre-trained model to predict sentiment labels and confidence scores.
