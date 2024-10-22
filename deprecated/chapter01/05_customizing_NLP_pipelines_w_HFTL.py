# Import necessary libraries
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
# Create the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
# Example text
text = "Hugging Face is a technology company based in New York and Paris."
# Perform named entity recognition
ner_results = ner_pipeline(text)
# Print detected entities and their labels
print("Detected Entities and their Labels:")
for entity in ner_results:
    print(f"Text: {entity['word']}, Entity: {entity['entity']}")
