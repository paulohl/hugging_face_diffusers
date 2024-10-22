from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier("This movie was amazing!")
