from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("This is an example.", return_tensors="pt", padding=True)
