import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Sample text data (could be expanded to a large corpus)
text = "Hello world! Welcome to the world of deep learning for natural language processing."

# Character-level tokenization
chars = sorted(set(text))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# Prepare the dataset
max_length = 10
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - max_length, step):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])
x = np.zeros((len(sentences), max_length, len(chars)), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(chars), output_dim=50, input_length=max_length),
    LSTM(128),
    Dense(len(chars), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x, y, batch_size=128, epochs=60)

# Generate text
def generate_text(length):
    start_index = np.random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_int[char]] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = int_to_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Example of generated text
print(generate_text(50))
