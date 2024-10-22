import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

# Example dataset (simplified)
sentences = [
    [("I", "PRON"), ("saw", "VERB"), ("the", "DET"), ("cat", "NOUN")],
    [("The", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"), ("the", "DET"), ("mat", "NOUN")]
]

# Feature extraction for CRF
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True

    return features

# Generate features
X = [[word2features(s, i) for i in range(len(s))] for s in sentences]
y = [[label for token, label in s] for s in sentences]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predictions
y_pred = crf.predict(X_test)
print("Accuracy:", metrics.flat_accuracy_score(y_test, y_pred))
