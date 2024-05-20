import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# data setup
def load_data_from_csv(filename):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, filename)
    df = pd.read_csv(file_path)
    return list(zip(df['Input'], df['Label']))

# Initialize lemmatizer and stopwords set
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Simple text preprocessing
def preprocess(text):
    text = text.lower() # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [token for token in tokens if token.isalpha()] # Remove non-alpha tokens
    tokens = [token for token in tokens if token not in stop_words] # Remove stopwords
    lemmas = [lemmatizer.lemmatize(token) for token in tokens] # Lemmatize the words
    return lemmas

filename = 'Synthetic_Request_Data.csv'
data = load_data_from_csv(filename)
data = np.array(data)
np.random.shuffle(data)

# Build the vocabulary
vocabulary = set()
for text, _ in data:
    words = preprocess(text)
    vocabulary.update(words)
vocabulary = list(vocabulary)

# Feature extraction: Bag of Words model
def feature_extraction(text):
    words = preprocess(text)
    features = np.zeros(len(vocabulary))
    for word in words:
        if word in vocabulary:
            features[vocabulary.index(word)] += 1
    return features

# Naive Bayes
class NaiveBayesClassifier:
    def __init__(self):
        self.log_prior = {}
        self.log_likelihood = {}
        self.classes = []

    def train(self, data):
        class_count = {}
        word_count = {}
        
        # Initialize counts
        for _, label in data:
            if label not in class_count:
                class_count[label] = 0
                word_count[label] = np.zeros(len(vocabulary))
            class_count[label] += 1

        total_docs = len(data)
        self.classes = list(class_count.keys())

        # Calculate log prior
        for c in self.classes:
            self.log_prior[c] = np.log(class_count[c] / total_docs)
        
        # Count words per class
        for text, label in data:
            features = feature_extraction(text)
            word_count[label] += features
        
        # Calculate log likelihood
        for c in self.classes:
            total_words = np.sum(word_count[c])
            self.log_likelihood[c] = np.log((word_count[c] + 1) / (total_words + len(vocabulary)))

    def predict(self, text):
        features = feature_extraction(text)
        probs = {}
        for c in self.classes:
            probs[c] = self.log_prior[c]
            for i, word in enumerate(features):
                if word > 0:
                    probs[c] += self.log_likelihood[c][i] * word
        return max(probs, key=probs.get)
    
def evaluate_model(classifier, validation_data):
    correct_predictions = 0
    total_predictions = len(validation_data)
    
    for text, true_label in validation_data:
        prediction = classifier.predict(text)
        if prediction == true_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# Split the data into training and validation sets (80% train, 20% validation)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
validation_data = data[train_size:]

# Train the model
classifier = NaiveBayesClassifier()
classifier.train(train_data)

accuracy = evaluate_model(classifier, validation_data)
'''print(f"Validation Accuracy: {accuracy:.2f}")'''

# Example prediction
'''example_text = "Dallas recommendations please"
prediction = classifier.predict(example_text)
print(f"The text is classified as: {prediction}")
example_text = "Tell me about landmarks in San Francisco ."
prediction = classifier.predict(example_text)
print(f"The text is classified as: {prediction}")
example_text = "Which restaurant have the best ratings nearby?"
prediction = classifier.predict(example_text)
print(f"The text is classified as: {prediction}")'''