import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from transformers import pipeline



nltk.download("stopwords")
nltk.download("vader_lexicon")



df = pd.read_csv("goemotions.csv")


print(df.head())

print("\nDataset Columns:")
print(df.columns)




texts = df["text"]
labels = df["label"]



stop_words = set(stopwords.words("english"))


def preprocess_text(text):

 
    text = text.lower()

  
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

 
    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


cleaned_texts = texts.apply(preprocess_text)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    cleaned_texts,
    labels,
    test_size=0.2,
    random_state=42
)

print("\n==============================")
print("SVM MODEL")
print("==============================")

tfidf = TfidfVectorizer(max_features=5000)

X_train = tfidf.fit_transform(X_train_text)

X_test = tfidf.transform(X_test_text)

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nSVM Accuracy:", svm_accuracy)

print("\nSVM Classification Report:\n")

print(
    classification_report(
        y_test,
        svm_predictions,
        zero_division=1
    )
)


print("\n==============================")
print("VADER MODEL")
print("==============================")


vader = SentimentIntensityAnalyzer()


def vader_sentiment(text):

    score = vader.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"


print("\nSample VADER Predictions:\n")

for text in X_test_text.head(5):

    prediction = vader_sentiment(text)

    print("Text:", text)

    print("Prediction:", prediction)

    print()

print("\n==============================")
print("BERT MODEL")
print("==============================")


bert_model = pipeline(
    "sentiment-analysis"
)


print("\nSample BERT Predictions:\n")

for text in X_test_text.head(5):

    result = bert_model(text)

    print("Text:", text)

    print("Prediction:", result)

    print()


print("\n==============================")
print("CUSTOM TEXT ANALYSIS")
print("==============================")

user_text = input("\nEnter text:\n")


clean_input = preprocess_text(user_text)

vector_input = tfidf.transform([clean_input])

svm_result = svm_model.predict(vector_input)

print("\nSVM Prediction:", svm_result[0])


vader_result = vader_sentiment(user_text)

print("VADER Prediction:", vader_result)


bert_result = bert_model(user_text)

print("BERT Prediction:", bert_result)



emotion_counts = Counter(labels)

plt.figure(figsize=(10, 5))

plt.bar(
    emotion_counts.keys(),
    emotion_counts.values()
)

plt.xticks(rotation=90)

plt.title("Emotion Distribution in GoEmotions Dataset")

plt.xlabel("Emotion")

plt.ylabel("Count")

plt.tight_layout()

plt.show()
