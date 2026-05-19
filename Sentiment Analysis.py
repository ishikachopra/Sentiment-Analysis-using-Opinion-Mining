import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import pipeline

def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


text = read_text_file("read.txt")
if text is None:
    exit()

def preprocess_text(text):
    lowercase = text.lower()
    clean_text = lowercase.translate(str.maketrans("", "", string.punctuation))
    return clean_text


def label_sentiments(text, emotion_dict):
    final_words = []
    sentiments = []
    for word in word_tokenize(text, "english"):
        if word not in stopwords.words("english"):
            final_words.append(word)
            sentiment = emotion_dict.get(word, "Neutral")
            sentiments.append(sentiment)
    return final_words, sentiments


emotion_dict = {}
with open("emotions.txt", "r") as file:
    for line in file:
        clear_line = line.strip().replace(",", "").replace("'", "")
        word, emotion = clear_line.split(":")
        emotion_dict[word] = emotion

clean_text = preprocess_text(text)
final_words, sentiments = label_sentiments(clean_text, emotion_dict)


tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(final_words)


X_train, X_test, y_train, y_test = train_test_split(
    X, sentiments, test_size=0.2, random_state=42
)
svm_model = LinearSVC(dual=True) 
svm_model.fit(X_train, y_train)


print("=" * 60)
print("1. SVM CLASSIFICATION REPORT")
print("=" * 60)
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))


print("\n" + "=" * 60)
print("2. BERT CLASSIFICATION REPORT")
print("=" * 60)
print("Loading Pre-trained Emotion BERT pipeline...")

bert_classifier = pipeline("sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-emotion")

raw_words_train, raw_words_test, labels_train, labels_test = train_test_split(
    final_words, sentiments, test_size=0.2, random_state=42
)

print("Running inference using BERT on test words...")
bert_raw_outputs = bert_classifier(raw_words_test)

bert_preds = [res['label'].title() if res['label'] != 'joy' else 'Happy' for res in bert_raw_outputs]

print(classification_report(labels_test, bert_preds, zero_division=1))

def analyze_sentiment_vader(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score["neg"]
    pos = score["pos"]
    if neg > pos:
        return "Negative Sentiment"
    elif pos > neg:
        return "Positive Sentiment"
    else:
        return "Neutral Vibe"


def analyze_sentiment_bert(sentiment_text):
  
    binary_bert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    truncated_text = " ".join(sentiment_text.split()[:400])
    result = binary_bert(truncated_text)[0]
    
    return f"{result['label'].title()} Sentiment (Confidence: {result['score']:.4f})"


print("\n" + "=" * 60)
print("3. OVERALL DOCUMENT TEXT BENCHMARK")
print("=" * 60)

vader_result = analyze_sentiment_vader(clean_text)
print("Overall Sentiment (VADER):", vader_result)

bert_result = analyze_sentiment_bert(clean_text)
print("Overall Sentiment (BERT): ", bert_result)

word_count = Counter(sentiments)
fig, ax1 = plt.subplots()
ax1.bar(word_count.keys(), word_count.values())
fig.autofmt_xdate()
plt.show()
