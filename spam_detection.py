import pandas as pd

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print(df.head())
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['label'].value_counts())

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nAfter Conversion:")
print(df.head())

X = df['message']

y = df['label']

print("\nFeatures:")
print(X.head())

print("\nLabels:")
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining Matrix Shape:")
print(X_train_tfidf.shape)

print("\nTesting Matrix Shape:")
print(X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

print("\nModel trained successfully!")

y_pred = model.predict(X_test_tfidf)

print("\nPredictions made successfully!")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

message = ["Congratulations! You won $5000. Claim now!"]

message_tfidf = vectorizer.transform(message)

prediction = model.predict(message_tfidf)

probability = model.predict_proba(message_tfidf)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM")

print(f"Spam Probability: {probability[0][1]:.2%}")

import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved!")