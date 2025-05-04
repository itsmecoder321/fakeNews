import pandas as pd
import string
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


fake_df = pd.read_csv(r"fakeNews\Fake.csv")
true_df = pd.read_csv(r"fakeNews\True.csv")

fake_df["label"] = 1  # 1 = Fake
true_df["label"] = 0  # 0 = Real

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[["text", "label"]].dropna()

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df["cleaned_text"] = df["text"].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["cleaned_text"]).toarray()
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)




