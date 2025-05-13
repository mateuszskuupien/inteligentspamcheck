import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Dokładność:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred))

while True:
    msg = input("Wpisz przykładową wiadomość SMS (albo 'exit' żeby zakończyć): ")
    if msg.lower() == "exit":
        break
    vect_msg = vectorizer.transform([msg])
    pred = model.predict(vect_msg)[0]
    print("To JEST spam!" if pred else "To NIE jest spam!")
