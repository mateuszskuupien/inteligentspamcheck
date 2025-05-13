import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import re

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
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
print("Model wytrenowany")
print("Dokładność:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji:\n", classification_report(y_test, y_pred))

def show_visualization():
    label_counts = df["label"].value_counts()
    labels = ["HAM", "SPAM"]

    plt.style.use("ggplot")
    label_counts.index = labels
    label_counts.plot(kind="bar", color=["skyblue", "salmon"], edgecolor="black")
    plt.title("Rozkład klas: HAM vs SPAM", fontsize=14)
    plt.xlabel("Klasa", fontsize=12)
    plt.ylabel("Liczba wiadomości", fontsize=12)
    plt.xticks(rotation=0)

    for i, count in enumerate(label_counts):
        plt.text(i, count + 20, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

def classify_from_file():
    if not os.path.exists("examples.txt"):
        print("Brak pliku examples.txt")
        return

    with open("examples.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            print("Pomiń niepoprawny format:", line)
            continue

        label, message = parts
        vect_msg = vectorizer.transform([message])
        pred = model.predict(vect_msg)[0]
        result = "SPAM" if pred else "HAM"
        print(f"[{result}] {message}")
    print()

def manual_input():
    while True:
        msg = input("Wpisz przykładową wiadomość SMS (albo 'exit' żeby zakończyć): ")
        if msg.lower() == "exit":
            break
        vect_msg = vectorizer.transform([msg])
        pred = model.predict(vect_msg)[0]
        print("To JEST SPAM!" if pred else "To NIE jest SPAM!")

def main():
    while True:
        print("\n--- MENU ---")
        print("1. Klasyfikuj wiadomości z examples.txt")
        print("2. Wpisz wiadomości ręcznie")
        print("3. Pokaż wizualizację danych")
        print("0. Wyjście")

        choice = input("Wybierz opcję: ")
        if choice == "1":
            classify_from_file()
        elif choice == "2":
            manual_input()
        elif choice == "3":
            show_visualization()
        elif choice == "0":
            break
        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")

if __name__ == "__main__":
    main()
