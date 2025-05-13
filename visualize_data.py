import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])

label_counts = df["label"].value_counts()

plt.style.use("seaborn-vibrant")

label_counts.plot(kind="bar", color=["skyblue", "salmon"], edgecolor="black")

plt.title("📊 Rozkład klas: HAM vs SPAM", fontsize=14)
plt.xlabel("Klasa", fontsize=12)
plt.ylabel("Liczba wiadomości", fontsize=12)
plt.xticks(rotation=0)

for i, count in enumerate(label_counts):
    plt.text(i, count + 20, str(count), ha='center', fontsize=10)

plt.tight_layout()
plt.show()
