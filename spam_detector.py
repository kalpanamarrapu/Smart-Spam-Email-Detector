import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Features & Labels
X = df['message']
y = df['label']

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
# Save model
pickle.dump(model, open("spam_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# Test model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Take user input
user_msg = input("Enter a message: ")

msg = [user_msg]

msg_vector = vectorizer.transform(msg)

prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")
msg_vector = vectorizer.transform(msg)

prediction = model.predict(msg_vector)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")