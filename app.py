from flask import Flask, render_template, request, redirect, Response
import pickle
import csv
import io
import sqlite3

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Create database
def init_db():

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT,
        result TEXT,
        words INTEGER
    )
    """)

    conn.commit()
    conn.close()


init_db()


# HOME PAGE
@app.route("/")
def home():

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT message, result, words FROM history"
    )

    history = cursor.fetchall()

    conn.close()

    spam_count = sum(
        1 for item in history
        if item[1].startswith("Spam")
    )

    not_spam_count = sum(
        1 for item in history
        if item[1].startswith("Not Spam")
    )

    return render_template(
        "index.html",
        history=history,
        total=len(history),
        spam_count=spam_count,
        not_spam_count=not_spam_count
    )


# ✅ UPDATED PREDICT FUNCTION
@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    msg_vector = vectorizer.transform([message])

    prediction = model.predict(msg_vector)

    probability = model.predict_proba(msg_vector)

    confidence = probability.max() * 100

    word_count = len(message.split())

    # ✅ clean label (no % inside)
    if prediction[0] == 1:
        result = "Spam 🚫"
    else:
        result = "Not Spam ✅"

    # ✅ attach confidence separately
    result_with_conf = f"{result} ({confidence:.2f}%)"

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (message, result, words) VALUES (?, ?, ?)",
        (message, result_with_conf, word_count)
    )

    conn.commit()
    conn.close()

    return redirect("/")


# DOWNLOAD CSV
@app.route("/download")
def download_csv():

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT message, result, words FROM history"
    )

    history = cursor.fetchall()

    conn.close()

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([
        "Message",
        "Result",
        "Word Count"
    ])

    for item in history:
        writer.writerow(item)

    output.seek(0)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=spam_history.csv"
        }
    )


# GRAPH
@app.route("/chart")
def chart():

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT result FROM history"
    )

    results = cursor.fetchall()

    conn.close()

    spam_count = sum(
        1 for r in results
        if r[0].startswith("Spam")
    )

    not_spam_count = sum(
        1 for r in results
        if r[0].startswith("Not Spam")
    )

    labels = ["Spam", "Not Spam"]

    values = [
        spam_count,
        not_spam_count
    ]

    plt.figure()

    plt.bar(labels, values)

    img = io.BytesIO()

    plt.savefig(img, format="png")

    plt.close()

    img.seek(0)

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )


# CLEAR HISTORY
@app.route("/clear")
def clear_history():

    conn = sqlite3.connect("spam.db")

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM history"
    )

    conn.commit()
    conn.close()

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)