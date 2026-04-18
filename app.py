from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

app = Flask(__name__)

# Load model (UPDATED PATH FOR DEPLOYMENT)
model = pickle.load(open("model.pkl", "rb"))

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        score = float(request.form['score'])

        # Input validation
        if not (10 <= income <= 150 and 1 <= score <= 100):
            return render_template("index.html",
                                   prediction="Invalid Input ❌",
                                   info="Income must be 10–150 and score 1–100.")

        data = np.array([[income, score]])
        result = model.predict(data)[0]

        # Labels
        cluster_names = {
            0: "Low Income - Low Spending 🪙",
            1: "High Income - High Spending 💰",
            2: "Average Customer 🙂",
            3: "High Spending 💸",
            4: "Careful Customer 🤔"
        }

        # Explanation
        info_map = {
            0: "Customer has low income and low spending tendency.",
            1: "Customer has high income and spends a lot (premium segment).",
            2: "Customer has average behavior.",
            3: "Customer spends a lot regardless of income.",
            4: "Customer is careful with spending."
        }

        label = cluster_names.get(result, "Unknown")
        info = info_map.get(result, "")

        # Load dataset (UPDATED PATH)
        df = pd.read_csv("data/Mall_Customers.csv")

        # Predict clusters
        clusters = model.predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])

        # Plot graph
        plt.figure(figsize=(7, 6))

        plt.scatter(df['Annual Income (k$)'],
                    df['Spending Score (1-100)'],
                    c=clusters,
                    cmap='tab10',
                    s=40,
                    alpha=0.8,
                    label="Customers")

        # Centroids
        centers = model.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1],
                    color='black',
                    marker='X',
                    s=200,
                    label='Centroids')

        # User input
        plt.scatter(income, score,
                    color='red',
                    s=150,
                    label="Your Input")

        plt.xlim(0, 150)
        plt.ylim(0, 100)

        plt.xlabel("Annual Income (k$)")
        plt.ylabel("Spending Score (1-100)")
        plt.title("Customer Segmentation (K-Means)")
        plt.legend()
        plt.grid()

        # Save unique image (fix cache issue)
        filename = f"plot_{int(time.time())}.png"
        filepath = os.path.join("static", filename)
        plt.savefig(filepath)
        plt.close()

        return render_template("index.html",
                               prediction=f"{label} (Cluster {result})",
                               info=info,
                               image_file=filename)

    except Exception as e:
        return render_template("index.html",
                               prediction="Error ❌",
                               info="Something went wrong. Please try again.")

# REQUIRED FOR RENDER DEPLOYMENT
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)