import pandas as pd
from sklearn.cluster import KMeans
import pickle

# Load dataset
data = pd.read_csv("../data/Mall_Customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)

# Save model
pickle.dump(kmeans, open("../model.pkl", "wb"))

print("Model trained successfully!")