import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Split features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model directory
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/crop_recommender.pkl")

print("Model trained and saved to models/crop_recommender.pkl")
