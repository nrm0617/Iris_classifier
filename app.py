import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Iris Classifier App", layout="centered")

# Title
st.title("🌸 Iris Flower Classification App")
st.markdown("Enter flower measurements and click **Predict** to see the species.")

# Load CSV file
try:
    df = pd.read_csv("iris.csv")
    st.success("✅ iris.csv loaded successfully!")
except FileNotFoundError:
    st.error("❌ iris.csv not found in the current directory. Please add the file and restart.")
    st.stop()

# Convert feature columns to numeric (coerce errors to NaN)
feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# Encode target variable 'species'
le = LabelEncoder()
df['target'] = le.fit_transform(df['species'])

# Features and target
X = df[feature_cols]
y = df['target']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# Sidebar inputs
st.sidebar.header("📌 Enter Flower Measurements")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=float(X['sepal_length'].min()), max_value=float(X['sepal_length'].max()), value=float(X['sepal_length'].mean()))
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=float(X['sepal_width'].min()), max_value=float(X['sepal_width'].max()), value=float(X['sepal_width'].mean()))
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=float(X['petal_length'].min()), max_value=float(X['petal_length'].max()), value=float(X['petal_length'].mean()))
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=float(X['petal_width'].min()), max_value=float(X['petal_width'].max()), value=float(X['petal_width'].mean()))

# Predict button
if st.sidebar.button("🔮 Predict"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(sample)[0]
    predicted_species = le.inverse_transform([prediction])[0]
    st.subheader("Prediction Result:")
    st.write(f"🌼 The predicted Iris species is: **{predicted_species.capitalize()}**")

# Show confusion matrix
st.subheader("📊 Confusion Matrix (Model Performance on Training Data)")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

