# Iris_classifier

🌸 Iris Flower Classification App

This is a simple machine learning web app built with Streamlit to classify Iris flower species based on user-provided measurements. The app uses a Random Forest Classifier trained on the popular Iris dataset.
🔍 Description

The Iris Classification App allows users to enter flower measurements (sepal and petal length/width) via a sidebar and predicts the corresponding Iris species:

    Iris-setosa

    Iris-versicolor

    Iris-virginica

It also displays the model’s confusion matrix to show performance on the training data.
🧠 Machine Learning

    Algorithm: Random Forest Classifier

    Target Encoding: LabelEncoder

    Evaluation: Accuracy score, confusion matrix

📁 Files Required

    iris.csv – Should be placed in the same directory. Must contain the following columns:

    sepal_length, sepal_width, petal_length, petal_width, species

🛠️ How to Run the App

    Clone the repository:

git clone https://github.com/your-username/iris-classifier-app.git
cd iris-classifier-app

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run iris_classifier_app.py

    Use the sidebar to input flower measurements and click Predict.

📦 Required Python Libraries

    streamlit

    pandas

    numpy

    scikit-learn

    seaborn

    matplotlib
