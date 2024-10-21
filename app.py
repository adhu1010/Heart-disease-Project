from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load all models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Support Vector Machine": joblib.load("models/svm.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "K-Nearest Neighbors": joblib.load("models/knn.pkl")
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Model selection
        selected_model = request.form['model']

        # Prepare input data for prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Get the selected model
        model = models[selected_model]

        # Prediction
        prediction = model.predict(data)[0]

        # Result
        if prediction == 1:
            result = 'High risk of heart disease'
        else:
            result = 'Low risk of heart disease'

        return render_template('result.html', prediction=result, model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
