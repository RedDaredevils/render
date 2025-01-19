import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('svr_best_model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    try:
        G2 = float(request.form.get('G2'))
        failures = float(request.form.get('failures'))
        absences = float(request.form.get('absences'))
        Fedu = float(request.form.get('Fedu'))
        studytime = float(request.form.get('studytime'))
        schoolsup = int(request.form.get('schoolsup'))
        higher = int(request.form.get('higher'))
        internet = int(request.form.get('internet'))
    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numbers.")

    # Prepare the input features
    features = np.array([[G2, failures, absences, Fedu, studytime, schoolsup, higher, internet]])

    # Make the prediction
    prediction = model.predict(features)[0]

    # Render the result on the web page
    return render_template('index.html', prediction_text=f"Predicted Marks: {prediction:.2f}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
