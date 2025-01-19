import numpy as np
from flask import Flask, request, render_template
import pickle
from flask import Flask, request, render_template, jsonify
# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('svr_best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    G2 = float(request.form['G2'])
    failures = int(request.form['failures'])
    absences = int(request.form['absences'])
    Fedu = int(request.form['Fedu'])
    studytime = int(request.form['studytime'])
    schoolsup = int(request.form['schoolsup'])
    higher = int(request.form['higher'])
    internet = int(request.form['internet'])

    # Dummy prediction logic (replace with your ML model logic)
    predicted_marks = (G2 + (5 - failures) + (studytime * 2) - (absences / 10) + (Fedu * 2)) * 0.9

    # Render the output back to the user
    return render_template('index.html', prediction_text=f"Predicted Final Marks: {predicted_marks:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
