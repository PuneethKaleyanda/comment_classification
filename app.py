from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.stem.snowball import SnowballStemmer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('comment_classification\model\logistic_regression_model_balance.pkl')

# Initialize stemmer
stemmer = SnowballStemmer('english')

# Pre-compiled regex patterns for text cleaning
patterns = {
    r"what's": "what is",
    r"\'s": " ",
    r"\'ve": " have",
    r"can't": "can not",
    r"n't": " not",
    r"i'm": "i am",
    r"\'re": " are",
    r"\'d": " would",
    r"\'ll": " will",
    r'\W': ' ',  # Replace non-word characters
    r'\s+': ' '  # Replace multiple spaces with single space
}

# Preprocess function (clean + stem)
def preprocess_text(text):
    text = text.lower()
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    text = text.strip()
    return " ".join(stemmer.stem(word) for word in text.split())

# Route to render the HTML frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and return model predictions
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['comment']
    
    # Preprocess the input
    processed_text = preprocess_text(user_input)
    
    # Make prediction using the loaded model
    prediction = model.predict([processed_text])[0]
    
    # Convert prediction to a readable format
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result = {label: bool(prediction[idx]) for idx, label in enumerate(labels)}
    
    # Return result to the frontend
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
