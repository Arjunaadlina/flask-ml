from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

# Load your model (assuming it's a scikit-learn model saved as .sav)
model = joblib.load('svm_tfidf_86.sav')

# Load the saved TF-IDF vocabulary
with open('vokabuler.pkl', 'rb') as vocab_file:
    vocab = pickle.load(vocab_file)

# Initialize TfidfVectorizer with the loaded vocabulary
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # JSON input
    news_title = data['title']
    
    # Transform the news title into a TF-IDF feature vector
    tfidf_features = tfidf_vectorizer.fit_transform([news_title])
    
    # Make a prediction using the SVM model
    prediction = model.predict(tfidf_features)
    
    return jsonify({'category': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
