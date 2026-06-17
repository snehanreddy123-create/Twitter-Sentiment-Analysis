from flask import Flask, request, jsonify, render_template
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

script_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(script_dir, 'templates'))

loaded_model = pickle.load(open(os.path.join(script_dir, 'trained_model.sav'), 'rb'))
loaded_vectorizer = pickle.load(open(os.path.join(script_dir, 'vectorizer.sav'), 'rb'))

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    content = ' '.join(content)
    return content

THRESHOLD = 0.70

def predict_neutral(text):
    s = stemming(text)
    x_vec = loaded_vectorizer.transform([s])
    probs = loaded_model.predict_proba(x_vec)[0]

    neg_p, pos_p = probs[0], probs[1]
    max_p = max(neg_p, pos_p)

    if max_p < THRESHOLD:
        return "Neutral", round(max_p * 100, 2)
    elif neg_p > pos_p:
        return "Negative", round(neg_p * 100, 2)
    else:
        return "Positive", round(pos_p * 100, 2)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']
    sentiment, confidence = predict_neutral(tweet)

    return jsonify({
        "result": sentiment,
        "confidence": confidence
    })


if __name__ == '__main__':
    app.run(debug=True)
