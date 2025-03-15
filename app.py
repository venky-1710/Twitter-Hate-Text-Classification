# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from collections import Counter

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Text preprocessing functions
def simplify(text):
    '''Function to handle the diacritics in the text'''
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)

def remove_stopwords(text):
    '''Function to remove the stop words from the text corpus'''
    clean_text = [word for word in text if not word in stop_words]
    return clean_text

def remove_hashsymbols(text):
    '''Function to remove the hashtag symbol from the text'''
    pattern = re.compile(r'#')
    text = ' '.join(text)
    clean_text = re.sub(pattern, '', text)
    return tokenizer.tokenize(clean_text)

def rem_digits(text):
    '''Function to remove the digits from the list of strings'''
    no_digits = []
    for word in text:
        no_digits.append(re.sub(r'\d', '', word))
    return ' '.join(no_digits)

def rem_nonalpha(text):
    '''Function to remove the non-alphanumeric characters from the text'''
    text = [word for word in text if word.isalpha()]
    return text

# Initialize tokenizer and stopwords
tokenizer = TweetTokenizer(preserve_case=True)
stop_words = stopwords.words('english')
# Add additional stop words
additional_list = ['amp', 'rt', 'u', "can't", 'ur']
for words in additional_list:
    stop_words.append(words)

# Define common aggressive words/hate speech terms
AGGRESSIVE_WORDS = {
    'hate': 8,
    'kill': 10,
    'die': 7, 
    'stupid': 5,
    'idiot': 6,
    'dumb': 5,
    'fool': 4,
    'garbage': 5,
    'trash': 5,
    'worthless': 7,
    'ugly': 4,
    'disgust': 6,
    'violent': 7,
    'attack': 6,
    'terrorist': 9,
    'disgusting': 7,
    'horrible': 6,
    'evil': 7,
    'terrible': 5,
    'worst': 5,
    'destroy': 7,
    'ugly': 5,
    'sick': 4,
    'pathetic': 6,
    'loser': 5,
    'filthy': 6,
    'racist': 9,
    'bigot': 8,
    'sexist': 8,
    'moron': 6,
    'loathe': 7,
    'despise': 7,
    'damn': 4,
    'hell': 4,
    'retard': 9,
    'scum': 7,
    'crap': 5,
}

def identify_aggressive_words(text):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    aggressive_found = [word for word in words if word in AGGRESSIVE_WORDS]
    severity = 0
    if aggressive_found:
        total_weight = sum(AGGRESSIVE_WORDS[word] for word in aggressive_found)
        max_possible = 10 * len(words)
        severity = min(100, int((total_weight / max_possible) * 100))
    
    highlighted_text = text
    for word in set(aggressive_found):
        pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span class="highlight">{word}</span>', highlighted_text)
    
    return {
        'aggressive_words': aggressive_found,
        'severity': severity,
        'highlighted_text': highlighted_text,
        'count': len(aggressive_found)
    }

def predict_hate_speech(text, model, vectorizer):
    if isinstance(text, str):
        text = [text]
    
    processed_text = []
    original_text = text[0]
    
    for t in text:
        t = simplify(t)
        t = re.sub(r'@\w+', '', t)
        t = re.sub(r'http\S+', '', t)
        tokens = tokenizer.tokenize(t)
        tokens = [word for word in tokens if not word in stop_words]
        tokens = remove_hashsymbols(tokens)
        tokens = rem_digits(tokens)
        tokens = tokenizer.tokenize(tokens)
        tokens = rem_nonalpha(tokens)
        processed_text.append(' '.join(tokens))
    
    X = vectorizer.transform(processed_text)
    predictions = model.predict(X)
    proba = model.predict_proba(X)
    aggressive_analysis = identify_aggressive_words(original_text)
    
    return predictions, proba, aggressive_analysis

def load_or_train_model():
    model_path = './hate_speech_model.joblib'
    vectorizer_path = './vectorizer.joblib'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("Loading existing model and vectorizer...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Training new model...")
        try:
            tweet = pd.read_csv('./TwitterHate.csv', delimiter=',', engine='python', encoding='utf-8-sig')
        except:
            return None, None, "Dataset not found. Please place TwitterHate.csv in the application directory."
        
        if 'id' in tweet.columns:
            tweet.drop('id', axis=1, inplace=True)
        
        df = tweet.copy()
        df['tweet'] = df['tweet'].apply(simplify)
        df['tweet'].replace(r'@\w+', '', regex=True, inplace=True)
        df['tweet'].replace(r'http\S+', '', regex=True, inplace=True)
        df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
        df['tweet'] = df['tweet'].apply(remove_stopwords)
        df['tweet'] = df['tweet'].apply(remove_hashsymbols)
        df['tweet'] = df['tweet'].apply(rem_digits)
        df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
        df['tweet'] = df['tweet'].apply(rem_nonalpha)
        df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x))
        
        X = df['tweet']
        y = df['label']
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(X)
        
        weights = {0: 1.0, 1: 13.0}
        model = LogisticRegression(C=0.16731783677034165, penalty='l2', solver='liblinear', class_weight=weights)
        model.fit(X, y)
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
    
    return model, vectorizer, None

@app.route('/')
def home():
    return render_template('index.html', error=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/predict', methods=['POST'])
def predict():
    model, vectorizer, error_message = load_or_train_model()
    if model is None or vectorizer is None:
        return jsonify({'error': error_message})
    
    text = request.form['tweet']
    predictions, probabilities, aggressive_analysis = predict_hate_speech([text], model, vectorizer)
    
    result = {
        'prediction': int(predictions[0]),
        'probability': float(probabilities[0][1]),
        'is_hate_speech': bool(predictions[0] == 1),
        'processed_text': text,
        'aggressive_words': aggressive_analysis['aggressive_words'],
        'aggressive_count': aggressive_analysis['count'],
        'severity': aggressive_analysis['severity'],
        'highlighted_text': aggressive_analysis['highlighted_text']
    }
    
    return jsonify(result)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/report', methods=['POST'])
def report_hate_speech():
    if request.method == 'POST':
        report_data = request.json
        return jsonify({'success': True, 'message': 'Report submitted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
