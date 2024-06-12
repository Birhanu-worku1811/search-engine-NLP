"""Uncomment the following lines when running the first time then comment it"""
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

from flask import Flask, render_template, request
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Load the BBC News Archive Dataset
df = pd.read_csv('bbc-news-data.csv', delimiter='\t', header=None, names=['category', 'filename', 'title', 'content'],
                 skiprows=1)

# Preprocess the text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


df['clean_text'] = df['content'].apply(preprocess_text)

# Check if precomputed doc vectors exist, if not, compute and save them
try:
    doc_vectors = np.load('doc_vectors.npy')
except FileNotFoundError:
    doc_vectors = np.array([nlp(text).vector for text in df['clean_text']])
    np.save('doc_vectors.npy', doc_vectors)


# semantic search function
def semantic_search(query, doc_vectors, documents, top_n=3):
    query = preprocess_text(query)
    query_vector = nlp(query).vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    # Sort documents by similarity scores
    similar_doc_indices = np.argsort(similarities)[::-1][:top_n]
    results = []
    for idx in similar_doc_indices:
        title = documents.iloc[idx]['title']
        content = documents.iloc[idx]['content']
        results.append((title, content))
    return results


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    results = semantic_search(query, doc_vectors, df)
    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
