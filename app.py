import nltk
from nltk.data import find

from flask import Flask, render_template, request
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def download_nltk_data():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()


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


df['clean_title'] = df['title'].apply(preprocess_text)
df['clean_content'] = df['content'].apply(preprocess_text)

# Check if precomputed doc vectors exist, if not, compute and save them
try:
    doc_vectors_title = np.load('doc_vectors_title.npy')
    doc_vectors_content = np.load('doc_vectors_content.npy')
except FileNotFoundError:
    doc_vectors_title = np.array([nlp(text).vector for text in df['clean_title']])
    doc_vectors_content = np.array([nlp(text).vector for text in df['clean_content']])
    np.save('doc_vectors_title.npy', doc_vectors_title)
    np.save('doc_vectors_content.npy', doc_vectors_content)


# semantic search function
def semantic_search(query, doc_vectors_title, doc_vectors_content, documents, top_n=5):
    query = preprocess_text(query)
    query_vector = nlp(query).vector.reshape(1, -1)
    similarities_title = cosine_similarity(query_vector, doc_vectors_title).flatten()
    similarities_content = cosine_similarity(query_vector, doc_vectors_content).flatten()
    # Combine similarities from title and content
    similarities_combined = 0.5 * similarities_title + 0.5 * similarities_content
    # Sort documents by combined similarity scores
    similar_doc_indices = np.argsort(similarities_combined)[::-1][:top_n]
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
    num_results = int(request.form.get('num_results', 5))
    results = semantic_search(query, doc_vectors_title, doc_vectors_content, df, top_n=num_results)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
