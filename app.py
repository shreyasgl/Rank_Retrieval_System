from flask import Flask, request, render_template
import pandas as pd
import pickle
from collections import Counter

app = Flask(__name__)

# Load the documents
documents = pd.read_csv('documents.csv')  # Ensure this file exists in the same directory

# Load the BM25 model
class BM25Model:
    def _init_(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.avgdl = 0

    def load_model(self, filepath):
        """
        Load BM25 model parameters from a file.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.idf = data['idf']
            self.avgdl = data['avgdl']
            self.k1 = data['k1']
            self.b = data['b']

    def compute_score(self, query):
        """
        Compute BM25 scores for a single query against all documents.
        """
        tokenized_query = Counter(query.split())
        scores = []

        for idx, doc in enumerate(documents['body'].fillna("").tolist()):
            doc = Counter(doc.split())
            score = 0
            for term, freq in tokenized_query.items():
                if term in self.idf:
                    f_td = doc.get(term, 0)
                    numerator = self.idf[term] * f_td * (self.k1 + 1)
                    denominator = f_td + self.k1 * (1 - self.b + self.b * (sum(doc.values()) / self.avgdl))
                    score += numerator / denominator
            scores.append({'doc_id': documents['docid'][idx], 'title': documents['title'][idx], 'score': score})

        # Return ranked scores
        return sorted(scores, key=lambda x: x['score'], reverse=True)

bm25 = BM25Model()
bm25.load_model('C:\\Users\\anilj\\Downloads\\Ranked_Retrieval_System\\bm25_model (1).h5')  # Ensure this file exists in the same directory

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query = request.form['query']
        if query:
            results = bm25.compute_score(query)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)