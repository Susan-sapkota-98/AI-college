from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model = joblib.load('model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')
df = pd.read_csv('df_for_recommend.csv')

def recommend_similar(article_text, top_n=3):
    predicted_cat = model.predict([article_text])[0]
    same_cat_idx = df[df['Category'] == predicted_cat].index
    article_vec = tfidf.transform([article_text])
    sims = cosine_similarity(article_vec, tfidf_matrix[same_cat_idx]).flatten()
    top_idx = sims.argsort()[-top_n:][::-1]
    similar_articles = df.loc[same_cat_idx[top_idx], 'Text'].values
    return predicted_cat, similar_articles

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    recommendations = []
    article_text = ""
    if request.method == 'POST':
        article_text = request.form['article']
        prediction, recs = recommend_similar(article_text)
        recommendations = [r[:150] + "..." for r in recs]
    return render_template('index.html', prediction=prediction,
                           recommendations=recommendations, article_text=article_text)

if __name__ == '__main__':
    app.run(debug=True)