from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8000", "http://localhost:63342"]}})

# Load dữ liệu
df_jobs = pd.read_csv("JOB_DATA_FINAL.csv")
df_users = pd.read_csv("USER_DATA_FINAL.csv")

# Preprocess JobID at startup
df_jobs['JobID'] = df_jobs['JobID'].astype(str).str.strip()
df_users['UserID'] = df_users['UserID'].astype(str).str.strip()

# Tiền xử lý
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Initial preprocessing
df_jobs['Job Description'] = df_jobs['Job Description'].fillna('').apply(preprocess_text)
df_users['Skills'] = df_users['Skills'].fillna('').apply(preprocess_text)

# Kết hợp văn bản
df_jobs['Combined Text'] = df_jobs['Job Description'].fillna('') + ' ' + df_jobs['Career Level'].fillna('') + ' ' + \
                           df_jobs['Job Title'].fillna('') + ' ' + df_jobs['Company Address'].fillna('')
df_users['Combined Text'] = df_users['Desired Job'].fillna('') + ' ' + df_users['Industry'].fillna('') + ' ' + df_users[
    'Skills'].fillna('')

# Trích xuất đặc trưng TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
job_tfidf = tfidf.fit_transform(df_jobs['Combined Text'])
user_tfidf = tfidf.transform(df_users['Combined Text'])

# Tokenize
job_tokens = [word_tokenize(text) for text in df_jobs['Combined Text']]
user_tokens = [word_tokenize(text) for text in df_users['Combined Text']]

# Huấn luyện Word2Vec
word2vec_model = Word2Vec(sentences=job_tokens + user_tokens, vector_size=100, window=5, min_count=1, workers=4)


def text_to_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


job_embeddings = np.array([text_to_vector(tokens, word2vec_model) for tokens in job_tokens])
user_embeddings = np.array([text_to_vector(tokens, word2vec_model) for tokens in user_tokens])

# Kết hợp đặc trưng
job_features = hstack([job_tfidf, job_embeddings])
user_features = hstack([user_tfidf, user_embeddings])

# Tính toán cosine similarity
cosine_sim = cosine_similarity(user_features, job_features)


def update_model(new_user_df):
    global df_users, user_tfidf, user_embeddings, user_features, cosine_sim

    # Append new user
    df_users = pd.concat([df_users, new_user_df], ignore_index=True)

    # Save to CSV
    df_users.to_csv('USER_DATA_FINAL.csv', index=False)

    # Update preprocessing
    df_users['Skills'] = df_users['Skills'].fillna('').apply(preprocess_text)
    df_users['Combined Text'] = df_users['Desired Job'].fillna('') + ' ' + df_users['Industry'].fillna('') + ' ' + \
                                df_users['Skills'].fillna('')

    # Update TF-IDF
    user_tfidf = tfidf.transform(df_users['Combined Text'])

    # Update Word2Vec tokens
    user_tokens = [word_tokenize(text) for text in df_users['Combined Text']]
    user_embeddings = np.array([text_to_vector(tokens, word2vec_model) for tokens in user_tokens])

    # Update combined features
    user_features = hstack([user_tfidf, user_embeddings])

    # Recalculate cosine similarity
    cosine_sim = cosine_similarity(user_features, job_features)


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_index = int(data.get('user_index', 0))
    top_n = int(data.get('top_n', 5))

    if user_index < 0 or user_index >= len(df_users):
        return jsonify({"error": "Invalid user index"}), 400

    recommended_job_indices = cosine_sim.argsort(axis=1)[:, ::-1]
    recommended_jobs = []
    for job_index in recommended_job_indices[user_index][:top_n]:
        job_info = df_jobs.iloc[job_index][
            ['JobID', 'Job Title', 'Name Company', 'Career Level', 'Salary', 'Job Address']].to_dict()
        job_info['Similarity Score'] = float(cosine_sim[user_index, job_index])
        recommended_jobs.append(job_info)

    user_info = df_users.loc[
        user_index, ['User Name', 'Desired Job', 'Industry', 'Skills', 'Workplace Desired', 'Desired Salary', 'Gender',
                     'Marriage', 'Age', 'Target', 'Degree', 'Work Experience']].to_dict()
    return jsonify({
        'user_info': user_info,
        'recommended_jobs': recommended_jobs
    })


@app.route('/job_detail', methods=['GET'])
def job_detail():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "No job_id provided"}), 400

    job_id = str(job_id).strip()
    job_row = df_jobs[df_jobs['JobID'] == job_id]
    if job_row.empty:
        return jsonify({"error": f"Job with ID {job_id} not found"}), 404
    job = job_row.iloc[0].to_dict()
    return jsonify(job)


@app.route('/popular_jobs', methods=['GET'])
def popular_jobs():
    # Return top 100 jobs (or all if less than 100)
    top_n = min(100, len(df_jobs))
    popular_jobs = df_jobs.head(top_n).to_dict('records')
    return jsonify(popular_jobs)


@app.route('/save_cv', methods=['POST'])
def save_cv():
    data = request.get_json()
    new_cv = {
        'UserID': str(len(df_users) + 1),  # Generate new UserID
        'User Name': data.get('name', ''),
        'Email': data.get('email', ''),
        'Phone': data.get('phone', ''),
        'Desired Job': data.get('desiredJob', ''),
        'Industry': data.get('industry', ''),
        'Workplace Desired': data.get('workplace', ''),
        'Desired Salary': data.get('desiredSalary', ''),
        'Gender': data.get('gender', ''),
        'Marriage': data.get('marriage', ''),
        'Age': data.get('age', ''),
        'Target': data.get('target', ''),
        'Skills': data.get('skills', ''),
        'Degree': data.get('degree', ''),
        'Work Experience': data.get('workExperience', '')
    }

    # Convert to DataFrame and update model
    new_user_df = pd.DataFrame([new_cv])
    update_model(new_user_df)

    # Get recommendations for the new user
    new_user_index = len(df_users) - 1
    recommended_job_indices = cosine_sim.argsort(axis=1)[:, ::-1]
    recommended_jobs = []
    top_n = 5  # Default number of recommendations
    for job_index in recommended_job_indices[new_user_index][:top_n]:
        job_info = df_jobs.iloc[job_index][
            ['JobID', 'Job Title', 'Name Company', 'Career Level', 'Salary', 'Job Address']].to_dict()
        job_info['Similarity Score'] = float(cosine_sim[new_user_index, job_index])
        recommended_jobs.append(job_info)

    # Return the new user info and recommendations
    user_info = df_users.loc[
        new_user_index, ['User Name', 'Desired Job', 'Industry', 'Skills', 'Workplace Desired', 'Desired Salary',
                         'Gender', 'Marriage', 'Age', 'Target', 'Degree', 'Work Experience']].to_dict()
    return jsonify({
        'user_info': user_info,
        'recommended_jobs': recommended_jobs
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)