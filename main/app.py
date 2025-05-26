from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
from scipy.sparse import hstack
import pickle
import logging
import os
from uuid import uuid4
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8000", "http://localhost:63342"]}})

# Custom JSON encoder to handle NaN and other special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Vietnamese stop words
stop_words_vi = {'và', 'của', 'là', 'có', 'được', 'trong', 'cho', 'tại', 'bởi', 'với', 'như', 'rằng', 'nếu', 'khi'}

class DataProcessor:
    @staticmethod
    def standardize_degree(degree):
        if not isinstance(degree, str) or degree.strip() == '':
            return "Khác"
        degree = re.sub(r'Đơn vị đào tạo:?\s*', '', degree, flags=re.IGNORECASE)
        degree = re.sub(r'Mô tả công việc:.*', '', degree, flags=re.IGNORECASE)
        degree = re.sub(r'bằng\s*(giỏi|khá|trung bình).*', '', degree, flags=re.IGNORECASE)
        degree = degree.lower().strip()
        if re.search(r'đại học|thạc sĩ|tiến sĩ', degree):
            return "Đại học"
        elif re.search(r'cao đẳng', degree):
            return "Cao đẳng"
        elif re.search(r'trung cấp|trung cấp nghề', degree):
            return "Trung cấp"
        elif re.search(r'thpt|trung học phổ thông', degree):
            return "THPT"
        elif re.search(r'thcs|trung học cơ sở', degree):
            return "THCS"
        return "Khác"

    @staticmethod
    def standardize_location(location):
        if not isinstance(location, str) or location.strip() == '':
            return "Khác"
        location = location.lower().strip()
        # Chuẩn hóa các biến thể của địa điểm
        location_map = {
            r'hà nội|ha noi|hn': 'Hà Nội',
            r'hồ chí minh|tp hồ chí minh|hcm|ho chi minh': 'Hồ Chí Minh',
            r'đà nẵng|da nang': 'Đà Nẵng',
            r'bình dương|binh duong': 'Bình Dương',
            r'bắc giang|bac giang': 'Bắc Giang',
            # Thêm các địa điểm khác nếu cần
            r'.*': 'Khác'  # Các địa điểm không khớp với bất kỳ mẫu nào
        }
        for pattern, standard_name in location_map.items():
            if re.search(pattern, location):
                return standard_name
        return "Khác"

    @staticmethod
    def extract_degree_from_requirements(requirements):
        if not isinstance(requirements, str) or requirements.strip() == '':
            return "Khác"
        requirements = requirements.lower().strip()
        requirements = re.sub(r'[-–—]+.*$', '', requirements)
        requirements = re.sub(r'\d+\..*?$', '', requirements)
        if re.search(r'đại học|tốt nghiệp đại học|cao học|thạc sĩ|tiến sĩ', requirements):
            return "Đại học"
        elif re.search(r'cao đẳng|tốt nghiệp cao đẳng', requirements):
            return "Cao đẳng"
        elif re.search(r'trung cấp|trung cấp nghề', requirements):
            return "Trung cấp"
        elif re.search(r'thpt|trung học phổ thông', requirements):
            return "THPT"
        elif re.search(r'thcs|trung học cơ sở', requirements):
            return "THCS"
        return "Khác"

    @staticmethod
    def standardize_gender(gender):
        if not isinstance(gender, str) or gender.strip() == '':
            return "Không xác định"
        gender = gender.lower().strip()
        if re.search(r'nữ', gender):
            return "nữ"
        elif re.search(r'nam', gender):
            return "nam"
        elif re.search(r'không yêu cầu|không xác định', gender):
            return "Không xác định"
        return "Không xác định"

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        words = [word for word in words if word not in stop_words_vi]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    @staticmethod
    def load_and_preprocess_data():
        try:
            if os.path.exists("JOB_DATA_PROCESSED.csv") and os.path.exists("USER_DATA_PROCESSED.csv"):
                df_jobs = pd.read_csv("JOB_DATA_PROCESSED.csv")
                df_users = pd.read_csv("USER_DATA_PROCESSED.csv")
                logger.info("Loaded preprocessed data from CSV files.")
                logger.info("Loaded preprocessed data from CSV files.")
                logger.info(f"Sample Degree values after loading: {df_users['Degree'].head().to_list()}")
            else:
                df_jobs = pd.read_csv("JOB_DATA_FINAL.csv", encoding='utf-8')
                df_users = pd.read_csv("USER_DATA_FINAL.csv", encoding='utf-8')


                # Preprocess IDs
                df_jobs['JobID'] = df_jobs['JobID'].astype(str).str.strip()
                df_users['UserID'] = df_users['UserID'].astype(str).str.strip()

                # Chuẩn hóa địa điểm
                df_jobs['Job Address'] = df_jobs['Job Address'].apply(DataProcessor.standardize_location)
                df_users['Workplace Desired'] = df_users['Workplace Desired'].apply(DataProcessor.standardize_location)

                # Standardize Industry
                df_jobs['Industry'] = df_jobs['Industry'].str.strip().str.lower()
                df_users['Industry'] = df_users['Industry'].str.replace('-', '/').str.strip().str.lower()

                # Standardize Degree for users
                df_users['Degree'] = df_users['Degree'].apply(DataProcessor.standardize_degree)
                logger.info(f"Sample Degree values after standardization: {df_users['Degree'].head().to_list()}")

                # Extract and standardize Job Requirements
                df_jobs['Standardized Degree'] = df_jobs['Job Requirements'].apply(
                    DataProcessor.extract_degree_from_requirements)

                # Standardize Gender
                df_jobs['Gender'] = df_jobs['Gender'].apply(DataProcessor.standardize_gender)
                df_users['Gender'] = df_users['Gender'].apply(DataProcessor.standardize_gender)

                # Replace NaN with appropriate values for JSON compatibility
                df_jobs['Gender'] = df_jobs['Gender'].where(df_jobs['Gender'].notna(), "Không xác định")
                df_users['Gender'] = df_users['Gender'].where(df_users['Gender'].notna(), "Không xác định")
                df_users['Degree'] = df_users['Degree'].where(df_users['Degree'].notna(), "Khác")

                # Thay NaN bằng None cho toàn bộ DataFrame
                df_jobs = df_jobs.where(df_jobs.notna(), None)
                df_users = df_users.where(df_users.notna(), None)

                # Preprocess text columns
                text_columns_jobs = ['Job Description', 'Career Level', 'Job Title', 'Job Address', 'Salary',
                                     'Years of Experience']
                for col in text_columns_jobs:
                    if col in df_jobs.columns:
                        df_jobs[col] = df_jobs[col].fillna('').apply(DataProcessor.preprocess_text)

                text_columns_users = ['Skills', 'Desired Job', 'Workplace Desired', 'Target', 'Work Experience']
                for col in text_columns_users:
                    if col in df_users.columns:
                        df_users[col] = df_users[col].fillna('').apply(DataProcessor.preprocess_text)

                # Combine text for feature extraction
                df_jobs['Combined Text'] = df_jobs[[col for col in text_columns_jobs if col in df_jobs.columns]].agg(
                    ' '.join, axis=1)
                df_users['Combined Text'] = df_users[
                    [col for col in text_columns_users + ['Industry', 'Degree'] if col in df_users.columns]].agg(
                    ' '.join, axis=1)

                # Save preprocessed data
                df_jobs.to_csv("JOB_DATA_PROCESSED.csv", index=False)
                df_users.to_csv("USER_DATA_PROCESSED.csv", index=False)
                logger.info("Preprocessed data saved to CSV files.")

            return df_jobs, df_users

        except FileNotFoundError:
            logger.error("Data files not found.")
            raise
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

class RecommenderSystem:
    def __init__(self, df_jobs, df_users):
        self.df_jobs = df_jobs
        self.df_users = df_users
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.word2vec_model = None
        self.job_tfidf = None
        self.user_tfidf = None
        self.job_embeddings = None
        self.user_embeddings = None
        self.job_features = None
        self.user_features = None
        self.cosine_sim = None

    def train(self):
        logger.info("Training recommender system...")
        self.job_tfidf = self.tfidf.fit_transform(self.df_jobs['Combined Text'])
        self.user_tfidf = self.tfidf.transform(self.df_users['Combined Text'])
        job_tokens = [text.split() for text in self.df_jobs['Combined Text']]
        user_tokens = [text.split() for text in self.df_users['Combined Text']]
        self.word2vec_model = Word2Vec(sentences=job_tokens + user_tokens, vector_size=100, window=5, min_count=1, workers=4)
        self.job_embeddings = np.array([self.text_to_vector(tokens) for tokens in job_tokens])
        self.user_embeddings = np.array([self.text_to_vector(tokens) for tokens in user_tokens])
        self.job_features = hstack([self.job_tfidf, self.job_embeddings])
        self.user_features = hstack([self.user_tfidf, self.user_embeddings])
        self.cosine_sim = cosine_similarity(self.user_features, self.job_features)
        logger.info("Training completed.")

    def text_to_vector(self, tokens):
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word2vec_model.vector_size)

    def save_features(self):
        features = {
            'job_tfidf.pkl': self.job_tfidf,
            'user_tfidf.pkl': self.user_tfidf,
            'job_embeddings.pkl': self.job_embeddings,
            'user_embeddings.pkl': self.user_embeddings,
            'cosine_sim.pkl': self.cosine_sim
        }
        for filename, feature in features.items():
            with open(filename, 'wb') as f:
                pickle.dump(feature, f)
        logger.info("Saved precomputed features to files.")

    def load_features(self):
        feature_files = ['job_tfidf.pkl', 'user_tfidf.pkl', 'job_embeddings.pkl', 'user_embeddings.pkl', 'cosine_sim.pkl']
        if all(os.path.exists(f) for f in feature_files):
            for filename in feature_files:
                with open(filename, 'rb') as f:
                    setattr(self, filename.split('.')[0], pickle.load(f))
            logger.info("Loaded precomputed features from files.")
            return True
        return False

    def calculate_requirement_score(self, user_index, job_index):
        score = 1.0
        user = self.df_users.iloc[user_index]
        job = self.df_jobs.iloc[job_index]

        # Kiểm tra khớp địa điểm
        if job['Job Address'] != "Khác" and job['Job Address'] != user['Workplace Desired']:
            score *= 0.1  # Giảm mạnh điểm nếu địa điểm không khớp

        # Kiểm tra bằng cấp
        if job['Standardized Degree'] != "Khác" and job['Standardized Degree'] != user['Degree']:
            score *= 0.5

        # Kiểm tra giới tính
        if job['Gender'] != "Không xác định" and job['Gender'] != user['Gender']:
            score *= 0.4

        return score

    def update_model(self, new_user_df):
        self.df_users = pd.concat([self.df_users, new_user_df], ignore_index=True)
        self.df_users.to_csv('USER_DATA_PROCESSED.csv', index=False)
        text_columns = ['Skills', 'Desired Job', 'Workplace Desired', 'Target', 'Work Experience']
        for col in text_columns:
            self.df_users[col] = self.df_users[col].fillna('').apply(DataProcessor.preprocess_text)
        self.df_users['Gender'] = self.df_users['Gender'].apply(DataProcessor.standardize_gender)
        self.df_users['Combined Text'] = self.df_users[text_columns + ['Industry', 'Degree']].agg(' '.join, axis=1)
        self.user_tfidf = self.tfidf.transform(self.df_users['Combined Text'])
        user_tokens = [text.split() for text in self.df_users['Combined Text']]
        self.user_embeddings = np.array([self.text_to_vector(tokens) for tokens in user_tokens])
        self.user_features = hstack([self.user_tfidf, self.user_embeddings])
        self.cosine_sim = cosine_similarity(self.user_features, self.job_features)
        logger.info("Model updated with new user data.")

# Initialize and preprocess data
df_jobs, df_users = DataProcessor.load_and_preprocess_data()
recommender = RecommenderSystem(df_jobs, df_users)

# Load or train model
if not recommender.load_features():
    recommender.train()
    recommender.save_features()


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_index = int(data.get('user_index', 0))
        top_n = int(data.get('top_n', 5))
        if user_index < 0 or user_index >= len(recommender.df_users):
            return jsonify({"error": "Invalid user index"}), 400

        user_info = recommender.df_users.iloc[user_index][
            ['User Name', 'Desired Job', 'Industry', 'Skills', 'Workplace Desired', 'Desired Salary',
             'Gender', 'Marriage', 'Age', 'Target', 'Degree', 'Work Experience', 'URL User']].to_dict()
        logger.info(f"Degree returned to frontend: {user_info['Degree']}")

        user_location = user_info['Workplace Desired']
        if user_location != "Khác":
            location_mask = (recommender.df_jobs['Job Address'] == user_location) | (
                        recommender.df_jobs['Job Address'] == "Khác")
            filtered_job_indices = recommender.df_jobs[location_mask].index
        else:
            filtered_job_indices = recommender.df_jobs.index

        similarity_scores = recommender.cosine_sim[user_index][filtered_job_indices]
        requirement_scores = np.array([recommender.calculate_requirement_score(user_index, job_index)
                                       for job_index in filtered_job_indices])
        combined_scores = similarity_scores * requirement_scores
        recommended_job_indices = filtered_job_indices[combined_scores.argsort()[::-1][:top_n]]

        recommended_jobs = []
        for job_index in recommended_job_indices:
            job_info = recommender.df_jobs.iloc[job_index][
                ['JobID', 'Job Title', 'Name Company', 'Career Level', 'Salary', 'Job Address']].to_dict()
            job_info['Similarity Score'] = float(combined_scores[list(filtered_job_indices).index(job_index)])
            recommended_jobs.append(job_info)

        return jsonify({'user_info': user_info, 'recommended_jobs': recommended_jobs})
    except Exception as e:
        logger.error(f"Error in /recommend: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/job_detail', methods=['GET'])
def job_detail():
    try:
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({"error": "No job_id provided"}), 400
        job_id = str(job_id).strip()
        job_row = recommender.df_jobs[recommender.df_jobs['JobID'] == job_id]
        if job_row.empty:
            return jsonify({"error": f"Job with ID {job_id} not found"}), 404
        job_dict = job_row.iloc[0].to_dict()
        return jsonify(job_dict)
    except Exception as e:
        logger.error(f"Error in /job_detail: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/popular_jobs', methods=['GET'])
def popular_jobs():
    try:
        top_n = min(100, len(recommender.df_jobs))
        popular_jobs = recommender.df_jobs.head(top_n).to_dict('records')
        return jsonify(popular_jobs)
    except Exception as e:
        logger.error(f"Error in /popular_jobs: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/save_cv', methods=['POST'])
def save_cv():
    try:
        data = request.get_json()
        new_cv = {
            'UserID': str(uuid4()),
            'User Name': data.get('name', ''),
            'Email': data.get('email', ''),
            'Phone': data.get('phone', ''),
            'Desired Job': data.get('desiredJob', ''),
            'Industry': data.get('industry', '').lower().replace('-', '/'),
            'Workplace Desired': DataProcessor.standardize_location(data.get('workplace', '')),
            'Desired Salary': data.get('desiredSalary', ''),
            'Gender': DataProcessor.standardize_gender(data.get('gender', 'Không xác định')),
            'Marriage': data.get('marriage', ''),
            'Age': data.get('age', ''),
            'Target': data.get('target', ''),
            'Skills': data.get('skills', ''),
            'Degree': DataProcessor.standardize_degree(data.get('degree', '')),
            'Work Experience': data.get('workExperience', ''),
            'URL User': data.get('url_user', '')  # Thêm trường URL User
        }
        new_user_df = pd.DataFrame([new_cv])
        recommender.update_model(new_user_df)
        new_user_index = len(recommender.df_users) - 1

        user_location = recommender.df_users.iloc[new_user_index]['Workplace Desired']

        # Lọc công việc theo địa điểm
        if user_location != "Khác":
            location_mask = (recommender.df_jobs['Job Address'] == user_location) | (
                        recommender.df_jobs['Job Address'] == "Khác")
            filtered_job_indices = recommender.df_jobs[location_mask].index
        else:
            filtered_job_indices = recommender.df_jobs.index

        similarity_scores = recommender.cosine_sim[new_user_index][filtered_job_indices]
        requirement_scores = np.array([recommender.calculate_requirement_score(new_user_index, job_index)
                                       for job_index in filtered_job_indices])
        combined_scores = similarity_scores * requirement_scores
        recommended_job_indices = filtered_job_indices[combined_scores.argsort()[::-1][:5]]

        recommended_jobs = []
        for job_index in recommended_job_indices:
            job_info = recommender.df_jobs.iloc[job_index][
                ['JobID', 'Job Title', 'Name Company', 'Career Level', 'Salary', 'Job Address']].to_dict()
            job_info['Similarity Score'] = float(combined_scores[list(filtered_job_indices).index(job_index)])
            recommended_jobs.append(job_info)

        user_info = recommender.df_users.iloc[new_user_index][
            ['User Name', 'Desired Job', 'Industry', 'Skills', 'Workplace Desired', 'Desired Salary',
             'Gender', 'Marriage', 'Age', 'Target', 'Degree', 'Work Experience', 'URL User']].to_dict()  # Thêm URL User
        return jsonify({'user_info': user_info, 'recommended_jobs': recommended_jobs})
    except Exception as e:
        logger.error(f"Error in /save_cv: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=False)