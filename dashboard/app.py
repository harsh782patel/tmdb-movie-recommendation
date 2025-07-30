import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import duckdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime
import logging
import json
from scipy.sparse import save_npz, load_npz

MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODEL_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('model_training')

def load_movie_data():
    """Load movie data from DuckDB database"""
    try:
        db_path = os.path.join(DATA_DIR, 'movies.duckdb')
        conn = duckdb.connect(db_path)
        movies = conn.execute("""
            SELECT 
                id, title, overview, genres, 
                vote_average, vote_count, popularity,
                release_date, poster_path,
                revenue, runtime, budget
            FROM movies
        """).fetchdf()
        conn.close()
        
        if movies.empty:
            logger.warning("No movies found in database. Check data ingestion.")
            return pd.DataFrame()
            
        logger.info(f"Successfully loaded {len(movies)} movies from database")
        return movies
        
    except Exception as e:
        logger.error(f"Error loading data from DuckDB: {str(e)}")
        return pd.DataFrame()

def preprocess_data(movies):
    """Clean and prepare movie data for modeling"""
    if movies.empty:
        return movies
        
    # Handle missing values
    movies['overview'] = movies['overview'].fillna('')
    movies['vote_count'] = movies['vote_count'].fillna(0).astype(int)
    movies['vote_average'] = movies['vote_average'].fillna(0)
    movies['genres'] = movies['genres'].fillna('')
    movies['poster_path'] = movies['poster_path'].fillna('')
    
    # Create enhanced content field
    movies['content'] = movies['title'] + ' ' + movies['overview'] + ' ' + movies['genres']
    
    # Calculate weighted ratings (IMDB formula)
    m = movies['vote_count'].quantile(0.8)
    C = movies['vote_average'].mean()
    movies['weighted_rating'] = movies.apply(
        lambda x: (x['vote_count'] * x['vote_average'] + m * C) / (x['vote_count'] + m),
        axis=1
    )
    
    # Normalize ratings
    scaler = MinMaxScaler()
    movies['norm_rating'] = scaler.fit_transform(movies[['weighted_rating']])
    movies['last_updated'] = datetime.now()
    
    return movies

def train_content_model(movies):
    """Train content-based recommendation model"""
    logger.info("Training content-based model...")
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000,
        dtype=np.float32
    )
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    
    logger.info("Computing cosine similarity...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create movie index mapping
    indices = pd.Series(movies.index, index=movies['id']).drop_duplicates()
    
    # Precompute top 50 recommendations
    logger.info("Precomputing top recommendations...")
    recommendations = {}
    k = 50
    
    for idx, movie_id in enumerate(movies['id']):
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]  # Skip self
        movie_recs = []
        
        for i, score in sim_scores:
            rec_id = movies.iloc[i]['id']
            movie_recs.append((rec_id, float(score)))
            
        recommendations[movie_id] = movie_recs
    
    logger.info(f"Precomputed recommendations for {len(recommendations)} movies")
    return tfidf, recommendations, indices

def save_models(tfidf, recommendations, indices, movies):
    """Persist models and data to disk"""
    try:
        joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
        joblib.dump(recommendations, os.path.join(MODEL_DIR, 'recommendations.pkl'))
        joblib.dump(indices, os.path.join(MODEL_DIR, 'indices.pkl'))
        movies.to_pickle(os.path.join(MODEL_DIR, 'movies_df.pkl'))
        
        valid_dates = movies['release_date'].dropna()
        min_year = max_year = None
        
        if not valid_dates.empty:
            years = valid_dates.str[:4]
            numeric_years = pd.to_numeric(years, errors='coerce').dropna()
            
            if not numeric_years.empty:
                min_year = int(numeric_years.min())
                max_year = int(numeric_years.max())

        metadata = {
            'trained_at': datetime.now().isoformat(),
            'movie_count': int(len(movies)),
            'min_id': int(movies['id'].min()),
            'max_id': int(movies['id'].max()),
            'min_year': min_year,
            'max_year': max_year,
            'genres': list(set(g for genres in movies['genres'] 
                             for g in genres.split(', ') if g))
        }
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved all models and data to {MODEL_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        return False

def test_recommendations(movies, recommendations, sample_size=3):
    """Test recommendations with sample movies"""
    if len(movies) < sample_size:
        logger.warning("Not enough movies for testing")
        return
        
    logger.info("Testing recommendations...")
    sample_movies = movies.sample(sample_size)
    
    for _, movie in sample_movies.iterrows():
        movie_id = movie['id']
        
        if movie_id not in recommendations:
            logger.warning(f"Movie ID {movie_id} not in recommendations")
            continue
            
        recs = recommendations[movie_id][:3]  # Top 3
        
        try:
            title = movie['title']
            logger.info(f"\nRecommendations for '{title}':")
        except UnicodeEncodeError:
            safe_title = movie['title'].encode('utf-8', 'ignore').decode('utf-8')
            logger.info(f"\nRecommendations for '{safe_title}':")
        
        for rec_id, score in recs:
            rec_movie = movies[movies['id'] == rec_id].iloc[0]
            try:
                logger.info(f"- {rec_movie['title']} (ID: {rec_id}, Similarity: {score:.4f})")
            except UnicodeEncodeError:
                safe_title = rec_movie['title'].encode('utf-8', 'ignore').decode('utf-8')
                logger.info(f"- {safe_title} (ID: {rec_id}, Similarity: {score:.4f})")

def train_models():
    """Main function to train all recommendation models"""
    logger.info("Starting model training process...")
    
    movies = load_movie_data()
    if movies.empty:
        logger.error("Aborting training due to empty dataset")
        return False
    
    movies = preprocess_data(movies)
    logger.info(f"Data preprocessing complete. Features: {list(movies.columns)}")
    
    tfidf, recommendations, indices = train_content_model(movies)
    success = save_models(tfidf, recommendations, indices, movies)
    test_recommendations(movies, recommendations)
    
    return success

if __name__ == "__main__":
    logger.info("Movie Recommendation Model Training")
    logger.info("==================================")
    
    start_time = datetime.now()
    success = train_models()
    duration = datetime.now() - start_time
    
    if success:
        logger.info(f"Model training completed in {duration.total_seconds():.2f} seconds")
    else:
        logger.error("Model training failed")