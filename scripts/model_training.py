import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import duckdb
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime
import logging
import json
import nmslib
import gc
import psutil
import threading
import time
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Memory optimization settings
MAX_MEMORY_USAGE = 0.85  # 85% of available RAM (6.8GB)
INITIAL_BATCH_SIZE = 20000  # Start with this batch size

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(os.path.join(MODEL_DIR, 'training.log')), logging.StreamHandler()])
logger = logging.getLogger('model_training')

def safe_str(s):
    """Safely convert strings to UTF-8 for logging"""
    return s.encode('utf-8', 'ignore').decode('utf-8') if isinstance(s, str) else str(s)

def memory_usage():
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent / 100

def optimize_memory(df):
    """Optimize DataFrame memory usage"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def load_movie_data():
    """Load movie data from DuckDB database with memory optimization"""
    try:
        db_path = os.path.join(DATA_DIR, 'movies.duckdb')
        conn = duckdb.connect(db_path)
        
        # Load in chunks for memory efficiency
        chunk_size = 100000
        offset = 0
        all_movies = []
        
        while True:
            query = f"""
                SELECT 
                    id, title, overview, genres, 
                    vote_average, vote_count, popularity,
                    release_date, poster_path
                FROM movies
                LIMIT {chunk_size} OFFSET {offset}
            """
            chunk = conn.execute(query).fetchdf()
            if chunk.empty:
                break
                
            optimized_chunk = optimize_memory(chunk)
            all_movies.append(optimized_chunk)
            offset += chunk_size
            gc.collect()
            
        conn.close()
        
        if not all_movies:
            logger.warning("No movies found in database. Check data ingestion.")
            return pd.DataFrame()
            
        movies = pd.concat(all_movies, ignore_index=True)
        logger.info(f"Successfully loaded {len(movies)} movies from database")
        return movies
        
    except Exception as e:
        logger.error(f"Error loading data from DuckDB: {str(e)}")
        return pd.DataFrame()

def preprocess_data(movies):
    """Clean and prepare movie data for modeling with memory optimization"""
    if movies.empty:
        return movies, []
        
    # Handle missing values
    movies['title'] = movies['title'].fillna('')
    movies['overview'] = movies['overview'].fillna('')
    movies['vote_count'] = movies['vote_count'].fillna(0).astype('int32')
    movies['vote_average'] = movies['vote_average'].fillna(0).astype('float32')
    movies['genres'] = movies['genres'].fillna('')
    movies['poster_path'] = movies['poster_path'].fillna('')
    
    # Create enhanced content field
    movies['content'] = movies['title'] + ' ' + movies['overview'] + ' ' + movies['genres']
    movies['content'] = movies['content'].fillna('')
    
    # Calculate weighted ratings (IMDB formula)
    m = movies['vote_count'].quantile(0.8)
    C = movies['vote_average'].mean()
    
    # Optimize memory for calculations
    vote_count = movies['vote_count'].values.astype('float32')
    vote_avg = movies['vote_average'].values.astype('float32')
    
    # Vectorized calculation
    weighted_rating = (vote_count * vote_avg + m * C) / (vote_count + m)
    movies['weighted_rating'] = weighted_rating.astype('float32')
    
    # Normalize ratings
    scaler = MinMaxScaler()
    movies['norm_rating'] = scaler.fit_transform(movies[['weighted_rating']]).astype('float32')
    movies['last_updated'] = datetime.now()
    
    # Extract unique genres before dropping column
    unique_genres = set()
    for genres in movies['genres']:
        if isinstance(genres, str):
            for g in genres.split(', '):
                if g:
                    unique_genres.add(g)
    
    # Drop intermediate columns to save memory
    movies = movies.drop(columns=['overview', 'genres', 'vote_count', 'vote_average', 'popularity'])
    
    # Convert IDs to Python int for DuckDB compatibility
    movies['id'] = movies['id'].astype(int)
    
    gc.collect()
    return movies, list(unique_genres)

def build_tfidf_matrix(movies):
    """Build TF-IDF matrix in batches to conserve memory"""
    logger.info("Building TF-IDF matrix in batches...")
    
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000,
        dtype=np.float32
    )
    
    # Process in chunks
    chunk_size = 100000
    tfidf_matrix = None
    
    for i in range(0, len(movies), chunk_size):
        chunk = movies['content'].iloc[i:i+chunk_size]
        
        if i == 0:
            chunk_matrix = tfidf.fit_transform(chunk)
        else:
            chunk_matrix = tfidf.transform(chunk)
        
        if tfidf_matrix is None:
            tfidf_matrix = chunk_matrix
        else:
            tfidf_matrix = vstack([tfidf_matrix, chunk_matrix])
        
        logger.info(f"Processed {min(i+chunk_size, len(movies))}/{len(movies)} documents")
        gc.collect()
    
    logger.info("TF-IDF matrix construction complete")
    return tfidf, tfidf_matrix

def train_content_model(movies):
    """Train content-based recommendation model with CPU optimization"""
    logger.info("Training content-based model with ANN (nmslib)...")
    
    # Reset index for consistent indexing
    movies = movies.reset_index(drop=True)
    
    # Build TF-IDF matrix
    tfidf, tfidf_matrix = build_tfidf_matrix(movies)
    
    # Free memory
    if 'content' in movies.columns:
        del movies['content']
    gc.collect()
    
    # Build ANN index with memory-optimized parameters
    logger.info("Building ANN index...")
    index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    index.addDataPointBatch(tfidf_matrix)
    
    # Optimized parameters for 8GB RAM
    index_params = {
        'M': 12,                     # Lower memory usage
        'indexThreadQty': 4,          # Fewer threads for memory
        'efConstruction': 100,        # Lower construction accuracy for memory
        'post': 0                     # Disable expensive post-processing
    }
    index.createIndex(index_params, print_progress=True)
    
    # Precompute top recommendations in batches
    logger.info("Finding nearest neighbors in batches...")
    db_path = os.path.join(MODEL_DIR, 'models.duckdb')
    conn = duckdb.connect(db_path)
    conn.execute("CREATE OR REPLACE TABLE recommendations (movie_id BIGINT, rec_id BIGINT, score FLOAT)")
    
    # Use local variable for dynamic batch size
    current_batch_size = INITIAL_BATCH_SIZE
    
    # Process in batches to conserve memory
    i = 0
    while i < len(movies):
        # Adjust batch size based on current memory usage
        current_mem = memory_usage()
        if current_mem > MAX_MEMORY_USAGE:
            current_batch_size = max(5000, int(current_batch_size * 0.7))
            logger.warning(f"High memory usage ({current_mem*100:.1f}%), reducing batch size to {current_batch_size}")
        
        batch_end = min(i + current_batch_size, len(movies))
        batch_size = batch_end - i
        logger.info(f"Processing batch {i} to {batch_end} (size: {batch_size})")
        
        # Get batch matrix
        batch_matrix = tfidf_matrix[i:batch_end]
        
        # Query the index with fewer neighbors if memory is tight
        k = 51  # 50 neighbors + self
        if current_mem > 0.8:  # If memory is very tight
            k = 31  # Only get 30 neighbors
        
        # Query the index
        neighbors = index.knnQueryBatch(batch_matrix, k=k, num_threads=2)
        
        # Prepare batch data - ensure native Python types
        batch_data = []
        for j, (ids, dists) in enumerate(neighbors):
            idx = i + j
            movie_id = movies.iloc[idx]['id']
            
            # Skip self and get top recommendations
            for k_idx in range(1, min(len(ids), 51)):  # Ensure we don't go out of bounds
                rec_idx = ids[k_idx]
                rec_id = movies.iloc[rec_idx]['id']
                similarity = 1.0 - dists[k_idx]  # Convert distance to similarity
                
                # Convert to native Python types for DuckDB compatibility
                batch_data.append((
                    int(movie_id),     # Convert to Python int
                    int(rec_id),       # Convert to Python int
                    float(similarity)  # Convert to Python float
                ))
        
        # Insert into DuckDB
        if batch_data:
            conn.executemany(
                "INSERT INTO recommendations VALUES (?, ?, ?)", 
                batch_data
            )
        
        # Free memory
        del neighbors, batch_data
        gc.collect()
        
        # Move to next batch
        i = batch_end
    
    # Create movie index mapping
    logger.info("Creating movie indices...")
    indices = pd.Series(movies.index, index=movies['id']).reset_index()
    indices.columns = ['id', 'idx']
    
    # Convert to native Python types
    indices['id'] = indices['id'].astype(int)
    indices['idx'] = indices['idx'].astype(int)
    
    conn.execute("CREATE OR REPLACE TABLE movie_indices (id BIGINT, idx INTEGER)")
    conn.executemany(
        "INSERT INTO movie_indices VALUES (?, ?)", 
        indices[['id', 'idx']].to_records(index=False).tolist()
    )
    
    # Save preprocessed movies
    logger.info("Saving preprocessed movies...")
    movies.to_sql('preprocessed_movies', conn, if_exists='replace', index=False)
    
    conn.close()
    logger.info("ANN processing complete")
    
    return tfidf

def save_models(tfidf, movies, unique_genres):
    """Persist models and data to disk with memory optimization"""
    try:
        logger.info("Saving TF-IDF model...")
        joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_model.pkl'), protocol=4)
        
        # Extract metadata
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
            'genres': unique_genres  # Use precomputed genres
        }
        
        # Save metadata
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved models and metadata to {MODEL_DIR}")
        return True
        
    except Exception as e:
        logger.exception(f"Error saving models: {str(e)}")
        return False

def test_recommendations(movies, sample_size=3):
    """Test recommendations with sample movies using DuckDB"""
    if len(movies) < sample_size:
        logger.warning("Not enough movies for testing")
        return
        
    logger.info("Testing recommendations...")
    sample_movies = movies.sample(sample_size)
    
    db_path = os.path.join(MODEL_DIR, 'models.duckdb')
    conn = duckdb.connect(db_path)
    
    for _, movie in sample_movies.iterrows():
        movie_id = movie['id']
        
        recs = conn.execute(f"""
            SELECT rec_id, score 
            FROM recommendations 
            WHERE movie_id = {movie_id}
            ORDER BY score DESC
            LIMIT 3
        """).fetchdf()
        
        if recs.empty:
            logger.warning(f"No recommendations found for movie ID {movie_id}")
            continue
            
        title = safe_str(movie['title'])
        logger.info(f"\nRecommendations for '{title}':")
        
        for _, rec in recs.iterrows():
            rec_id = rec['rec_id']
            score = rec['score']
            rec_movie = movies[movies['id'] == rec_id].iloc[0]
            rec_title = safe_str(rec_movie['title'])
            logger.info(f"- {rec_title} (ID: {rec_id}, Similarity: {score:.4f})")
    
    conn.close()

def memory_monitor():
    """Monitor memory usage during training"""
    logger.info("Starting memory monitor thread...")
    while not training_complete:
        mem = psutil.virtual_memory()
        logger.info(f"Memory usage: {mem.percent}% (Used: {mem.used/1024**3:.2f}GB, Available: {mem.available/1024**3:.2f}GB)")
        time.sleep(60)

def train_models():
    """Main function to train all recommendation models"""
    global training_complete
    training_complete = False
    
    logger.info("Starting model training process...")
    
    # Start memory monitor thread
    mem_thread = threading.Thread(target=memory_monitor, daemon=True)
    mem_thread.start()
    
    # Load and preprocess data
    movies = load_movie_data()
    if movies.empty:
        logger.error("Aborting training due to empty dataset")
        return False
    
    movies, unique_genres = preprocess_data(movies)
    logger.info(f"Data preprocessing complete. Features: {list(movies.columns)}")
    
    # Train model
    tfidf = train_content_model(movies)
    
    # Save models
    success = save_models(tfidf, movies, unique_genres)
    
    # Test recommendations
    test_recommendations(movies)
    
    training_complete = True
    return success

if __name__ == "__main__":
    training_complete = False
    
    logger.info("Movie Recommendation Model Training - Optimized CPU Version")
    logger.info("==========================================================")
    logger.info(f"Hardware Constraints: 8GB RAM")
    logger.info(f"Initial Batch Size: {INITIAL_BATCH_SIZE}")
    
    start_time = datetime.now()
    success = train_models()
    duration = datetime.now() - start_time
    
    if success:
        logger.info(f"Model training completed in {duration.total_seconds()/60:.2f} minutes")
    else:
        logger.error("Model training failed")