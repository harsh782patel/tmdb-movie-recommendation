import schedule
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import pandas as pd
import duckdb
import os
import json
import re
import logging
import argparse
from datetime import datetime

# Configuration
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'ingestion.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_ingestion')

# Get API key from environment variable with fallback
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "e835a4cfc2ffd54fedcfc4d94f80b4fe")

def create_session_with_retries():
    """Create requests session with retry strategy"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=10,
        backoff_factor=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )
    session.mount("https://", adapter)
    
    proxies = {}
    if os.getenv('HTTP_PROXY'):
        proxies['http'] = os.getenv('HTTP_PROXY')
    if os.getenv('HTTPS_PROXY'):
        proxies['https'] = os.getenv('HTTPS_PROXY')
    if proxies:
        session.proxies = proxies
    
    return session

def fetch_tmdb_data(pages=50):
    """Fetch multiple pages of movie data from TMDB API"""
    base_url = "https://api.themoviedb.org/3/movie/popular"
    headers = {
        "Accept": "application/json",
        "User-Agent": "MovieRecommender/1.0",
        "RateLimit-Limit": "40",
        "RateLimit-Remaining": "39",
        "RateLimit-Reset": "10"
    }
    
    all_movies = []
    total_movies = 0
    session = None
    
    try:
        session = create_session_with_retries()
        logger.info(f"Fetching {pages} pages from TMDB API...")
        
        # First fetch genre mappings
        genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}"
        genre_response = session.get(genre_url)
        genre_data = genre_response.json()
        genre_map = {g['id']: g['name'] for g in genre_data.get('genres', [])}
        
        for page in range(1, pages + 1):
            logger.info(f"Fetching page {page}/{pages}...")
            params = {
                'api_key': TMDB_API_KEY,
                'page': page
            }
            
            try:
                timeout = 10 + (page % 3)
                response = session.get(base_url, headers=headers, params=params, timeout=timeout)
                logger.info(f"  Page {page} status: {response.status_code}, time: {response.elapsed.total_seconds():.2f}s")
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' not in data:
                    logger.warning(f"Warning: Unexpected response format on page {page}")
                    continue
                    
                # Process each movie to add genres and keep poster_path
                page_movies = []
                for movie in data['results']:
                    movie['genres'] = ', '.join([genre_map[gid] for gid in movie.get('genre_ids', []) 
                                              if gid in genre_map])
                    movie['poster_path'] = movie.get('poster_path', '')
                    page_movies.append(movie)
                
                movies_count = len(page_movies)
                all_movies.extend(page_movies)
                total_movies += movies_count
                logger.info(f"  Added {movies_count} movies from page {page}")
                
                delay = 0.5 + (page % 10)/10
                if page < pages:
                    time.sleep(delay)
                    
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"Warning: Failed for page {page}: {str(e)}")
                time.sleep(2)
                continue
                
        logger.info(f"\nSuccessfully fetched {total_movies} movies from {pages} pages")
        return pd.DataFrame(all_movies)
        
    except Exception as e:
        logger.error(f"Warning: Unexpected error: {str(e)}")
        return pd.DataFrame(all_movies)
    finally:
        if session:
            session.close()

def load_kaggle_dataset():
    """Load and process Kaggle TMDB dataset"""
    kaggle_path = os.path.join(DATA_DIR, 'TMDB_movie_dataset_v11.csv')
    if not os.path.exists(kaggle_path):
        logger.warning("Kaggle dataset not found")
        return pd.DataFrame()
    
    try:
        logger.info("Loading Kaggle TMDB dataset...")
        df = pd.read_csv(kaggle_path)
        
        # Map Kaggle columns to existing schema + new columns
        column_mapping = {
            'id': 'id',
            'title': 'title',
            'vote_average': 'vote_average',
            'vote_count': 'vote_count',
            'release_date': 'release_date',
            'revenue': 'revenue',
            'runtime': 'runtime',
            'adult': 'adult',
            'backdrop_path': 'backdrop_path',
            'budget': 'budget',
            'homepage': 'homepage',
            'imdb_id': 'imdb_id',
            'original_language': 'original_language',
            'original_title': 'original_title',
            'overview': 'overview',
            'popularity': 'popularity',
            'poster_path': 'poster_path',
            'tagline': 'tagline',
            'genres': 'genres',
            'production_companies': 'production_companies',
            'production_countries': 'production_countries',
            'spoken_languages': 'spoken_languages',
            'keywords': 'keywords'
        }
        df = df.rename(columns=column_mapping)
        return df
    except Exception as e:
        logger.error(f"Error loading Kaggle dataset: {str(e)}")
        return pd.DataFrame()

def load_sample_data():
    """Load sample data from file when API fails"""
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    try:
        logger.info(f"Loading sample data from {sample_path}")
        with open(sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        logger.warning("Sample data file not found")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logger.warning("Error decoding sample data")
        return pd.DataFrame()

def clean_movie_data(df):
    """Clean and prepare movie data for storage"""
    if df.empty:
        return df
        
    # Ensure ID is integer
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce', downcast='integer')
        df = df.dropna(subset=['id'])
        df['id'] = df['id'].astype(int)
        
        # Remove duplicate IDs (fix for constraint violation)
        duplicate_count = df.duplicated(subset=['id']).sum()
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} duplicate movie IDs - removing duplicates")
            df = df.drop_duplicates(subset=['id'], keep='first')
    
    # Clean release_date format
    if 'release_date' in df.columns:
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        valid_dates = df['release_date'].apply(
            lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False
        )
        invalid_count = (~valid_dates).sum()
        if invalid_count:
            logger.info(f"Found {invalid_count} invalid date formats - cleaning...")
            df.loc[~valid_dates, 'release_date'] = None
        current_year = datetime.now().year
        future_dates = df['release_date'].apply(
            lambda x: int(x[:4]) > current_year + 5 if pd.notnull(x) and re.match(pattern, x) else False
        )
        if future_dates.sum() > 0:
            logger.info(f"Found {future_dates.sum()} future dates - cleaning...")
            df.loc[future_dates, 'release_date'] = None

    # Convert numeric columns
    numeric_cols = ['vote_count', 'vote_average', 'popularity', 
                   'revenue', 'runtime', 'budget']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert boolean columns
    if 'adult' in df.columns:
        df['adult'] = df['adult'].astype(bool)
    
    # Handle list-type columns
    list_cols = ['genres', 'production_companies', 'production_countries',
                'spoken_languages', 'keywords']
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
    
    return df

def store_data(api_df, kaggle_df):
    """Store data in DuckDB with merge capability"""
    # Clean both datasets
    api_df = clean_movie_data(api_df) if not api_df.empty else pd.DataFrame()
    kaggle_df = clean_movie_data(kaggle_df) if not kaggle_df.empty else pd.DataFrame()
    
    # Define expected columns in the DuckDB table
    expected_columns = [
        'id', 'title', 'vote_average', 'vote_count', 'status', 'release_date', 
        'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage', 
        'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 
        'poster_path', 'tagline', 'genres', 'production_companies', 
        'production_countries', 'spoken_languages', 'keywords', 'last_updated'
    ]
    base_columns = expected_columns[:-1]  # without last_updated
    
    # Prepare data: add last_updated timestamp
    current_time = datetime.now()
    
    # Add missing columns and reorder for API data
    if not api_df.empty:
        for col in base_columns:
            if col not in api_df.columns:
                api_df[col] = None
        api_df['last_updated'] = current_time
        api_df = api_df[expected_columns]
    
    # Add missing columns and reorder for Kaggle data
    if not kaggle_df.empty:
        for col in base_columns:
            if col not in kaggle_df.columns:
                kaggle_df[col] = None
        kaggle_df['last_updated'] = current_time
        kaggle_df = kaggle_df[expected_columns]
    
    # DuckDB Storage
    duckdb_path = os.path.join(DATA_DIR, 'movies.duckdb')
    duckdb_conn = None
    try:
        duckdb_conn = duckdb.connect(duckdb_path)
        
        # Create table if not exists
        duckdb_conn.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title VARCHAR,
            vote_average FLOAT,
            vote_count INTEGER,
            status VARCHAR,
            release_date VARCHAR,
            revenue BIGINT,
            runtime INTEGER,
            adult BOOLEAN,
            backdrop_path VARCHAR,
            budget BIGINT,
            homepage VARCHAR,
            imdb_id VARCHAR,
            original_language VARCHAR,
            original_title VARCHAR,
            overview VARCHAR,
            popularity FLOAT,
            poster_path VARCHAR,
            tagline VARCHAR,
            genres VARCHAR,
            production_companies VARCHAR,
            production_countries VARCHAR,
            spoken_languages VARCHAR,
            keywords VARCHAR,
            last_updated TIMESTAMP
        )
        """)
        
        # Merge API data
        if not api_df.empty:
            duckdb_conn.register('api_df', api_df)
            duckdb_conn.execute("""
            INSERT OR REPLACE INTO movies
            SELECT * FROM api_df
            """)
            logger.info(f"Stored/updated {len(api_df)} movies from API")
        
        # Merge Kaggle data (only new/updated)
        if not kaggle_df.empty:
            duckdb_conn.register('kaggle_df', kaggle_df)
            # First count how many new movies we have
            new_movies_count = duckdb_conn.execute("""
                SELECT COUNT(*) 
                FROM kaggle_df 
                WHERE id NOT IN (SELECT id FROM movies)
            """).fetchone()[0]
            
            if new_movies_count > 0:
                duckdb_conn.execute("""
                INSERT OR IGNORE INTO movies
                SELECT * FROM kaggle_df
                WHERE id NOT IN (SELECT id FROM movies)
                """)
                logger.info(f"Added {new_movies_count} new movies from Kaggle")
            else:
                logger.info("No new movies from Kaggle to add")
            
        # Log total movies count
        total_movies = duckdb_conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        logger.info(f"Total movies in database: {total_movies}")
        logger.info(f"Data stored/updated in DuckDB: {duckdb_path}")
        return True
    except duckdb.Error as e:
        logger.error(f"DuckDB error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in store_data: {str(e)}")
        return False
    finally:
        if duckdb_conn:
            duckdb_conn.close()

def create_sample_file():
    """Create sample data file for fallback"""
    sample_data = [
        {
            "id": 1,
            "title": "Sample Movie",
            "overview": "This is a sample movie",
            "release_date": "2023-01-01",
            "vote_average": 7.5,
            "vote_count": 100,
            "popularity": 50.0,
            "genres": "Action, Adventure",
            "poster_path": "/sample1.jpg",
            "status": "Released",
            "revenue": 10000000,
            "runtime": 120,
            "adult": False,
            "backdrop_path": "/backdrop1.jpg",
            "budget": 5000000,
            "homepage": "http://example.com",
            "imdb_id": "tt1234567",
            "original_language": "en",
            "original_title": "Sample Movie",
            "tagline": "A sample tagline",
            "production_companies": "Sample Productions",
            "production_countries": "United States",
            "spoken_languages": "English",
            "keywords": "sample, movie"
        },
        {
            "id": 2,
            "title": "Another Sample",
            "overview": "Another sample movie",
            "release_date": "2023-02-15",
            "vote_average": 8.0,
            "vote_count": 150,
            "popularity": 75.0,
            "genres": "Drama, Romance",
            "poster_path": "/sample2.jpg",
            "status": "Released",
            "revenue": 20000000,
            "runtime": 130,
            "adult": False,
            "backdrop_path": "/backdrop2.jpg",
            "budget": 8000000,
            "homepage": "http://example2.com",
            "imdb_id": "tt7654321",
            "original_language": "en",
            "original_title": "Another Sample",
            "tagline": "Another tagline",
            "production_companies": "Another Productions",
            "production_countries": "United Kingdom",
            "spoken_languages": "English",
            "keywords": "another, sample"
        }
    ]
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    logger.info(f"Created sample data file: {sample_path}")

def periodic_data_refresh():
    """Refresh data from multiple sources"""
    logger.info("\nStarting periodic data refresh...")
    try:
        # Fetch from TMDB API
        api_df = fetch_tmdb_data(pages=50)
        
        # Load Kaggle dataset
        kaggle_df = load_kaggle_dataset()
        
        # Store combined data
        success = store_data(api_df, kaggle_df)
        if success:
            logger.info("Data refresh successful!")
        return success
    except Exception as e:
        logger.error(f"Refresh failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie Data Ingestion Script')
    parser.add_argument('--daemon', action='store_true', 
                        help='Run in daemon mode with scheduled refreshes')
    args = parser.parse_args()

    if os.name == 'nt':
        os.system('chcp 65001 > nul')
    
    sample_path = os.path.join(DATA_DIR, 'sample_movies.json')
    if not os.path.exists(sample_path):
        create_sample_file()
    
    try:
        movies_df = fetch_tmdb_data(pages=50) 
    except Exception as e:
        logger.error(f"Critical error during fetch: {str(e)}")
        movies_df = pd.DataFrame()
    
    if not movies_df.empty:
        # Load Kaggle dataset for initial storage
        kaggle_df = load_kaggle_dataset()
        success = store_data(movies_df, kaggle_df)
        if success:
            logger.info("Initial data stored successfully!")
        else:
            logger.error("Initial data storage failed!")
    
    if not args.daemon:
        logger.info("Initial data ingestion completed. Exiting.")
        exit(0)
    
    schedule.every().day.at("03:00").do(periodic_data_refresh)
    
    logger.info("Running in daemon mode. Scheduled refreshes set for 3 AM daily.")
    logger.info("Press Ctrl+C to exit...")
    
    while True:
        schedule.run_pending()
        time.sleep(60)