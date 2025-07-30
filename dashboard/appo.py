import os
import sys
import logging

# Apply critical fix BEFORE any imports
if "numpy" in sys.modules:
    del sys.modules["numpy"]  # Ensure clean numpy import

# Add compatibility shim for NumPy 1.x
try:
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core.multiarray'] = _core.multiarray
except ImportError:
    # If numpy isn't installed yet, create placeholder
    class DummyModule:
        def __init__(self, name):
            self.__name__ = name
        def __getattr__(self, name):
            return None
    sys.modules['numpy._core'] = DummyModule('numpy._core')
    sys.modules['numpy._core.multiarray'] = DummyModule('numpy._core.multiarray')

# Now import other modules
import streamlit as st
import pandas as pd
import duckdb
import joblib
import json
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
MODEL_DIR = os.path.join(project_root, "models")
DATA_DIR = os.path.join(project_root, "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if project_root not in sys.path:
    sys.path.append(project_root)

# --- Page Setup ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def safe_joblib_load(path):
    """Safely load joblib files with NumPy compatibility"""
    try:
        return joblib.load(path)
    except Exception as e:
        if "numpy._core" in str(e):
            # Try to repair the module structure
            try:
                import numpy.core as _core
                sys.modules['numpy._core'] = _core
                sys.modules['numpy._core.multiarray'] = _core.multiarray
                return joblib.load(path)
            except Exception:
                pass
            
            # Fallback to direct unpickling
            try:
                from joblib import load as joblib_load
                with open(path, 'rb') as f:
                    return joblib_load(f)
            except Exception as e2:
                st.error(f"Critical model loading error: {str(e2)}")
                raise
        raise

@st.cache_data(show_spinner="Loading models...")
def load_models():
    """Load trained recommendation models"""
    try:
        required_files = [
            'tfidf_model.pkl', 
            'recommendations.pkl',
            'indices.pkl',
            'movies_df.pkl',
            'metadata.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(MODEL_DIR, file)):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"Missing model files: {', '.join(missing_files)}")
            st.info("Please run the training pipeline first")
            st.stop()
        
        # Load models with safe loader
        tfidf = safe_joblib_load(os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
        recommendations = joblib.load(os.path.join(MODEL_DIR, 'recommendations.pkl'))
        indices = joblib.load(os.path.join(MODEL_DIR, 'indices.pkl'))
        movies_df = joblib.load(os.path.join(MODEL_DIR, 'movies_df.pkl'))
        
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        return tfidf, recommendations, indices, movies_df, metadata
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

@st.cache_data(show_spinner="Loading movie data...")
def load_movie_data():
    """Load movie data from DuckDB"""
    try:
        db_path = os.path.join(DATA_DIR, 'movies.duckdb')
        if not os.path.exists(db_path):
            st.error("Movie database not found. Run data ingestion first.")
            return pd.DataFrame()
        
        try:
            conn = duckdb.connect(db_path)
            movies = conn.execute("SELECT * FROM movies").fetchdf()
        except duckdb.IOException as e:
            if "version number" in str(e):
                conn = duckdb.connect(':memory:')
                conn.execute(f"SET legacy_extension='true';")
                conn.execute(f"ATTACH '{db_path}' AS legacy_db (READ_ONLY);")
                movies = conn.execute("SELECT * FROM legacy_db.movies").fetchdf()
            else:
                raise
        finally:
            if 'conn' in locals():
                conn.close()
                
        return movies
    except Exception as e:
        st.error(f"Error loading movie data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=24*3600, show_spinner=False)
def load_poster(poster_path, width=200):
    """Load and cache movie poster"""
    if not poster_path or pd.isna(poster_path):
        return None
        
    try:
        url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).resize((width, int(width * 1.5)))
        return None
    except Exception:
        return None

# --- Main Application ---
def main():
    st.title("🎬 Movie Recommendation Dashboard")
    
    # Load models and data
    with st.spinner("Initializing application..."):
        try:
            tfidf, recommendations, indices, movies_df, metadata = load_models()
            movies = load_movie_data()
            
            if movies.empty:
                st.warning("No movie data available. Run data ingestion first.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            st.stop()
    
    # Updated recommendation function
    def get_recommendations(movie_id, top_n=10):
        """Get precomputed recommendations"""
        try:
            if movie_id not in recommendations:
                st.error(f"Movie ID {movie_id} not found in recommendations")
                return pd.DataFrame(), []
                
            recs = recommendations[movie_id]
            movie_ids = [r[0] for r in recs[:top_n]]
            scores = [r[1] for r in recs[:top_n]]
            
            # Get movie details in order
            mask = movies_df['id'].isin(movie_ids)
            recommended_movies = movies_df[mask]
            order_mapping = {id_: idx for idx, id_ in enumerate(movie_ids)}
            recommended_movies = recommended_movies.copy()
            recommended_movies['order'] = recommended_movies['id'].map(order_mapping)
            recommended_movies = recommended_movies.sort_values('order').drop(columns=['order'])
            
            return recommended_movies, scores
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame(), []
    
    # Extract year range
    min_year = metadata.get('min_year', 1900) or 1900
    max_year = metadata.get('max_year', datetime.now().year) or datetime.now().year
    
    # --- Sidebar ---
    st.sidebar.header("Filters")
    selected_year = st.sidebar.slider(
        "Release Year", 
        min_value=min_year, 
        max_value=max_year, 
        value=(max(2000, min_year), max_year)
    )
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 7.0, 0.5)
    
    # Apply filters
    filtered_movies = movies.copy()
    try:
        filtered_movies['year'] = pd.to_numeric(filtered_movies['release_date'].str[:4], errors='coerce')
        filtered_movies = filtered_movies[
            (filtered_movies['vote_average'] >= min_rating) &
            (filtered_movies['year'].between(selected_year[0], selected_year[1]))]
    except Exception as e:
        st.warning(f"Error applying filters: {str(e)} - showing all movies")
        filtered_movies = movies.copy()
    
    # --- Main Content ---
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("System Info")
        st.metric("Total Movies", len(movies))
        st.metric("Model Version", metadata.get('trained_at', 'Unknown').split("T")[0])
        st.metric("Newest Movie Year", max_year)
        
        st.subheader("Get Recommendations")
        
        if not filtered_movies.empty:
            movie_options = [
                (row['id'], f"{row['title']} ({row['release_date'][:4]})")
                for _, row in filtered_movies.iterrows()
            ]
            movie_dict = {id_: display for id_, display in movie_options}
            
            selected_movie = st.selectbox(
                "Select a Movie", 
                options=list(movie_dict.keys()),
                format_func=lambda x: movie_dict[x]
            )
            
            if st.button("Recommend Similar Movies"):
                with st.spinner("Finding recommendations..."):
                    recommendations_df, scores = get_recommendations(selected_movie, 5)
                    if not recommendations_df.empty:
                        st.success("Top Recommendations:")
                        for i, (_, row) in enumerate(recommendations_df.iterrows()):
                            with st.container():
                                poster_col, text_col = st.columns([1, 3])
                                with poster_col:
                                    poster = load_poster(row.get('poster_path'), 150)
                                    if poster:
                                        st.image(poster)
                                    else:
                                        st.image(Image.new('RGB', (150, 225), color=(40,40,40)), caption="No poster")
                                with text_col:
                                    st.subheader(f"{i+1}. {row['title']} ({row['release_date'][:4]})")
                                    st.write(f"⭐ **Rating**: {row['vote_average']}/10 | 📊 **Similarity**: {scores[i]:.2f}")
                                    st.caption(row['overview'][:200] + "..." if len(row['overview']) > 200 else row['overview'])
                            st.divider()
                    else:
                        st.warning("No recommendations found")
        else:
            st.warning("No movies available for selection")

    with col2:
        st.subheader(f"Top Rated Movies ({len(filtered_movies)} filtered)")
        if not filtered_movies.empty:
            top_movies = filtered_movies.sort_values('vote_average', ascending=False).head(10)
            
            # Improved grid handling
            cols = st.columns(5)
            for i, (_, row) in enumerate(top_movies.iterrows()):
                if i >= 10:
                    break
                with cols[i % 5]:
                    with st.container():
                        poster = load_poster(row.get('poster_path'))
                        if poster:
                            st.image(poster)
                        else:
                            st.image(Image.new('RGB', (200, 300), color=(40, 40, 40)), caption="No poster")
                        st.write(f"**{row['title']}**")
                        st.caption(f"{row['release_date'][:4]}")
                        st.write(f"⭐ {row['vote_average']} | 📊 {row['vote_count']} votes")
                        st.progress(float(row['vote_average']) / 10.0)

            st.subheader("Movie Database")
            st.dataframe(
                filtered_movies[['title', 'release_date', 'vote_average', 'vote_count']].sort_values('vote_average', ascending=False),
                height=300,
                column_config={
                    "title": "Movie Title",
                    "release_date": st.column_config.DateColumn(
                        "Release Date",
                        format="YYYY"
                    ),
                    "vote_average": st.column_config.ProgressColumn(
                        "Rating",
                        min_value=0,
                        max_value=10,
                        format="%.1f"
                    ),
                    "vote_count": "Votes"
                }
            )
        else:
            st.warning("No movies available to display")

    # Footer
    st.divider()
    st.caption(f"Last updated: {metadata.get('trained_at', 'Unknown').split('T')[0]} | Movies: {min_year}-{max_year}")

if __name__ == "__main__":
    main()