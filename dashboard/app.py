import streamlit as st
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# Configuration - Use relative paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/200x300?text=No+Poster"

# Page setup
st.set_page_config(
    page_title="Movie Recommendation Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .movie-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #0E1117;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .header {
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 10px;
    }
    .recommendation {
        padding: 10px;
        border-radius: 8px;
        background-color: #1a1d29;
        margin: 8px 0;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    .stSelectbox>div>div>div>input {
        background-color: #1a1d29;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Optimized data loading functions
@st.cache_resource
def get_db_connection(db_name):
    """Create DuckDB connection with optimized settings"""
    db_path = os.path.join(DATA_DIR if 'movies' in db_name else MODEL_DIR, db_name)
    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("PRAGMA threads=2;")
    return conn

# Load metadata first (lightweight)
@st.cache_data
def load_metadata():
    try:
        metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        return {}

metadata = load_metadata()

# Data validation with optimized types
def validate_movie_data(df):
    """Ensure critical columns have correct types and handle missing values"""
    if df.empty:
        return df
        
    # Optimized type conversions
    type_conversions = {
        'id': 'int32',
        'title': 'string',
        'genres': 'string',
        'vote_average': 'float32',
        'vote_count': 'int32',
        'popularity': 'float32',
        'poster_path': 'string',
        'overview': 'string'
    }
    
    for col, dtype in type_conversions.items():
        if col in df.columns:
            if dtype == 'string':
                df[col] = df[col].astype('string').fillna('')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
    
    # Handle release date separately
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year.fillna(0).astype('int16')
    else:
        df['year'] = 0
        
    return df

# Load data in chunks with optimized caching
@st.cache_data(ttl=3600, show_spinner="Loading movie data...")
def load_movie_data():
    """Load and validate movie data in chunks"""
    try:
        conn = get_db_connection('movies.duckdb')
        # Get count first to determine chunking strategy
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        chunk_size = 50000  # Process in chunks of 50K records
        
        dfs = []
        for offset in range(0, count, chunk_size):
            query = f"""
                SELECT id, title, genres, vote_average, vote_count, 
                       popularity, release_date, poster_path, overview
                FROM movies
                LIMIT {chunk_size} OFFSET {offset}
            """
            chunk = conn.execute(query).fetchdf()
            dfs.append(validate_movie_data(chunk))
        
        conn.close()
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading movie data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_recommendations():
    """Load precomputed recommendations with optimized types"""
    try:
        conn = get_db_connection('models.duckdb')
        # Load only necessary columns
        recommendations = conn.execute("""
            SELECT movie_id, rec_id, score
            FROM recommendations
        """).fetchdf()
        
        movie_indices = conn.execute("""
            SELECT id, idx
            FROM movie_indices
        """).fetchdf()
        conn.close()
        
        # Optimize types
        if not recommendations.empty:
            recommendations = recommendations.astype({
                'movie_id': 'int32',
                'rec_id': 'int32',
                'score': 'float32'
            })
            
        if not movie_indices.empty:
            movie_indices = movie_indices.astype({
                'id': 'int32',
                'idx': 'int32'
            })
            
        return recommendations, movie_indices
    except Exception as e:
        st.error(f"Error loading recommendations: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Load data
movies_df = load_movie_data()
rec_df, idx_df = load_recommendations()

# Precompute genres for filter
@st.cache_data(ttl=3600)
def get_all_genres(_df):
    """Extract all unique genres efficiently"""
    if _df.empty or 'genres' not in _df.columns:
        return []
    
    # Use set operations for efficiency
    return sorted(set(g for sublist in _df['genres'].str.split(', ') 
                for g in sublist if g and g != ''))

all_genres = get_all_genres(movies_df)

# Sidebar filters
st.sidebar.title("🎬 Movie Filters")
st.sidebar.markdown("Filter the movie database")

# Genre filter
selected_genres = st.sidebar.multiselect(
    "Select genres:", 
    all_genres,
    default=[]
)

# Year filter
if not movies_df.empty and 'year' in movies_df.columns:
    min_year = max(1900, int(movies_df['year'].min()))
    max_year = min(datetime.now().year, int(movies_df['year'].max()))
    year_range = st.sidebar.slider(
        "Release Year Range:", 
        min_year, max_year, 
        (min_year, max_year))
else:
    year_range = (1900, datetime.now().year)

# Rating filter
rating_range = st.sidebar.slider(
    "Minimum Rating:", 
    0.0, 10.0, 5.0, 0.5
)

# Search box
search_query = st.sidebar.text_input("Search Movies by Title:")

# Apply filters efficiently with memory optimization
def apply_filters(df, genres, year_range, rating, query):
    """Apply filters with memory optimization"""
    if df.empty:
        return df
        
    # Apply filters sequentially to reduce memory footprint
    filtered = df
    
    # Genre filter
    if genres:
        filtered = filtered[filtered['genres'].apply(
            lambda x: any(genre in x for genre in genres) if x else False
        )]
    
    # Year filter
    if 'year' in filtered.columns:
        filtered = filtered[
            (filtered['year'] >= year_range[0]) & 
            (filtered['year'] <= year_range[1])]
    
    # Rating filter
    if 'vote_average' in filtered.columns:
        filtered = filtered[filtered['vote_average'] >= rating]
    
    # Search filter
    if query:
        filtered = filtered[
            filtered['title'].str.contains(query, case=False, na=False)
        ]
    
    return filtered.reset_index(drop=True)

filtered_movies = apply_filters(
    movies_df, selected_genres, year_range, rating_range, search_query
)

# Main content
st.title("🍿 Movie Recommendation Dashboard")
st.markdown("Discover movies and get personalized recommendations")

# Stats header with optimized metrics
@st.cache_data(ttl=300)
def compute_metrics(_df, _metadata):
    """Compute dashboard metrics efficiently"""
    return {
        'count': len(_df),
        'avg_rating': _df['vote_average'].mean() if not _df.empty else 0.0,
        'latest_year': int(_df['year'].max()) if not _df.empty and 'year' in _df.columns else "N/A",
        'rec_count': _metadata.get('movie_count', 0)
    }

metrics = compute_metrics(movies_df, metadata)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies", metrics['count'])
col2.metric("Average Rating", f"{metrics['avg_rating']:.1f}/10")
col3.metric("Latest Movie", metrics['latest_year'])
col4.metric("Recommendations", metrics['rec_count'])

# Movie selection and recommendation section
st.header("🎯 Get Recommendations", divider="red")
st.markdown("Select a movie to get personalized recommendations")

if not movies_df.empty:
    # Optimized movie selector with caching and virtual scrolling
    @st.cache_data(ttl=3600)
    def get_movie_options(_df):
        """Generate movie selection options efficiently"""
        # Create options with minimal data
        options = _df[['title', 'year']].copy()
        # Vectorized string operation for performance
        mask = options['year'] > 1900
        options.loc[mask, 'label'] = options['title'] + ' (' + options['year'].astype(str) + ')'
        options.loc[~mask, 'label'] = options['title']
        return options['label'].tolist(), options.index.tolist()
    
    # Get options and their indices
    movie_options, movie_indices = get_movie_options(movies_df)
    
    # Use searchable selectbox with virtual scrolling
    selected_movie_title = st.selectbox(
        "Choose a movie:", 
        movie_options,
        index=0,
        key="movie_selector"
    )
    
    # Get selected movie index directly from cached mapping
    selected_index = movie_indices[movie_options.index(selected_movie_title)]
    selected_movie = movies_df.iloc[selected_index]
    
    # Display selected movie details with optimized rendering
    with st.container():
        col1, col2 = st.columns([1, 3])
        poster_path = selected_movie['poster_path']
        poster_url = (POSTER_BASE_URL + poster_path 
                    if poster_path and isinstance(poster_path, str) and poster_path.strip() 
                    else PLACEHOLDER_POSTER)
        
        with col1:
            st.image(poster_url, width=200, caption=selected_movie['title'])
        
        with col2:
            st.subheader(selected_movie['title'])
            year = selected_movie['year'] if selected_movie['year'] > 1900 else "N/A"
            st.caption(f"Released: {year}")
            
            rating_col, vote_col = st.columns(2)
            rating = selected_movie.get('vote_average', 0) or 0
            votes = selected_movie.get('vote_count', 0) or 0
            
            rating_col.metric("Rating", f"{rating:.1f}/10")
            vote_col.metric("Votes", f"{votes:,}")
            
            if genres := selected_movie.get('genres', ''):
                if isinstance(genres, str) and genres.strip():
                    st.write("**Genres:** " + ", ".join(genres.split(', ')[:3]))
            
            if overview := selected_movie.get('overview', ''):
                if isinstance(overview, str) and overview.strip():
                    with st.expander("Overview"):
                        st.write(overview)
    
    # Get recommendations with optimized data handling
    if not rec_df.empty and not idx_df.empty:
        movie_id = selected_movie['id']
        
        # Get movie index efficiently
        movie_idx = idx_df.loc[idx_df['id'] == movie_id, 'idx'].values
        if movie_idx.size > 0:
            # Get top 10 recommendations
            movie_recs = rec_df[rec_df['movie_id'] == movie_id] \
                .nlargest(10, 'score')
            
            if not movie_recs.empty:
                # Get movie details for recommendations
                rec_ids = movie_recs['rec_id'].unique()
                rec_movies = movies_df[movies_df['id'].isin(rec_ids)]
                
                # Merge with scores
                rec_movies = rec_movies.merge(
                    movie_recs[['rec_id', 'score']], 
                    left_on='id', 
                    right_on='rec_id'
                ).sort_values('score', ascending=False)
                
                st.subheader("Recommended Movies", divider="red")
                
                # Display recommendations in 2 columns with lazy rendering
                cols = st.columns(2)
                for i, (_, row) in enumerate(rec_movies.iterrows()):
                    with cols[i % 2]:
                        with st.container():
                            st.markdown("<div class='recommendation'>", unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns([1, 2])
                            poster_path = row['poster_path']
                            rec_poster = (POSTER_BASE_URL + poster_path 
                                        if poster_path and isinstance(poster_path, str) and poster_path.strip()
                                        else PLACEHOLDER_POSTER)
                            
                            with col_a:
                                st.image(rec_poster, width=100)
                            
                            with col_b:
                                title = row.get('title', 'Unknown Title') or 'Unknown Title'
                                st.markdown(f"**{title}**")
                                
                                year = row['year'] if row['year'] > 1900 else "N/A"
                                st.caption(f"Released: {year}")
                                
                                similarity = row.get('score', 0) or 0
                                st.progress(
                                    min(similarity / 100, 1.0), 
                                    text=f"Similarity: {similarity:.1f}%"
                                )
                                
                                if genres := row.get('genres', ''):
                                    if isinstance(genres, str) and genres.strip():
                                        st.caption(", ".join(genres.split(', ')[:2]))
                            
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this movie")
        else:
            st.warning("Movie index not found in recommendation database")
    else:
        st.warning("Recommendation data not loaded")

# Data exploration section with memory optimization
st.header("🔍 Explore Movies", divider="red")
st.markdown("Browse and filter our movie collection")

if not filtered_movies.empty:
    # Pagination with chunked data loading
    items_per_page = 20
    total_pages = max(1, (len(filtered_movies) - 1) // items_per_page + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_movies = filtered_movies.iloc[start_idx:end_idx]

    st.subheader(f"Showing {len(page_movies)} of {len(filtered_movies)} Movies")
    
    # Display movies in grid with optimized rendering
    cols = st.columns(4)
    for i, (_, row) in enumerate(page_movies.iterrows()):
        with cols[i % 4]:
            with st.container():
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                
                poster_path = row['poster_path']
                poster_url = (POSTER_BASE_URL + poster_path 
                            if poster_path and isinstance(poster_path, str) and poster_path.strip()
                            else PLACEHOLDER_POSTER)
                st.image(poster_url, use_container_width=True)
                
                title = row.get('title', 'Unknown Title') or 'Unknown Title'
                year = row['year'] if row['year'] > 1900 else "N/A"
                st.markdown(f"**{title}** ({year})")
                
                rating = row.get('vote_average', 0) or 0
                votes = row.get('vote_count', 0) or 0
                st.markdown(f"⭐ **{rating:.1f}**/10 ({votes:,} votes)")
                
                if genres := row.get('genres', ''):
                    if isinstance(genres, str) and genres.strip():
                        st.caption(", ".join(genres.split(', ')[:2]))
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Pagination controls
    if total_pages > 1:
        st.write(f"Page {page} of {total_pages}")
        prev_col, _, next_col = st.columns([1, 10, 1])
        with prev_col:
            if st.button("← Previous", disabled=(page == 1)):
                page = max(1, page - 1)
        with next_col:
            if st.button("Next →", disabled=(page == total_pages)):
                page = min(total_pages, page + 1)
    
    # Data table option with lazy loading
    with st.expander("View as Data Table"):
        display_cols = ['title', 'year', 'vote_average', 'genres']
        display_df = filtered_movies[display_cols].reset_index(drop=True)
        
        # Optimized type conversions
        display_df['year'] = display_df['year'].apply(
            lambda x: str(int(x)) if x > 1900 else 'N/A'
        )
        display_df['vote_average'] = display_df['vote_average'].fillna(0)
        
        st.dataframe(
            display_df,
            height=400,
            column_config={
                "title": "Movie Title",
                "year": "Year",
                "vote_average": st.column_config.NumberColumn(
                    "Rating",
                    format="%.1f ⭐"
                ),
                "genres": "Genres"
            }
        )
    
    # Visualizations with optimized data handling
    st.header("📊 Insights & Analytics", divider="red")
    
    @st.cache_data(ttl=300)
    def create_visualizations(_df):
        """Generate visualizations with minimal data"""
        figs = {}
        
        # Rating Distribution
        if 'vote_average' in _df.columns:
            fig1, ax1 = plt.subplots()
            sns.histplot(
                _df['vote_average'].dropna(), 
                bins=20, 
                kde=True, 
                ax=ax1
            )
            ax1.set_xlabel('Rating')
            ax1.set_ylabel('Number of Movies')
            figs['rating'] = fig1
        
        # Genre Popularity
        if 'genres' in _df.columns:
            genre_counts = {}
            for genres in _df['genres'].dropna():
                for genre in genres.split(', '):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if genre_counts:
                top_genres = pd.Series(genre_counts).nlargest(10)
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x=top_genres.values, 
                    y=top_genres.index, 
                    palette="viridis", 
                    ax=ax2
                )
                ax2.set_xlabel('Number of Movies')
                ax2.set_ylabel('Genre')
                figs['genre'] = fig2
        
        # Release Year Trends
        if 'year' in _df.columns:
            valid_years = _df[_df['year'] > 1900]['year']
            if not valid_years.empty:
                year_counts = valid_years.value_counts().sort_index()
                fig3, ax3 = plt.subplots()
                sns.lineplot(
                    x=year_counts.index, 
                    y=year_counts.values, 
                    ax=ax3
                )
                ax3.set_xlabel('Release Year')
                ax3.set_ylabel('Number of Movies')
                figs['year'] = fig3
        
        return figs
    
    viz_figs = create_visualizations(filtered_movies)
    
    tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Genre Popularity", "Release Trends"])
    
    with tab1:
        st.subheader("Rating Distribution")
        if 'rating' in viz_figs:
            st.pyplot(viz_figs['rating'])
        else:
            st.warning("No rating data available")
    
    with tab2:
        st.subheader("Genre Popularity")
        if 'genre' in viz_figs:
            st.pyplot(viz_figs['genre'])
        else:
            st.warning("No genre data available")
    
    with tab3:
        st.subheader("Release Year Trends")
        if 'year' in viz_figs:
            st.pyplot(viz_figs['year'])
        else:
            st.warning("No valid year data available")
else:
    st.warning("No movies found with current filters")

# Footer
st.divider()
st.caption(f"Data updated: {metadata.get('trained_at', 'N/A')}")
st.caption(f"Movie database contains {len(movies_df)} titles")