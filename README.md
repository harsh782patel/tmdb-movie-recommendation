# TMDB Movie Recommender 

[![Python](https://img.shields.io/badge/Python%20-%20646464?style=flat&logo=Python&logoColor=FFFFFF&labelColor=4584b6&color=ffde57)](#)
[![Docker Build](https://img.shields.io/badge/Docker%20-%20%231D63ED?style=flat&logo=DOCKER&labelColor=%23E5F2FC&color=%231D63ED)](#)
[![Prefect](https://img.shields.io/badge/Prefect%20-%20%231D63ED?style=flat&logo=Prefect&labelColor=%23000000&color=%23FFFFFF)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit%20-%20646464?style=flat&logo=STREAMLIT&labelColor=FFFFFF&color=000000)](#)
[![DuckDB](https://img.shields.io/badge/DuckDB%20-%20%231D63ED?style=flat&logo=DUCKDB&labelColor=%23000000&color=%23FFFFFF)](#)
[![Numpy](https://img.shields.io/badge/Numpy%20-%20646464?style=flat&logo=Numpy&logoColor=rgb(77%2C%20171%2C%20207)&labelColor=FFFFFF&color=rgb(77%2C%20171%2C%20207))](#)
[![Pandas](https://img.shields.io/badge/Pandas%20-%20646464?style=flat&logo=Pandas&logoColor=150458&labelColor=FFFFFF&color=FFCA00)](#)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn%20-%20646464?style=flat&logo=Scikit-Learn&logoColor=%23F7931E&labelColor=FFFFFF&color=%2329ABE2)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A robust movie recommendation system powered by TMDB data with automated data pipelines, content-based filtering, and an interactive dashboard. This project demonstrates a complete machine learning workflow from data ingestion to model deployment.

## Key Features
- **Automated Data Pipeline** - Scheduled daily updates from TMDB API
- **Content-Based Recommendations** - TF-IDF vectorization for accurate suggestions
- **Self-Healing Architecture** - Automatic fallback to sample data during API failures
- **Data Quality Monitoring** - Validation checks with email alert system
- **Lightning Fast Storage** - DuckDB database for efficient query performance
- **Interactive Dashboard** - Streamlit interface with movie posters
- **Containerized Deployment** - Docker support for seamless deployment
- **Workflow Orchestration** - Prefect pipeline for end-to-end automation

## System Architecture
```mermaid
graph LR
    A[TMDB API] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D{Data Quality}
    D -->|Pass| E[Model Training]
    D -->|Fail| F[Send Alert]
    E --> G[Recommendation Models]
    G --> H[Streamlit Dashboard]
    H --> I[End Users]
```

## Getting Started

### Prerequisites
- Python 3.10+
- TMDB API key ([free account](https://www.themoviedb.org/settings/api))
- Docker (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/tmdb-movie-recommender.git
cd tmdb-movie-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Create environment file:
   ```bash
   cp .env.sample .env
   ```
2. Edit `.env` with your credentials:
   ```ini
   # Required configuration
   TMDB_API_KEY="your_api_key_here"
   
   # Optional email alerts
   ALERT_EMAIL_USER="your@email.com"
   ALERT_EMAIL_PASS="app_password"
   ```

## Usage Guide

### Running the Full Pipeline
```bash
python pipelines/movie_pipeline.py
```

### Individual Components
| Component | Command | Description |
|-----------|---------|-------------|
| Data Ingestion | `python scripts/data_ingestion.py` | Fetches movie data from TMDB API |
| Model Training | `python scripts/model_training.py` | Trains recommendation models |
| Data Monitoring | `python scripts/monitoring.py` | Performs data quality checks |
| Dashboard | `streamlit run dashboard/app.py` | Launches recommendation UI |

### Continuous Operation Mode
```bash
python scripts/data_ingestion.py --daemon
```

## Docker Deployment
```bash
# Build Docker image
docker build -f docker/Dockerfile -t movie-recommender .

# Run container with persistent storage
docker run -d \
  -p 8501:8501 \
  -v ./data:/app/data \
  -v ./models:/app/models \
  --env-file .env \
  --name movie-rec \
  movie-recommender
```
Access dashboard at: http://localhost:8501

## Dashboard Features
- **Interactive Filters** - Filter by release year and minimum rating
- **Movie Recommendations** - Get personalized suggestions based on content similarity
- **Top Rated Movies** - Discover highest rated films with visual ratings
- **Database Explorer** - Browse all movies in a sortable table
- **System Monitoring** - View dataset statistics and model version

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Movie+Recommender+Interface)
*Interactive dashboard showing movie recommendations and filters*

## Configuration Options
| Environment Variable | Required | Default | Purpose |
|----------------------|----------|---------|---------|
| `TMDB_API_KEY`       | Yes      | -       | TMDB API access key |
| `HTTP_PROXY`         | No       | None    | HTTP proxy configuration |
| `HTTPS_PROXY`        | No       | None    | HTTPS proxy configuration |
| `ALERT_EMAIL_USER`   | No       | None    | Email for data quality alerts |
| `ALERT_EMAIL_PASS`   | No       | None    | Email app password |
| `ALERT_RECIPIENT`    | No       | Same as user | Alert notification recipient |

## Troubleshooting Guide
| Issue | Solution |
|-------|----------|
| API requests failing | Verify TMDB_API_KEY in .env file |
| Missing movie posters | Check internet connection and TMDB image service |
| Database version errors | Delete `data/movies.duckdb` and re-run ingestion |
| Email alerts not working | Verify app password and enable less secure apps |
| Docker build failing | Ensure Docker has at least 2GB memory allocation |

## Contribution Guidelines
1. Report issues in GitHub tracker
2. Fork repository and create feature branches
3. Submit pull requests with detailed descriptions
4. Follow PEP 8 coding standards
5. Update documentation for new features

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Movie data provided by [The Movie Database](https://www.themoviedb.org/)
- Powered by [Streamlit](https://streamlit.io), [Prefect](https://prefect.io), and [DuckDB](https://duckdb.org)
- Inspired by similar recommender systems from Netflix and Hulu
