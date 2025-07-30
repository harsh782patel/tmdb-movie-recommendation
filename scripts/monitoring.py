import pandas as pd
import duckdb
import os
import smtplib
import logging
import re
from datetime import datetime

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODEL_DIR, 'monitoring.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_monitoring')

def load_movie_data():
    """Load movie data from DuckDB database"""
    try:
        db_path = os.path.join(DATA_DIR, 'movies.duckdb')
        conn = duckdb.connect(db_path)
        df = conn.execute("SELECT * FROM movies").fetchdf()
        conn.close()
        
        if df.empty:
            logger.warning("No movies found in database")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def perform_data_checks(df):
    """Enhanced data quality checks"""
    checks_failed = []
    
    # 1. Check for required columns
    required_columns = ['id', 'title', 'release_date', 'vote_average', 'vote_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        checks_failed.append(f"Missing columns: {', '.join(missing_columns)}")
    
    # 2. Check for duplicate IDs
    duplicate_ids = df['id'].duplicated().sum()
    if duplicate_ids > 0:
        checks_failed.append(f"Duplicate IDs found: {duplicate_ids}")
    
    # 3. Check for null values
    critical_columns = ['id', 'title']
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            checks_failed.append(f"Null values in {col}: {null_count}")
    
    # 4. Validate release_date format
    if 'release_date' in df.columns:
        valid_dates = df['release_date'].apply(
            lambda x: bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(x))) if pd.notnull(x) else True
        )
        invalid_count = (~valid_dates).sum()
        if invalid_count > 0:
            checks_failed.append(f"Invalid date formats: {invalid_count}")
    
    # 5. Validate numerical ranges
    numerical_checks = [
        ('vote_average', 0, 10),
        ('vote_count', 0, 1000000),
        ('popularity', 0, 1000000),
        ('revenue', 0, 1e12),  # 1 trillion
        ('runtime', 0, 1000),   # 1000 minutes
        ('budget', 0, 1e12)     # 1 trillion
    ]
    
    for col, min_val, max_val in numerical_checks:
        if col in df.columns:
            valid_values = df[col].dropna()
            if not valid_values.empty:
                out_of_range = ((valid_values < min_val) | (valid_values > max_val)).sum()
                if out_of_range > 0:
                    checks_failed.append(f"Values out of range in {col}: {out_of_range}")
    
    # 6. Check for empty strings in critical fields
    text_checks = ['title', 'overview']
    for col in text_checks:
        if col in df.columns:
            empty_count = (df[col].fillna('') == '').sum()
            if empty_count > 0:
                checks_failed.append(f"Empty {col} fields: {empty_count}")
    
    # 7. Check for adult content flag consistency
    if 'adult' in df.columns:
        # Check if there are any non-boolean values
        non_bool = df['adult'].apply(lambda x: not isinstance(x, bool)).sum()
        if non_bool > 0:
            checks_failed.append(f"Non-boolean values in 'adult' column: {non_bool}")
    
    return checks_failed

def data_quality_check():
    """Main data quality check function"""
    logger.info("Starting data quality checks...")
    df = load_movie_data()
    if df.empty:
        logger.error("Skipping checks - no data available")
        return False

    checks_failed = perform_data_checks(df)
    
    if checks_failed:
        alert_msg = "DATA QUALITY ALERT:\n" + "\n".join(checks_failed)
        logger.warning(alert_msg)
        
        if os.getenv("ALERT_EMAIL_USER") and os.getenv("ALERT_EMAIL_PASS"):
            send_alert(alert_msg)
        else:
            logger.warning("Skipping email alert - credentials not available")
        return False
    
    logger.info("All data quality checks passed")
    return True

def send_alert(message):
    """Send email alert using environment variables"""
    try:
        # Get credentials from environment
        email_user = os.getenv("ALERT_EMAIL_USER")
        email_pass = os.getenv("ALERT_EMAIL_PASS")
        recipient = os.getenv("ALERT_RECIPIENT", email_user)
        
        if not email_user or not email_pass:
            logger.error("Missing email credentials in environment variables")
            return
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        try:
            server.login(email_user, email_pass)
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Email authentication failed: {str(e)}")
            return
        subject = "Data Quality Alert"
        body = f"Subject: {subject}\n\n{message}"
        
        server.sendmail(email_user, recipient, body)
        server.quit()
        logger.info(f"Alert email sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send alert: {str(e)}")

def load_env_file():
    """Attempt to load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        logger.info("Loading environment variables from .env file")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip()

def check_env_vars():
    """Check and log status of required environment variables"""
    logger.info("Checking environment variables...")
    env_vars = {
        'ALERT_EMAIL_USER': os.getenv("ALERT_EMAIL_USER"),
        'ALERT_EMAIL_PASS': os.getenv("ALERT_EMAIL_PASS"),
        'ALERT_RECIPIENT': os.getenv("ALERT_RECIPIENT")
    }
    
    for var, value in env_vars.items():
        status = "Set" if value else "Not set"
        logger.info(f"{var}: {status}")

if __name__ == "__main__":
    logger.info("Starting data monitoring process")
    
    # Try to load from .env file if exists
    load_env_file()
    
    # Check environment variables status
    check_env_vars()
    
    # Run data quality checks
    data_quality_check()
    logger.info("Monitoring completed")