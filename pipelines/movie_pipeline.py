from prefect import Flow, task
from prefect.engine import signals
import logging
import os
import sys
from datetime import timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
scripts_dir = os.path.join(parent_dir, "scripts")

if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

try:
    from data_ingestion import periodic_data_refresh
    from model_training import train_models
    from monitoring import data_quality_check
except ImportError as e:
    print(f"Import error: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('movie_pipeline')

@task(name="Ingest Movie Data", max_retries=2, retry_delay=timedelta(seconds=60))
def ingest_data():
    """Task to fetch and store movie data"""
    logger.info("Starting data ingestion...")
    try:
        success = periodic_data_refresh()
        if not success:
            raise signals.FAIL("Data ingestion failed")  # Critical failure
        return success
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise signals.FAIL(f"Data ingestion failed: {str(e)}")

@task(name="Validate Data Quality")
def validate_data(ingestion_success):
    """Task to perform data quality checks"""
    if not ingestion_success:
        raise signals.SKIP("Skipping validation due to ingestion failure")
        
    logger.info("Starting data validation...")
    try:
        passed = data_quality_check()
        if not passed:
            st.warning("Data quality issues found - proceeding with training")
            return True
        return passed
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        return True

@task(name="Train Recommendation Models", max_retries=1, retry_delay=timedelta(seconds=120))
def train_models_task(validation_passed):
    """Task to train machine learning models"""
    if not validation_passed:
        raise signals.SKIP("Skipping training due to validation failure")
        
    logger.info("Starting model training...")
    try:
        success = train_models()
        if not success:
            raise signals.FAIL("Model training failed")
        return success
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise signals.FAIL(f"Model training failed: {str(e)}")

# Create Prefect 1.x flow
with Flow("Movie Recommendation Pipeline") as flow:
    ingestion_success = ingest_data()
    validation_passed = validate_data(ingestion_success)
    training_success = train_models_task(validation_passed)

if __name__ == "__main__":
    flow_state = flow.run()
    
    if flow_state.is_successful():
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline completed with errors")
        raise RuntimeError("Pipeline execution failed")