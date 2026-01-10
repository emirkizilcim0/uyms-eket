from EKET.ingest import main as ingest_data
from EKET.utils import get_config, Path

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Log to stderr to avoid interfering with JSON stdout
)

logger = logging.getLogger(__name__)

def data_ingestion():
    logger.info("Documents in data folder:")
    for f in Path(get_config()['DATA_DIR']).glob('*'):
        logger.info(f"- {f.name}")
    
    ingest_data()

    