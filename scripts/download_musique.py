
import os
import json
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/raw/musique"
OUTPUT_FILE = "musique_ans_v1.0_dev.json"

def download_musique():
    """
    Downloads the MuSiQue dataset (validation split) and saves it as a JSON file.
    """
    logger.info("Downloading dgslibisey/MuSiQue (default, validation split)...")
    try:
        ds = load_dataset("dgslibisey/MuSiQue", "default", split="validation")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    # Convert to list of dicts
    data = []
    logger.info(f"Processing {len(ds)} rows...")
    for row in ds:
        data.append(row)
        
    logger.info(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    logger.info("Download complete.")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    download_musique()
