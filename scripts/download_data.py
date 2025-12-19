#!/usr/bin/env python3
"""
Download HotpotQA dataset and prepare HotpotQA-S.

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.preprocess.loader import HotpotQALoader
from src.utils.logger import setup_logger

def main():
    """Download and prepare data."""
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(
        level=config['logging']['level'],
        log_file=config['logging']['file'],
        console=config['logging']['console']
    )
    
    logger.info("Starting data download and preparation...")
    
    # Initialize loader
    loader = HotpotQALoader(config)
    
    # Check if data already exists
    existing_data = loader.load_processed_data()
    
    if existing_data:
        logger.info("Processed data already exists!")
        documents, questions = existing_data
        logger.info(f"Found {len(documents)} documents and {len(questions)} questions")
        return
    
    # Download dataset
    logger.info("Downloading HotpotQA dataset...")
    loader.download_dataset()
    
    # Create HotpotQA-S
    logger.info("Creating HotpotQA-S...")
    documents, questions = loader.create_hotpotqa_s()
    
    # Save processed data
    logger.info("Saving processed data...")
    loader.save_processed_data(documents, questions)
    
    logger.info("Data preparation complete!")
    logger.info(f"Documents: {len(documents)}")
    logger.info(f"Questions: {len(questions)}")

if __name__ == "__main__":
    main()
