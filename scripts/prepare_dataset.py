#!/usr/bin/env python3
"""
Prepare dataset: chunk documents and build graph.

Usage:
    python scripts/prepare_dataset.py [--use-simple-chunker]
"""

import sys
import argparse
from pathlib import Path
import pickle
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from src.preprocess.loader import HotpotQALoader
from src.preprocess.chunker import LLMChunker, simple_chunker
from src.embeddings.encoder import EmbeddingEncoder
from src.graph.builder import GraphBuilder
from src.generation.llm import LLMClient
from src.utils.logger import setup_logger
from src.utils.helpers import load_prompts

def main():
    """Prepare dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-simple-chunker', action='store_true',
                       help='Use simple chunker instead of LLM-based chunker')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
    prompts = load_prompts(str(prompts_path))
    
    # Setup logger
    logger = setup_logger(
        level=config['logging']['level'],
        log_file=config['logging']['file'],
        console=config['logging']['console']
    )
    
    logger.info("Starting dataset preparation...")
    prep_start = time.time()
    
    # Load data
    loader = HotpotQALoader(config)
    documents, questions = loader.load_processed_data()
    
    if not documents:
        logger.error("No data found! Run download_data.py first.")
        return
    
    # Check if already processed
    processed_dir = Path(config['dataset']['cache_dir']) / 'processed'
    chunks_file = processed_dir / 'chunks.pkl'
    
    if chunks_file.exists():
        logger.info("Loading existing chunks and embeddings...")
        with open(chunks_file, 'rb') as f:
            data = pickle.load(f)
            chunks = data['chunks']
            embeddings = data['embeddings']
            graph = data['graph']
        logger.info(f"Loaded {len(chunks)} chunks")
    else:
        # Chunk documents
        logger.info("Chunking documents...")
        chunk_start = time.time()
        
        # Check config for chunking method
        chunking_method = config.get('chunking', {}).get('method', 'simple')
        
        if args.use_simple_chunker or chunking_method == 'simple':
            logger.info("Using simple chunker (no LLM)")
            chunks = simple_chunker(documents, chunk_size=512)
        else:
            logger.info("Using LLM-based chunker")
            llm_client = LLMClient(config)
            chunker = LLMChunker(llm_client, prompts, config)
            chunks = chunker.chunk_documents(documents)
        
        chunk_time = time.time() - chunk_start
        logger.info(f"Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        # Encode chunks
        logger.info("Encoding chunks...")
        encode_start = time.time()
        encoder = EmbeddingEncoder(config)
        chunk_texts = [c['text'] for c in chunks]
        embeddings = encoder.encode(chunk_texts)
        encode_time = time.time() - encode_start
        logger.info(f"Encoded to shape {embeddings.shape} in {encode_time:.2f}s")
        
        # Build graph
        logger.info("Building chunk graph...")
        graph_start = time.time()
        graph_builder = GraphBuilder(config)
        graph = graph_builder.build_chunk_graph(chunks, embeddings)
        graph_time = time.time() - graph_start
        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges in {graph_time:.2f}s"
        )
        
        # Save
        logger.info("Saving chunks, embeddings, and graph...")
        with open(chunks_file, 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'embeddings': embeddings,
                'graph': graph
            }, f)
    
    total_time = time.time() - prep_start
    logger.info(f"Dataset preparation complete in {total_time:.2f}s!")

if __name__ == "__main__":
    main()
