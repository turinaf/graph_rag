#!/usr/bin/env python3
"""
Run RAG benchmarks on HotpotQA-S.

Usage:
    python scripts/run_benchmark.py [--methods naive_rag node_rag lp_rag] [--quick-test]
"""

import sys
import argparse
from pathlib import Path
import pickle
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.loader import HotpotQALoader
from src.embeddings.encoder import EmbeddingEncoder
from src.generation.llm import LLMClient
from src.retrieval.naive_rag import NaiveRAG
from src.retrieval.hyde import HyDE
from src.retrieval.graphrag import GraphRAG
from src.retrieval.lightrag import LightRAG
from src.retrieval.node_rag import NodeRAG
from src.retrieval.lp_rag import LPRAG
# SOTA implementations
from src.retrieval.hyde_sota import HyDESOTA
from src.retrieval.graphrag_sota import GraphRAGSOTA
from src.retrieval.node_rag_sota import NodeRAGSOTA
from src.retrieval.lightrag_sota import LightRAGSOTA
from src.link_prediction.heuristics import LinkPredictionHeuristics
from src.link_prediction.placeholder import CH3L3Predictor
from src.evaluation.benchmark import RAGBenchmark
from src.graph.builder import GraphBuilder
from src.utils.logger import setup_logger
from src.utils.helpers import load_prompts
import re


def _parse_bracketed_questions(response: str) -> list:
    """Extract bracketed questions from LLM response."""
    candidates = re.findall(r"\[(.*?)\]", response, flags=re.DOTALL)
    questions = []
    for cand in candidates:
        text = cand.strip()
        if not text:
            continue
        lower = text.lower()
        if any(prefix in lower for prefix in ["chunk", "questions", "sentence", "start", "end"]):
            continue
        if not text.endswith('?'):
            text = text.rstrip('.') + '?'
        if len(text) > 5:
            questions.append(text)
    return questions


def _fallback_questions(chunk: dict, needed: int) -> list:
    """Deterministic fallback question templates when LLM parsing fails."""
    title = chunk.get('doc_title') or "this document"
    base = chunk.get('text', '')[:80]
    templates = [
        f"What key fact about {title} is described?",
        f"What does the text say about {base}?"
    ]
    return templates[:needed]


def build_chunk_query_graph_if_needed(
    config,
    chunks,
    chunk_graph,
    encoder,
    llm_client,
    prompts,
    processed_dir,
    graph_builder,
    logger
):
    synthetic_conf = config.get('synthetic_queries', {})
    if not synthetic_conf.get('enabled', False):
        logger.info("Synthetic queries disabled; LP-RAG will use chunk-only graph.")
        return chunk_graph, []

    synthetic_path = processed_dir / 'synthetic_queries.pkl'
    if synthetic_path.exists():
        with open(synthetic_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(
            f"Loaded {len(data.get('queries', []))} synthetic queries from cache at {synthetic_path}"
        )
        return data['chunk_query_graph'], data.get('queries', [])

    logger.info("Generating synthetic queries for LP-RAG...")
    queries_per_chunk = synthetic_conf.get('queries_per_chunk', 2)
    generated_queries = []

    for chunk in chunks:
        try:
            prompt_text = prompts['synthetic_query_prompt'].format(
                chunks=f"[Chunk]\n{chunk['text']}\n"
            )
            response = llm_client.generate(prompt_text, max_tokens=512)
            parsed = _parse_bracketed_questions(response)
        except Exception as e:
            logger.warning(f"Synthetic query generation failed for chunk {chunk['chunk_id']}: {e}")
            parsed = []

        if len(parsed) < queries_per_chunk:
            parsed.extend(_fallback_questions(chunk, queries_per_chunk - len(parsed)))

        parsed = parsed[:queries_per_chunk]

        for question in parsed:
            generated_queries.append({
                'query_id': len(generated_queries),
                'text': question,
                'chunk_id': chunk['chunk_id']
            })

    query_texts = [q['text'] for q in generated_queries]
    query_embeddings = encoder.encode(query_texts)

    chunk_query_graph = graph_builder.build_chunk_query_graph(
        chunk_graph=chunk_graph,
        synthetic_queries=generated_queries,
        query_embeddings=query_embeddings
    )

    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    with open(synthetic_path, 'wb') as f:
        pickle.dump({
            'queries': generated_queries,
            'query_embeddings': query_embeddings,
            'chunk_query_graph': chunk_query_graph
        }, f)

    logger.info(
        f"Generated {len(generated_queries)} synthetic queries and built chunk-query graph "
        f"with {chunk_query_graph.number_of_nodes()} nodes"
    )

    return chunk_query_graph, generated_queries

def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description='Run RAG benchmarks')
    parser.add_argument('--methods', nargs='+', 
                       default=['naive_rag', 'hyde', 'graphrag', 'lightrag', 'node_rag', 'lp_rag'],
                       help='Methods to benchmark')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with only 10 questions')
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
    
    logger.info("="*80)
    logger.info("RAG BENCHMARKING SYSTEM")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading data...")
    loader = HotpotQALoader(config)
    documents, questions = loader.load_processed_data()
    
    if not documents:
        logger.error("No data found! Run download_data.py first.")
        return
    
    # Load chunks and embeddings
    processed_dir = Path(config['dataset']['cache_dir']) / 'processed'
    chunks_file = processed_dir / 'chunks.pkl'
    
    if not chunks_file.exists():
        logger.error("No chunks found! Run prepare_dataset.py first.")
        return
    
    logger.info("Loading chunks, embeddings, and graph...")
    with open(chunks_file, 'rb') as f:
        data = pickle.load(f)
        chunks = data['chunks']
        embeddings = data['embeddings']
        graph = data['graph']
    
    logger.info(f"Loaded {len(chunks)} chunks, {len(questions)} questions")
    
    # Quick test mode
    if args.quick_test:
        logger.info("QUICK TEST MODE: Using only 10 questions")
        questions = questions[:10]
    
    # Initialize components
    logger.info("Initializing components...")
    encoder = EmbeddingEncoder(config)
    llm_client = LLMClient(config)

    # Build synthetic queries and chunk-query graph for LP-RAG (only if LP-RAG is selected)
    graph_builder = GraphBuilder(config)
    if 'lp_rag' in args.methods:
        chunk_query_graph, synthetic_queries = build_chunk_query_graph_if_needed(
            config,
            chunks,
            graph,
            encoder,
            llm_client,
            prompts,
            processed_dir,
            graph_builder,
            logger
        )
    else:
        logger.info("LP-RAG not selected, skipping synthetic query generation")
        chunk_query_graph = graph
        synthetic_queries = []
    
    # Initialize retrievers
    retrievers = {}
    
    def timed_index(retriever, name: str, graph_override=None):
        start = time.time()
        retriever.index(chunks, embeddings, graph_override or graph)
        duration = time.time() - start
        logger.info(f"Indexed {name} in {duration:.2f}s")
        return duration

    if 'naive_rag' in args.methods:
        logger.info("Initializing NaiveRAG...")
        naive = NaiveRAG(config, encoder, llm_client)
        idx_time = timed_index(naive, 'naive_rag')
        retrievers['naive_rag'] = {'retriever': naive, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'hyde' in args.methods:
        logger.info("Initializing HyDE...")
        hyde = HyDE(config, encoder, llm_client, prompts)
        idx_time = timed_index(hyde, 'hyde')
        retrievers['hyde'] = {'retriever': hyde, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'graphrag' in args.methods:
        logger.info("Initializing GraphRAG...")
        graphrag = GraphRAG(config, encoder, llm_client)
        idx_time = timed_index(graphrag, 'graphrag')
        retrievers['graphrag'] = {'retriever': graphrag, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'lightrag' in args.methods:
        logger.info("Initializing LightRAG...")
        lightrag = LightRAG(config, encoder, llm_client)
        idx_time = timed_index(lightrag, 'lightrag')
        retrievers['lightrag'] = {'retriever': lightrag, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'node_rag' in args.methods:
        logger.info("Initializing NodeRAG (SOTA baseline)...")
        noderag = NodeRAG(config, encoder, llm_client)
        idx_time = timed_index(noderag, 'node_rag')
        retrievers['node_rag'] = {'retriever': noderag, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'lp_rag' in args.methods:
        logger.info("Initializing LP-RAG...")
        # Use common neighbors as link predictor
        link_predictor = LinkPredictionHeuristics(
            method=config['link_prediction']['method'],
            normalize=config['link_prediction']['normalize_scores']
        )
        lprag = LPRAG(config, encoder, llm_client, link_predictor)
        idx_time = timed_index(lprag, 'lp_rag', graph_override=chunk_query_graph)
        retrievers['lp_rag'] = {'retriever': lprag, 'prompts': prompts, 'index_time_sec': idx_time}
    
    # SOTA Implementations
    if 'hyde_sota' in args.methods:
        logger.info("Initializing HyDE SOTA (multi-hypothesis)...")
        hyde_sota = HyDESOTA(config, encoder, llm_client, prompts)
        idx_time = timed_index(hyde_sota, 'hyde_sota')
        retrievers['hyde_sota'] = {'retriever': hyde_sota, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'graphrag_sota' in args.methods:
        logger.info("Initializing GraphRAG SOTA (with summarization)...")
        graphrag_sota = GraphRAGSOTA(config, encoder, llm_client, prompts)
        idx_time = timed_index(graphrag_sota, 'graphrag_sota')
        retrievers['graphrag_sota'] = {'retriever': graphrag_sota, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'node_rag_sota' in args.methods:
        logger.info("Initializing NodeRAG SOTA (heterogeneous graph)...")
        noderag_sota = NodeRAGSOTA(config, encoder, llm_client)
        idx_time = timed_index(noderag_sota, 'node_rag_sota')
        retrievers['node_rag_sota'] = {'retriever': noderag_sota, 'prompts': prompts, 'index_time_sec': idx_time}
    
    if 'lightrag_sota' in args.methods:
        logger.info("Initializing LightRAG SOTA (dual-graph entity extraction)...")
        lightrag_sota = LightRAGSOTA(config, encoder, llm_client)
        idx_time = timed_index(lightrag_sota, 'lightrag_sota')
        retrievers['lightrag_sota'] = {'retriever': lightrag_sota, 'prompts': prompts, 'index_time_sec': idx_time}
    
    # Run benchmarks
    logger.info("="*80)
    logger.info("RUNNING BENCHMARKS")
    logger.info("="*80)
    
    # Pass LLM client for judge evaluation
    benchmark = RAGBenchmark(config, llm_client=llm_client)
    results = benchmark.run_all_benchmarks(retrievers, questions)
    
    # Print summary
    logger.info("="*80)
    logger.info("BENCHMARK RESULTS")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Method':<15} {'Accuracy (%)':<15} {'F1 (%)':<12} {'Avg Tokens':<12} {'Avg Query Time (s)':<18} {'Indexing Time (min)':<20} {'Storage (MB)':<12}")
    print("-"*80)
    
    for row in results['summary']:
        print(
            f"{row['Method']:<15} {row['Accuracy (%)']:<15} {row['F1 (%)']:<12} "
            f"{row['Avg Tokens']:<12} {row['Avg Query Time (s)']:<18} "
            f"{row['Indexing Time (min)']:<20} {row['Storage (MB)']:<12}"
        )
    
    print("="*80)
    
    logger.info(f"Results saved to {config['benchmark']['output_dir']}")
    logger.info("Benchmark complete!")

if __name__ == "__main__":
    main()
