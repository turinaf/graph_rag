import pickle
import sys
from pathlib import Path
import yaml
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def load_and_inspect_chunks():
    """Load and inspect the saved chunks, embeddings, and graph."""
    
    # Load config to get the cache directory
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Construct path to the pickle file
    processed_dir = Path(config['dataset']['cache_dir']) / 'musique_processed'
    chunks_file = processed_dir / 'chunks.pkl'
    
    if not chunks_file.exists():
        print(f"Pickle file not found at: {chunks_file}")
        print("Run prepare_dataset.py first to create the processed data.")
        return
    
    print(f"Loading data from: {chunks_file}")
    
    # Load the pickle file
    with open(chunks_file, 'rb') as f:
        data = pickle.load(f)
    
    chunks = data['chunks']
    embeddings = data['embeddings']
    graph = data['graph']
    
    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    
    # Inspect chunks
    print(f"\nCHUNKS:")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Type: {type(chunks)}")
    
    if chunks:
        print(f"\nFirst chunk structure:")
        first_chunk = chunks[0]
        print(f"  Type: {type(first_chunk)}")
        if isinstance(first_chunk, dict):
            print(f"  Keys: {list(first_chunk.keys())}")
            for key, value in first_chunk.items():
                print(f"    {key}: {type(value)} - {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        else:
            print(f"  Content: {str(first_chunk)[:200]}{'...' if len(str(first_chunk)) > 200 else ''}")
    
    # Inspect embeddings
    print(f"\nEMBEDDINGS:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Type: {type(embeddings)}")
    print(f"  Data type: {embeddings.dtype}")
    print(f"  Min value: {embeddings.min():.6f}")
    print(f"  Max value: {embeddings.max():.6f}")
    print(f"  Mean: {embeddings.mean():.6f}")
    
    # Inspect graph
    print(f"\nGRAPH:")
    print(f"  Type: {type(graph)}")
    print(f"  Number of nodes: {graph.number_of_nodes()}")
    print(f"  Number of edges: {graph.number_of_edges()}")
    
    if graph.number_of_nodes() > 0:
        # Sample some nodes
        sample_nodes = list(graph.nodes())[:5]
        print(f"  Sample nodes: {sample_nodes}")
        
        # Check node attributes
        if sample_nodes:
            first_node = sample_nodes[0]
            node_data = graph.nodes[first_node]
            print(f"  Node {first_node} attributes: {list(node_data.keys()) if node_data else 'None'}")
    
    if graph.number_of_edges() > 0:
        # Sample some edges
        sample_edges = list(graph.edges(data=True))[:5]
        print(f"  Sample edges (with data): {sample_edges}")
    
    print("\n" + "="*50)
    print("SPECIFIC CHUNKS - 12 AND 86")
    print("="*50)
    
    # Load specific chunks
    # 12, 86, 1207,
    target_chunks = [ 1299]
    
    for chunk_idx in target_chunks:
        if chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            print(f"\n{'='*20} CHUNK {chunk_idx} {'='*20}")
            
            if isinstance(chunk, dict):
                for key, value in chunk.items():
                    print(f"{key}:")
                    if key == 'text':
                        print(f"  {value}")
                    else:
                        print(f"  {value}")
                    print()
            else:
                print(f"Content: {chunk}")
            
            # Show corresponding embedding info
            if chunk_idx < len(embeddings):
                embedding = embeddings[chunk_idx]
                print(f"Embedding shape: {embedding.shape}")
                print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
            
            print("="*60)
        else:
            print(f"\nChunk {chunk_idx} does not exist (only {len(chunks)} chunks available)")

    print("\n" + "="*50)
    print("DETAILED CHUNK INSPECTION (First 3)")
    print("="*50)
    
    # Show first few chunks in detail
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        if isinstance(chunk, dict):
            for key, value in chunk.items():
                if key == 'text':
                    print(f"  {key}: {value[:300]}{'...' if len(value) > 300 else ''}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Content: {str(chunk)[:300]}{'...' if len(str(chunk)) > 300 else ''}")
        print("-" * 30)

if __name__ == "__main__":
    load_and_inspect_chunks()