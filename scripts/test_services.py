#!/usr/bin/env python3
"""Test connections to LLM and embedding services."""

import requests
import sys

def test_embedding_service():
    """Test embedding service."""
    print("Testing embedding service at http://localhost:8071/v1/embeddings...")
    try:
        response = requests.post(
            "http://localhost:8071/v1/embeddings",
            json={"input": ["This is a test"], "model": "qwen-embedding"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                embedding = data['data'][0]['embedding']
                print(f"✓ Embedding service OK - returned embedding of shape {len(embedding)}")
                return True
            else:
                print(f"✗ Unexpected response format: {data}")
                return False
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def test_llm_service():
    """Test LLM service."""
    print("\nTesting LLM service at http://localhost:8078/v1...")
    try:
        # Test models endpoint
        response = requests.get("http://localhost:8078/v1/models", timeout=10)
        if response.status_code == 200:
            print(f"✓ LLM service OK - models endpoint accessible")
            data = response.json()
            if 'data' in data:
                print(f"  Available models: {[m.get('id', 'unknown') for m in data['data']]}")
            return True
        else:
            print(f"✗ Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING RAG SERVICES")
    print("="*60)
    
    embed_ok = test_embedding_service()
    llm_ok = test_llm_service()
    
    print("\n" + "="*60)
    if embed_ok and llm_ok:
        print("✓ ALL SERVICES OK")
        sys.exit(0)
    else:
        print("✗ SOME SERVICES FAILED")
        if not embed_ok:
            print("  - Embedding service is not responding")
        if not llm_ok:
            print("  - LLM service is not responding")
        sys.exit(1)
