"""Download and prepare MuSiQue, 2WikiMultiHop, and RAG-QA Arena datasets."""

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List
import zipfile
import gzip
import shutil


def download_file(url: str, dest_path: str) -> None:
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")


def download_musique():
    """Download MuSiQue dataset from GitHub."""
    base_url = "https://raw.githubusercontent.com/stonybrooknlp/musique/main/data"
    data_dir = Path("data/musique")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dev and test sets
    files = {
        "dev": "musique_ans_v1.0_dev.jsonl",
        "test": "musique_ans_v1.0_test.jsonl"
    }
    
    for split, filename in files.items():
        url = f"{base_url}/{filename}"
        dest = data_dir / filename
        if not dest.exists():
            try:
                download_file(url, str(dest))
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print(f"Please manually download from: {url}")
        else:
            print(f"Already exists: {dest}")
    
    return data_dir


def download_2wikimultihop():
    """Download 2WikiMultiHop dataset from GitHub."""
    base_url = "https://raw.githubusercontent.com/Alab-NII/2wikimultihop/master/dataset"
    data_dir = Path("data/2wikimultihop")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dev and test sets
    files = {
        "dev": "dev.json",
        "test": "test.json"
    }
    
    for split, filename in files.items():
        url = f"{base_url}/{filename}"
        dest = data_dir / filename
        if not dest.exists():
            try:
                download_file(url, str(dest))
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print(f"Please manually download from: {url}")
        else:
            print(f"Already exists: {dest}")
    
    return data_dir


def download_rag_qa_arena():
    """Download RAG-QA Arena dataset."""
    # Note: RAG-QA Arena requires AWS credentials or manual download
    # We'll create a placeholder structure
    data_dir = Path("data/rag_qa_arena")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("RAG-QA Arena requires manual download from:")
    print("https://github.com/awslabs/rag-qa-arena")
    print(f"Please download and place files in: {data_dir}")
    
    return data_dir


def convert_musique_to_standard(input_path: Path, output_path: Path, max_samples: int = None):
    """Convert MuSiQue JSONL format to standard format."""
    questions = []
    documents = []
    doc_id_map = {}
    
    with open(input_path, 'r') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
                
            item = json.loads(line)
            
            # Extract question and answer
            question = {
                'id': item.get('id', f'musique_{idx}'),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'type': item.get('question_type', 'multi_hop')
            }
            questions.append(question)
            
            # Extract paragraphs as documents
            if 'paragraphs' in item:
                for para in item['paragraphs']:
                    doc_id = para.get('idx', f'doc_{len(documents)}')
                    if doc_id not in doc_id_map:
                        doc = {
                            'id': doc_id,
                            'title': para.get('title', ''),
                            'text': para.get('paragraph_text', '')
                        }
                        documents.append(doc)
                        doc_id_map[doc_id] = len(documents) - 1
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'questions': questions,
            'documents': documents
        }, f, indent=2)
    
    print(f"Converted {len(questions)} questions and {len(documents)} documents")
    return len(questions), len(documents)


def convert_2wikimultihop_to_standard(input_path: Path, output_path: Path, max_samples: int = None):
    """Convert 2WikiMultiHop format to standard format."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    documents = []
    doc_id_map = {}
    
    for idx, item in enumerate(data):
        if max_samples and idx >= max_samples:
            break
            
        # Extract question and answer
        question = {
            'id': item.get('_id', f'2wiki_{idx}'),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'type': item.get('type', 'comparison')
        }
        questions.append(question)
        
        # Extract context as documents
        if 'context' in item:
            for context_item in item['context']:
                doc_id = f"doc_{len(documents)}"
                # Context is typically [title, sentences]
                if isinstance(context_item, list) and len(context_item) >= 2:
                    title = context_item[0]
                    text = ' '.join(context_item[1]) if isinstance(context_item[1], list) else context_item[1]
                else:
                    title = ''
                    text = str(context_item)
                
                doc = {
                    'id': doc_id,
                    'title': title,
                    'text': text
                }
                documents.append(doc)
                doc_id_map[doc_id] = len(documents) - 1
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'questions': questions,
            'documents': documents
        }, f, indent=2)
    
    print(f"Converted {len(questions)} questions and {len(documents)} documents")
    return len(questions), len(documents)


def main():
    """Download and convert all datasets."""
    print("="*80)
    print("DOWNLOADING MULTI-HOP QA DATASETS")
    print("="*80)
    
    # Download MuSiQue
    print("\n[1/3] MuSiQue Dataset")
    musique_dir = download_musique()
    
    # Convert MuSiQue dev set
    musique_dev = musique_dir / "musique_ans_v1.0_dev.jsonl"
    if musique_dev.exists():
        output_path = Path("data/musique/processed_dev.json")
        convert_musique_to_standard(musique_dev, output_path, max_samples=500)
    
    # Download 2WikiMultiHop
    print("\n[2/3] 2WikiMultiHop Dataset")
    wiki_dir = download_2wikimultihop()
    
    # Convert 2WikiMultiHop dev set
    wiki_dev = wiki_dir / "dev.json"
    if wiki_dev.exists():
        output_path = Path("data/2wikimultihop/processed_dev.json")
        convert_2wikimultihop_to_standard(wiki_dev, output_path, max_samples=500)
    
    # RAG-QA Arena
    print("\n[3/3] RAG-QA Arena Dataset")
    download_rag_qa_arena()
    
    print("\n" + "="*80)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*80)
    print("\nProcessed datasets saved to:")
    print("  - data/musique/processed_dev.json")
    print("  - data/2wikimultihop/processed_dev.json")
    print("\nNote: Some datasets may require manual download due to access restrictions.")


if __name__ == "__main__":
    main()
