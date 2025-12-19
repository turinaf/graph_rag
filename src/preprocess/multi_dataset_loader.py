"""
Comprehensive multi-dataset loader for HotpotQA, MuSiQue, 2WikiMultiHop, and RAG-QA Arena.
Handles context window limitations and provides unified interface.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """
    Unified loader for multiple multi-hop QA datasets.
    
    Context window handling:
    - Max chunk size: 2048 chars (~512 tokens)
    - Max document: 8192 chars (~2048 tokens for safe embedding)
    - Embedding limit: 8192 tokens (qwen-embedding)
    - LLM limit: 32768 tokens (Qwen-VLM)
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.max_chunk_size = 2048  # chars
        self.max_doc_size = 8192  # chars (~2048 tokens)
        self.max_context_tokens = 8000  # Safe limit for embedding model
    
    def load_dataset(self, dataset_name: str, split: str = "dev", 
                    max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load dataset by name.
        
        Args:
            dataset_name: 'hotpotqa', 'hotpotqa_full', 'musique', '2wikimultihop', 'ragqa'
            split: Data split (train/dev/test/validation)
            max_samples: Limit number of samples
            
        Returns:
            (documents, questions) tuple
        """
        loader_map = {
            'hotpotqa': self._load_hotpotqa,
            'hotpotqa_full': self._load_hotpotqa_full,
            'musique': self._load_musique,
            '2wikimultihop': self._load_2wikimultihop,
            'ragqa': self._load_ragqa,
        }
        
        if dataset_name not in loader_map:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loader_map.keys())}")
        
        logger.info(f"Loading {dataset_name} ({split}, max_samples={max_samples})...")
        return loader_map[dataset_name](split, max_samples)
    
    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to fit context window."""
        if max_length is None:
            max_length = self.max_doc_size
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _load_hotpotqa(self, split: str, max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """Load HotpotQA-Small (existing preprocessed data)."""
        docs_path = self.data_dir / "raw" / "processed" / "documents.json"
        questions_path = self.data_dir / "raw" / "processed" / "questions.json"
        
        if not docs_path.exists() or not questions_path.exists():
            raise FileNotFoundError(f"HotpotQA data not found in {self.data_dir / 'raw' / 'processed'}")
        
        with open(docs_path) as f:
            documents = json.load(f)
        with open(questions_path) as f:
            questions = json.load(f)
        
        if max_samples:
            questions = questions[:max_samples]
        
        logger.info(f"Loaded HotpotQA: {len(documents)} docs, {len(questions)} questions")
        return documents, questions
    
    def _load_hotpotqa_full(self, split: str, max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """Load full HotpotQA distractor/fullwiki set."""
        # Map split names
        split_map = {
            'dev': 'hotpot_dev_distractor_v1.json',
            'validation': 'hotpot_dev_distractor_v1.json',
            'test': 'hotpot_test_fullwiki_v1.json',
            'train': 'hotpot_train_v1.1.json'
        }
        
        raw_file = self.data_dir / "HotpotQA" / "raw" / split_map.get(split, 'hotpot_dev_distractor_v1.json')
        
        if not raw_file.exists():
            raise FileNotFoundError(f"Full HotpotQA not found: {raw_file}")
        
        with open(raw_file) as f:
            data = json.load(f)
        
        # Shuffle and limit samples
        if max_samples:
            random.shuffle(data)
            data = data[:max_samples]
        
        # Extract documents and questions
        doc_map = {}
        questions = []
        
        for idx, item in enumerate(data):
            # Extract question
            question = {
                'question_id': item.get('_id', f"hotpot_{idx}"),
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'comparison'),
                'supporting_facts': item.get('supporting_facts', [])
            }
            questions.append(question)
            
            # Extract context documents
            for doc_title, doc_sentences in item['context']:
                if doc_title not in doc_map:
                    # Join sentences
                    full_text = ' '.join(doc_sentences) if isinstance(doc_sentences, list) else doc_sentences
                    full_text = self._truncate_text(full_text)
                    
                    doc_map[doc_title] = {
                        'doc_id': f"hotpot_{len(doc_map)}",
                        'title': doc_title,
                        'text': full_text
                    }
        
        documents = list(doc_map.values())
        logger.info(f"Loaded HotpotQA Full ({split}): {len(documents)} docs, {len(questions)} questions")
        return documents, questions
    
    def _load_musique(self, split: str, max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Load MuSiQue dataset.
        
        Format: JSONL with question, answer, paragraphs (title, paragraph_text)
        """
        # Map split names
        split_map = {
            'dev': 'musique_ans_v1.0_dev.jsonl',
            'validation': 'musique_ans_v1.0_dev.jsonl',
            'test': 'musique_ans_v1.0_test.jsonl',
            'train': 'musique_ans_v1.0_train.jsonl'
        }
        
        musique_file = self.data_dir / "musique" / "data" / split_map.get(split, 'musique_ans_v1.0_dev.jsonl')
        
        if not musique_file.exists():
            raise FileNotFoundError(f"MuSiQue not found: {musique_file}")
        
        doc_map = {}
        questions = []
        
        with open(musique_file, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples and len(questions) >= max_samples:
                    break
                
                item = json.loads(line)
                
                # Extract question
                question = {
                    'question_id': item.get('id', f"musique_{idx}"),
                    'question': item['question'],
                    'answer': item['answer'],
                    'answer_aliases': item.get('answer_aliases', []),
                    'type': item.get('question_type', 'multi_hop')
                }
                questions.append(question)
                
                # Extract paragraphs
                for para in item.get('paragraphs', []):
                    title = para.get('title', f"doc_{len(doc_map)}")
                    text = para.get('paragraph_text', '')
                    
                    if title not in doc_map and text:
                        text = self._truncate_text(text)
                        doc_map[title] = {
                            'doc_id': f"musique_{len(doc_map)}",
                            'title': title,
                            'text': text
                        }
        
        documents = list(doc_map.values())
        logger.info(f"Loaded MuSiQue ({split}): {len(documents)} docs, {len(questions)} questions")
        return documents, questions
    
    def _load_2wikimultihop(self, split: str, max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Load 2WikiMultiHop dataset.
        
        Format: JSON with context [[title, sentences]], question, answer
        """
        # Map split names
        split_map = {
            'dev': 'dev.json',
            'validation': 'dev.json',
            'test': 'test.json',
            'train': 'train.json'
        }
        
        wiki_file = self.data_dir / "2wikimultihop" / "data" / split_map.get(split, 'dev.json')
        
        if not wiki_file.exists():
            raise FileNotFoundError(f"2WikiMultiHop not found: {wiki_file}")
        
        with open(wiki_file) as f:
            data = json.load(f)
        
        # Shuffle and limit samples
        if max_samples:
            random.shuffle(data)
            data = data[:max_samples]
        
        doc_map = {}
        questions = []
        
        for idx, item in enumerate(data):
            # Extract question
            question = {
                'question_id': item.get('_id', f"2wiki_{idx}"),
                'question': item['question'],
                'answer': item['answer'],
                'type': item.get('type', 'comparison')
            }
            questions.append(question)
            
            # Extract context documents
            for context_item in item.get('context', []):
                if isinstance(context_item, list) and len(context_item) >= 2:
                    title = context_item[0]
                    sentences = context_item[1]
                    
                    if title not in doc_map:
                        # Join sentences
                        if isinstance(sentences, list):
                            text = ' '.join(sentences)
                        else:
                            text = str(sentences)
                        
                        text = self._truncate_text(text)
                        doc_map[title] = {
                            'doc_id': f"2wiki_{len(doc_map)}",
                            'title': title,
                            'text': text
                        }
        
        documents = list(doc_map.values())
        logger.info(f"Loaded 2WikiMultiHop ({split}): {len(documents)} docs, {len(questions)} questions")
        return documents, questions
    
    def _load_ragqa(self, split: str, max_samples: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Load RAG-QA Arena dataset from ColBERT format.
        
        Format: JSONL with query, positive_passages, negative_passages
        """
        # Find any JSONL file in data/from_colbert/
        colbert_dir = self.data_dir / "rag_qa_arena" / "data" / "from_colbert"
        
        if not colbert_dir.exists():
            raise FileNotFoundError(f"RAG-QA Arena not found: {colbert_dir}")
        
        # Load all available test files
        jsonl_files = list(colbert_dir.glob("*_test.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No test files found in {colbert_dir}")
        
        # Use first available file
        ragqa_file = jsonl_files[0]
        logger.info(f"Loading RAG-QA from: {ragqa_file.name}")
        
        doc_map = {}
        questions = []
        
        with open(ragqa_file, 'r') as f:
            for idx, line in enumerate(f):
                if max_samples and len(questions) >= max_samples:
                    break
                
                item = json.loads(line)
                
                # Extract question (query)
                question = {
                    'question_id': f"ragqa_{idx}",
                    'question': item.get('query', ''),
                    'answer': '',  # RAG-QA doesn't have explicit answers
                    'type': 'retrieval'
                }
                questions.append(question)
                
                # Extract positive passages as documents
                for passage in item.get('positive_passages', []):
                    if isinstance(passage, dict):
                        title = passage.get('title', f"doc_{len(doc_map)}")
                        text = passage.get('text', '')
                    else:
                        title = f"doc_{len(doc_map)}"
                        text = str(passage)
                    
                    if title not in doc_map and text:
                        text = self._truncate_text(text)
                        doc_map[title] = {
                            'doc_id': f"ragqa_{len(doc_map)}",
                            'title': title,
                            'text': text
                        }
        
        documents = list(doc_map.values())
        logger.info(f"Loaded RAG-QA Arena: {len(documents)} docs, {len(questions)} questions")
        return documents, questions


def main():
    """Test dataset loaders."""
    logging.basicConfig(level=logging.INFO)
    
    loader = MultiDatasetLoader()
    
    # Test each dataset
    datasets = ['hotpotqa_full', 'musique', '2wikimultihop', 'ragqa']
    
    for dataset in datasets:
        try:
            docs, questions = loader.load_dataset(dataset, split='dev', max_samples=5)
            print(f"\n{'='*80}")
            print(f"{dataset.upper()}: {len(docs)} docs, {len(questions)} questions")
            print(f"Sample question: {questions[0]['question']}")
            print(f"Sample answer: {questions[0].get('answer', 'N/A')}")
        except Exception as e:
            print(f"Error loading {dataset}: {e}")


if __name__ == "__main__":
    main()
