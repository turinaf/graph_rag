"""
Dataset loading and sampling for HotpotQA.

Creates HotpotQA-S by sampling 20 documents and extracting relevant questions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HotpotQALoader:
    """Loads and samples HotpotQA dataset."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache_dir = Path(config['dataset']['cache_dir'])
        self.sample_size = config['dataset']['sample_size']
        self.random_seed = config['dataset']['random_seed']
        self.max_questions = config['dataset'].get('max_questions', None)
        self.filter_mode = config['dataset'].get('filter_mode', 'support_only')
        self.dataset = None
        
    def download_dataset(self) -> None:
        """Load HotpotQA dataset from local JSON files."""
        logger.info("Loading HotpotQA dataset from local files...")
        
        try:
            # Path to ModelScope downloaded data
            data_dir = Path('./data/HotpotQA/raw').resolve()
            train_file = data_dir / 'hotpot_train_v1.1.json'
            dev_file = data_dir / 'hotpot_dev_fullwiki_v1.json'
            
            if not train_file.exists():
                raise FileNotFoundError(
                    f"HotpotQA dataset not found at {train_file}. "
                    "Please download using: git clone https://www.modelscope.cn/datasets/OpenDataLab/HotpotQA.git"
                )
            
            # Load both files
            logger.info(f"Loading training data from {train_file}...")
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            
            logger.info(f"Loading dev data from {dev_file}...")
            with open(dev_file, 'r') as f:
                dev_data = json.load(f)
            
            # Store as list (mimicking HuggingFace format)
            self.dataset = {'train': train_data + dev_data}
            logger.info(f"Loaded {len(self.dataset['train'])} total examples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def create_hotpotqa_s(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Create HotpotQA-S by sampling 20 documents.
        
        Returns:
            Tuple of (documents, questions)
            - documents: List of {doc_id, title, text}
            - questions: List of {question, answer, supporting_facts, type}
        """
        if self.dataset is None:
            self.download_dataset()
        
        logger.info(f"Creating HotpotQA-S with {self.sample_size} documents...")
        
        random.seed(self.random_seed)
        
        # Collect all unique documents from supporting facts
        all_docs = {}
        all_questions = []
        
        for example in self.dataset['train']:
            # Extract supporting facts (document titles and sentence indices)
            supporting_facts = example['supporting_facts']
            
            # Get document titles and contexts
            # Format: [[title, [sentences]], ...]
            context = example['context']
            
            doc_titles_in_example = []
            for doc_title, sentences in context:
                doc_text = ' '.join(sentences)
                if doc_title not in all_docs:
                    all_docs[doc_title] = {
                        'doc_id': len(all_docs),
                        'title': doc_title,
                        'text': doc_text
                    }
                doc_titles_in_example.append(doc_title)
            
            # Store question info
            all_questions.append({
                'question': example['question'],
                'answer': example['answer'],
                'supporting_facts': supporting_facts,
                'type': example['type'],
                'level': example['level'],
                'doc_titles': doc_titles_in_example
            })
        
        logger.info(f"Found {len(all_docs)} unique documents and {len(all_questions)} questions")
        
        # Sample questions first (greedy) to ensure supporting docs are included and capped by sample_size
        random.shuffle(all_questions)
        selected_questions = []
        selected_doc_titles: set = set()
        
        for q in all_questions:
            supporting_titles = set(title for title, _ in q['supporting_facts'])
            new_docs = supporting_titles - selected_doc_titles
            # If adding these docs exceeds sample_size, skip
            if len(selected_doc_titles) + len(new_docs) > self.sample_size:
                continue
            # Accept question
            selected_questions.append({
                'question': q['question'],
                'answer': q['answer'],
                'type': q['type'],
                'level': q['level']
            })
            selected_doc_titles.update(supporting_titles)
            if len(selected_doc_titles) >= self.sample_size:
                break
        
        # Build sampled docs from selected titles
        sampled_docs = [all_docs[title] for title in selected_doc_titles if title in all_docs]
        filtered_questions = selected_questions
        
        # Limit number of questions if specified
        if self.max_questions and len(filtered_questions) > self.max_questions:
            filtered_questions = random.sample(filtered_questions, self.max_questions)
        
        logger.info(f"Created HotpotQA-S: {len(sampled_docs)} docs, {len(filtered_questions)} questions")
        
        return sampled_docs, filtered_questions
    
    def save_processed_data(self, documents: List[Dict], questions: List[Dict]) -> None:
        """Save processed data to disk."""
        output_dir = self.cache_dir / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'documents.json', 'w') as f:
            json.dump(documents, f, indent=2)
        
        with open(output_dir / 'questions.json', 'w') as f:
            json.dump(questions, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    def load_processed_data(self) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """Load previously processed data."""
        input_dir = self.cache_dir / 'processed'
        
        if not (input_dir / 'documents.json').exists():
            logger.warning("No processed data found")
            return None
        
        with open(input_dir / 'documents.json', 'r') as f:
            documents = json.load(f)
        
        with open(input_dir / 'questions.json', 'r') as f:
            questions = json.load(f)
        
        logger.info(f"Loaded {len(documents)} docs, {len(questions)} questions")
        return documents, questions
