"""
Helper functions for the RAG benchmarking system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import re


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(prompts_path: str) -> Dict[str, str]:
    """Load prompts from YAML file."""
    with open(prompts_path, 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars ≈ 1 token)."""
    return len(text) // 4


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
