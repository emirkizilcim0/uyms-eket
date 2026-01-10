# Emir bey buradaydı.

from pathlib import Path
import os
import json

def get_config():
    """Load configuration from environment or defaults, using current working directory"""
    base_dir = Path.cwd()  # <-- ensures paths are relative to where script is run
    return {
        'API_KEY': os.getenv('TUTOR_API_KEY', 'API_KEY'),   # User yours please...
        'CHAT_MODEL': os.getenv('TUTOR_MODEL', 'models/gemini-2.5-flash'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL', 'models/embedding-001'),
        'DATA_DIR': base_dir / 'data',                                              # file path as args in ingest.py
        'SAVE_DATA_DIR': base_dir / 'saved_data',
        'CHROMA_PATH': base_dir / 'chroma'                                            
    }

def save_json(data, filename, subdir=None):
    """Save JSON data to file with consistent path handling"""
    config = get_config()
    output_dir = Path(config['SAVE_DATA_DIR'])
    if subdir:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filename, subdir=None):
    """Load JSON data from file"""
    config = get_config()
    input_dir = Path(config['SAVE_DATA_DIR'])
    if subdir:
        input_dir = input_dir / subdir
    
    with open(input_dir / filename, 'r', encoding='utf-8') as f:
        return json.load(f)
