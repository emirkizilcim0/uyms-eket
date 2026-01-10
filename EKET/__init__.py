from .answer import evaluate_answers, answer_checker
from .create import generate_quiz_from_context, parse_mcq, parse_open_ended
from .query import get_context_and_language
from .ingest import load_documents, clean_documents, split_text, save_to_chroma
from .clean import clean_text, clean_documents

__all__ = [
    'evaluate_answers',
    'answer_checker',
    'generate_quiz_from_context',
    'parse_mcq',
    'parse_open_ended',
    'get_context_and_language',
    'load_documents',
    'clean_documents',
    'split_text',
    'save_to_chroma',
    'clean_text'
]

__version__ = '0.1.0'