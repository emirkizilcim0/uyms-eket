import pytest
import os
from pathlib import Path
from EKET.utils import get_config

@pytest.fixture
def config():
    return get_config()

@pytest.fixture
def sample_context():
    return """
    Machine learning is a subset of artificial intelligence that focuses on building 
    systems that learn from data. There are three main types: supervised learning, 
    unsupervised learning, and reinforcement learning.
    """

@pytest.fixture
def sample_questions_data():
    return {
        "language": "English",
        "mcq": [
            {
                "Q1": "What is machine learning?",
                "A11": {"text": "A type of database", "correct": False},
                "A12": {"text": "A subset of AI", "correct": True},
                "A13": {"text": "A programming language", "correct": False},
                "E1": "Machine learning is a subset of AI that focuses on building systems that learn from data."
            }
        ],
        "open_ended": [
            {
                "Q1": "Name the three main types of machine learning.",
                "E1": "The three main types are supervised learning, unsupervised learning, and reinforcement learning."
            }
        ]
    }