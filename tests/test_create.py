from EKET.create import generate_quiz_from_context, parse_mcq, parse_open_ended
import pytest

def test_generate_quiz_from_context(sample_context):
    mcq, open_ended = generate_quiz_from_context(sample_context)
    
    assert isinstance(mcq, str)
    assert isinstance(open_ended, str)
    assert "1." in mcq  # Check for question numbering

def test_parse_mcq():
    mcq_text = """
1. What is machine learning?
   A) A type of database (False)
   B) A subset of AI (True)
   C) A programming language (False)
   Explanation: Machine learning is a subset of AI.
"""
    questions = parse_mcq(mcq_text)
    
    assert len(questions) == 1
    assert questions[0]["Q1"] == "What is machine learning?"
    assert questions[0]["A11"]["text"] == "A type of database"
    assert questions[0]["A12"]["correct"] == True

def test_parse_open_ended():
    open_text = """
1. Name the types of machine learning.
Explanation: The main types are supervised, unsupervised, and reinforcement learning.
"""
    questions = parse_open_ended(open_text)
    
    assert len(questions) == 1
    assert questions[0]["Q1"] == "Name the types of machine learning."
    assert "supervised" in questions[0]["E1"].lower()