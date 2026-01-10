from EKET.answer import evaluate_answers
import pytest

def test_evaluate_answers_mcq(sample_questions_data):
    results = evaluate_answers(sample_questions_data)
    
    assert "score" in results
    assert results["score"]["total_mcq"] == 1
    assert results["mcq_results"][0]["question"] == "What is machine learning?"
    assert results["mcq_results"][0]["correct_answer"] == "B"

def test_evaluate_answers_open_ended(sample_questions_data):
    results = evaluate_answers(sample_questions_data)
    
    assert "open_ended_results" in results
    assert results["open_ended_results"][0]["question"] == "Name the three main types of machine learning."