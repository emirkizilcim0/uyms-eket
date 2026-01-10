import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from EKET.summarization import DocumentSummarizer, get_combined_summary
from EKET.utils import get_config
import json

# Sample test data
SAMPLE_DOCUMENTS = [
    Document(
        page_content="Test content 1",
        metadata={"source": "test1.txt"}
    ),
    Document(
        page_content="Test content 2", 
        metadata={"source": "test2.txt"}
    )
]

@pytest.fixture
def mock_config():
    return {
        'API_KEY': 'test-api-key',
        'CHAT_MODEL': 'gemini-pro',
        'EMBEDDING_MODEL': 'models/embedding-test',
        'SAVE_DATA_DIR': 'test_data_dir'
    }

@pytest.fixture
def mock_genai_response():
    return MagicMock(text="This is a generated combined summary.")

def test_document_summarizer_init(mock_config):
    summarizer = DocumentSummarizer(config=mock_config)
    assert summarizer.config == mock_config
    assert summarizer.model.model_name == mock_config['CHAT_MODEL']

def test_summarize_combined_documents(mock_config, mock_genai_response):
    with patch('google.generativeai.GenerativeModel.generate_content', 
              return_value=mock_genai_response):
        summarizer = DocumentSummarizer(config=mock_config)
        result = summarizer.summarize_combined_documents(SAMPLE_DOCUMENTS)
        
        assert result['summary'] == "This is a generated combined summary."
        assert len(result['sources']) == 2
        assert result['total_chunks'] == 2

def test_summarize_combined_documents_error(mock_config):
    with patch('google.generativeai.GenerativeModel.generate_content', 
              side_effect=Exception("API error")):
        summarizer = DocumentSummarizer(config=mock_config)
        result = summarizer.summarize_combined_documents(SAMPLE_DOCUMENTS)
        
        assert "error" in result
        assert len(result['sources']) == 2

def test_get_combined_summary(mock_config, mock_genai_response):
    with patch('google.generativeai.GenerativeModel.generate_content',
              return_value=mock_genai_response), \
         patch('EKET.summarization.load_context_chunks',
              return_value=SAMPLE_DOCUMENTS), \
         patch('EKET.summarization.get_config',
              return_value=mock_config), \
         patch('EKET.summarization.save_json') as mock_save:
        
        results = get_combined_summary()
        
        assert results['combined_summary']['summary'] == "This is a generated combined summary."
        assert results['total_chunks'] == 2
        mock_save.assert_called_once()

def test_empty_documents(mock_config):
    summarizer = DocumentSummarizer(config=mock_config)
    result = summarizer.summarize_combined_documents([])
    
    assert result['summary'] == ""
    assert result['total_chunks'] == 0
    assert result['sources'] == []

def test_single_document(mock_config, mock_genai_response):
    single_doc = [SAMPLE_DOCUMENTS[0]]
    
    with patch('google.generativeai.GenerativeModel.generate_content', 
              return_value=mock_genai_response):
        summarizer = DocumentSummarizer(config=mock_config)
        result = summarizer.summarize_combined_documents(single_doc)
        
        assert result['summary'] == "This is a generated combined summary."
        assert result['total_chunks'] == 1
        assert result['sources'] == ["test1.txt"]

def test_large_document_combination(mock_config, mock_genai_response):
    large_docs = [
        Document(
            page_content="Large content " * 1000,
            metadata={"source": f"large_{i}.txt"}
        ) for i in range(10)
    ]
    
    with patch('google.generativeai.GenerativeModel.generate_content', 
              return_value=mock_genai_response):
        summarizer = DocumentSummarizer(config=mock_config)
        result = summarizer.summarize_combined_documents(large_docs)
        
        assert len(result['summary']) > 0
        assert result['total_chunks'] == 10

def test_unique_sources(mock_config, mock_genai_response):
    docs_with_duplicates = [
        Document(page_content="Content 1", metadata={"source": "source1"}),
        Document(page_content="Content 2", metadata={"source": "source1"}),
        Document(page_content="Content 3", metadata={"source": "source2"})
    ]
    
    with patch('google.generativeai.GenerativeModel.generate_content',
              return_value=mock_genai_response):
        summarizer = DocumentSummarizer(config=mock_config)
        result = summarizer.summarize_combined_documents(docs_with_duplicates)
        
        assert len(result['sources']) == 2  # Should deduplicate sources
        assert "source1" in result['sources']
        assert "source2" in result['sources']