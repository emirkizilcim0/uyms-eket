import pytest
from unittest.mock import patch, MagicMock
from EKET.query import get_context_and_language

@pytest.fixture
def mock_embed():
    with patch('google.generativeai.embed_content') as mock:
        mock.return_value = {"embedding": [0.1, 0.2, 0.3]}
        yield mock

@pytest.fixture
def mock_chroma():
    with patch('tutor.query.Chroma') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        
        # Setup collection mock
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'documents': ["doc1", "doc2"],
            'metadatas': [{"source": "test1"}, {"source": "test2"}]
        }
        mock_instance._collection = mock_collection
        yield mock

def test_get_context_and_language(mock_embed, mock_chroma, config):
    context, language, sources, count = get_context_and_language("test query")
    
    assert isinstance(context, str)
    assert context == "doc1\n\n---\n\ndoc2"
    assert isinstance(language, str)
    assert isinstance(sources, set)
    assert isinstance(count, int)