from EKET.clean import clean_text, clean_documents
from langchain.schema import Document
import pytest

def test_clean_text():
    dirty_text = "This   has  extra spaces.  \nAnd a URL: https://example.com"
    cleaned = clean_text(dirty_text)
    
    assert "  " not in cleaned
    assert "https://" not in cleaned

def test_clean_documents():
    doc = Document(
        page_content="Dirty   text  with  spaces.  ",
        metadata={"source": "test"}
    )
    cleaned = clean_documents([doc])
    
    assert len(cleaned) == 1
    assert "  " not in cleaned[0].page_content