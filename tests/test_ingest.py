from EKET.ingest import is_structured_text, smart_postprocess_document
from langchain.schema import Document
import pytest



doc = Document(
        page_content="1. First point\n2. Second point",
        metadata={"source": "test"}
    )

def test_is_structured_text():
    # Test with numbered list
    assert is_structured_text(doc.page_content) is True
    assert is_structured_text("- Item 1\n- Item 2") is True
    
    assert is_structured_text("This is a regular paragraph.") is False

def test_smart_postprocess_document():
    processed = smart_postprocess_document(doc)
    
    # Should split into two documents
    assert len(processed) == 2
    
    # First document should contain first point
    assert processed[0].page_content == "1. First point"
    
    # Metadata should be preserved
    assert processed[0].metadata == {"source": "test"}