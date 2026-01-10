from datetime import datetime
import sys

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    YoutubeLoader,
    BSHTMLLoader
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
import re
from google import genai
from langchain.embeddings.base import Embeddings
from .utils import get_config, save_json
from .clean import clean_documents
import pandas as pd
import logging

##############################################################################
#                              IMAGE PROCESSOR                               #
##############################################################################

import base64
from PIL import Image
config = get_config()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Log to stderr to avoid interfering with JSON stdout
)

logger = logging.getLogger(__name__)

class GeminiImageLoader:
    """Simplified image loader using only Gemini Vision"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        try:
            cfg = get_config()
            self.client = genai.Client(api_key=cfg["API_KEY"])
            self.model_name = "gemini-1.5-flash"
            self.gemini_available = True
        except Exception as e:
            logger.warning(f"Gemini initialization failed: {e}")
            self.gemini_available = False
    
    def load(self):
        """Load image with basic metadata and optional Gemini analysis"""
        try:
            with Image.open(self.file_path) as img:
                metadata = {
                    "source": self.file_path,
                    "file_type": "image",
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(self.file_path)
                    ).isoformat()
                }
            
            content = "Image file: " + os.path.basename(self.file_path)
            
            if self.gemini_available:
                try:
                    with open(self.file_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[
                            "Describe this image in detail including any text, objects, and context.",
                            {
                                "mime_type": f"image/{os.path.splitext(self.file_path)[1][1:]}",
                                "data": image_data
                            }
                        ]
                    )

                    content = "IMAGE ANALYSIS:\n" + (response.text or "")
                except Exception as e:
                    content += f"\n(Image analysis failed: {str(e)})"
            
            return [Document(page_content=content, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error processing image {self.file_path}: {str(e)}")
            return []

##############################################################################
#                              IMAGE PROCESSOR                               #
##############################################################################




##############################################################################
#                          XML DOCUMENT PROCESSOR                            #
##############################################################################

from typing import List, Dict, Any, Union, Optional, Generator
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import xmltodict
import json
from lxml import etree
from langchain.schema import Document
import os

class XMLProcessor:
    """XML document processor that preserves all original values exactly"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "preferred_method": "auto",
            "include_attributes": True,
            "max_depth": 10,
            "skip_namespaces": True,
            "text_separator": "\n",
            "pretty_print_json": True,
            "large_file_threshold": 10 * 1024 * 1024,
            "preserve_whitespace": False
        }
    
    def load_xml_document(self, filepath: str) -> List[Document]:
        try:
            if os.path.getsize(filepath) > self.config["large_file_threshold"]:
                return list(self.process_large_xml(filepath))
            
            with open(filepath, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            soup = BeautifulSoup(xml_content, 'lxml-xml')
            if not soup.find():
                raise ValueError("No valid XML content found")
            
            cleaned_xml = str(soup)
            parse_method = self._determine_parsing_method(cleaned_xml)
            
            if parse_method == "text":
                content = self._extract_all_text(cleaned_xml)
            elif parse_method == "structured":
                content = self._parse_to_structured_text(cleaned_xml)
            elif parse_method == "json":
                content = self._parse_to_json(cleaned_xml)
            elif parse_method == "markdown":
                content = self._parse_to_markdown(cleaned_xml)
            else:
                content = self._auto_select_parsing(cleaned_xml)
            
            return [Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "file_type": "xml",
                    "parsing_method": parse_method,
                    "xml_type": self._detect_xml_type(cleaned_xml)
                }
            )]
        
        except Exception as e:
            logger.error(f"Error processing XML file {filepath}: {e}")
            return []

    def _determine_parsing_method(self, xml_content: str) -> str:
        """Determine the best parsing method based on content and config"""
        if self.config["preferred_method"] != "auto":
            return self.config["preferred_method"]
        
        try:
            root = ET.fromstring(xml_content)
            if len(list(root.iter())) > 100:
                return "structured"
            if any(len(list(elem)) > 5 for elem in root.iter()):
                return "json"
            if sum(1 for _ in root.itertext()) > 20:
                return "text"
            return "markdown"
        except:
            return "text"

    def _extract_all_text(self, xml_content: str) -> str:
        """Extract all text content from XML exactly as-is"""
        root = ET.fromstring(xml_content)
        texts = []
        for t in root.itertext():
            if self.config["preserve_whitespace"] or t.strip():
                texts.append(t.strip() if not self.config["preserve_whitespace"] else t)
        return self.config["text_separator"].join(texts)

    def _parse_to_structured_text(self, xml_content: str) -> str:
        """Convert XML to structured text representation"""
        root = ET.fromstring(xml_content)
        return json.dumps(
            self._xml_to_dict(root),
            indent=2 if self.config["pretty_print_json"] else None,
            ensure_ascii=False
        )

    def _parse_to_json(self, xml_content: str) -> str:
        """Convert XML to JSON string using xmltodict"""
        return json.dumps(
            xmltodict.parse(xml_content),
            indent=2 if self.config["pretty_print_json"] else None,
            ensure_ascii=False
        )

    def _parse_to_markdown(self, xml_content: str) -> str:
        """Convert XML to human-readable Markdown"""
        root = ET.fromstring(xml_content)
        return self._xml_to_markdown(root)

    def _auto_select_parsing(self, xml_content: str) -> str:
        """Automatically select the best parsing method"""
        try:
            root = ET.fromstring(xml_content)
            if any(elem.text and list(elem) for elem in root.iter()):
                return self._parse_to_structured_text(xml_content)
            if max(len(list(elem)) for elem in root.iter()) < 3:
                return self._extract_all_text(xml_content)
            return self._parse_to_structured_text(xml_content)
        except:
            return self._extract_all_text(xml_content)

    def _xml_to_dict(self, element: ET.Element) -> Dict:
        """Convert XML element to dictionary preserving original values"""
        result = {}
        
        if self.config["include_attributes"] and element.attrib:
            result["@attributes"] = element.attrib
        
        for child in element:
            if len(child) == 0:
                child_data = child.text if child.text is not None else None
            else:
                child_data = self._xml_to_dict(child)
            
            tag = child.tag
            if self.config["skip_namespaces"] and "}" in tag:
                tag = tag.split("}")[-1]
            
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(child_data)
                else:
                    result[tag] = [result[tag], child_data]
            else:
                result[tag] = child_data
        
        return result

    def _xml_to_markdown(self, element: ET.Element, level: int = 0) -> str:
        """Convert XML to Markdown preserving original values"""
        lines = []
        indent = "  " * level
        tag_display = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        
        if self.config["include_attributes"] and element.attrib:
            attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())
            lines.append(f"{indent}- **<{tag_display} {attrs}>**")
        else:
            lines.append(f"{indent}- **<{tag_display}>**")
        
        if element.text is not None:
            text = element.text if self.config["preserve_whitespace"] else element.text.strip()
            if text or self.config["preserve_whitespace"]:
                lines.append(f"{indent}  {text}")
        
        for child in element:
            lines.append(self._xml_to_markdown(child, level + 1))
        
        return "\n".join(lines)

    def _detect_xml_type(self, xml_content: str) -> str:
        """Detect the type of XML document"""
        try:
            root = ET.fromstring(xml_content)
            if root.tag.endswith("}html") or root.tag == "html":
                return "html"
            if root.tag.endswith("}feed") or root.tag == "feed":
                return "rss/atom"
            if root.tag.endswith("}FMPXMLRESULT") or root.tag == "FMPXMLRESULT":
                return "filemaker"
            if any(e.tag.endswith("}schema") or e.tag == "schema" for e in root.iter()):
                return "xsd"
            if root.tag.endswith("}kml") or root.tag == "kml":
                return "kml"
            return "generic"
        except:
            return "unknown"

    def process_large_xml(self, filepath: str) -> Generator[Document, None, None]:
        """Process very large XML files incrementally"""
        context = etree.iterparse(filepath, events=("end",), huge_tree=True)
        
        try:
            for event, elem in context:
                if elem.tag is not None:
                    content = self._xml_to_markdown(elem)
                    yield Document(
                        page_content=content,
                        metadata={
                            "source": filepath,
                            "xml_tag": elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag,
                            "file_type": "xml_chunk"
                        }
                    )
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        finally:
            del context

##############################################################################
#                          XML DOCUMENT PROCESSOR                            #
##############################################################################



# Use config values instead of hardcoded ones
config = get_config()


class GeminiEmbedding(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=config["API_KEY"])
        self.model = config["EMBEDDING_MODEL"]

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            res = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(res.embeddings[0].values)
        return embeddings

    def embed_query(self, text):
        res = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return res.embeddings[0].values


def save_to_chroma(chunks: List[Document]):
    """Save document chunks to ChromaDB with error handling"""
    try:
        if os.path.exists(config['CHROMA_PATH']):
            shutil.rmtree(config['CHROMA_PATH'])

        embedding_model = GeminiEmbedding()

        if not chunks:
            raise ValueError("No chunks to save to Chroma.")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(config['CHROMA_PATH'])
        )
        
        if hasattr(db, 'persist'):
            db.persist()
        
        logger.info(f"Successfully saved {len(chunks)} chunks to Chroma")
        return db
        
    except Exception as e:
        logger.error(f"Error saving to Chroma: {str(e)}")
        raise

def is_structured_text(text):
    """Check if text appears to be a structured list/document"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    
    # Count lines that look like list items (numbered or bulleted)
    list_items = sum(
        1 for line in lines 
        if re.match(r'^(\d+[\.\)]|[-*+])\s', line)  # matches: 1., 1), -, *, +
    )
    return list_items / len(lines) > 0.5  # >50% of lines are list items

# If it is an structrued document, then we split the lines and 
def smart_postprocess_document(doc):
    """Split structured documents into individual list items"""
    if is_structured_text(doc.page_content):
        logger.warning("Structured document detected — splitting line by line.")
        lines = [
            line.strip() 
            for line in doc.page_content.splitlines() 
            if line.strip()
        ]
        return [
            Document(
                page_content=line,
                metadata=doc.metadata
            )
            for line in lines
        ]
    return [doc]


import magic
import mimetypes  # Add this at the top with other imports

def detect_file_type(filepath):
    """Improved file type detection with fallbacks"""
    try:
        # Protection from malwares
        mime = magic.from_file(filepath, mime=True)
        if mime != 'application/octet-stream':
            return mime
        
        # Fallback to mimetypes
        mime, _ = mimetypes.guess_type(filepath)
        if mime:
            return mime
            
        # Check file extension as last resort
        if filepath.lower().endswith('.pdf'):
            return 'application/pdf'
            
        return None
    except:
        # If all fails, check extension
        if filepath.lower().endswith('.pdf'):
            return 'application/pdf'
        return None


# Updated loader mapping with error handling
# Those tags are from magic -> https://mimetype.io/all-types
LOADER_MAPPING = {
    "application/pdf": PyPDFLoader,
    "text/markdown": UnstructuredMarkdownLoader,
    "text/csv": CSVLoader,
    "application/json": JSONLoader,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": UnstructuredPowerPointLoader,
    "text/plain": TextLoader,
    "text/html": BSHTMLLoader,
    "image/jpeg": GeminiImageLoader,
    "image/png": GeminiImageLoader,
    "image/jpg": GeminiImageLoader,
    "application/xml": "xml_processor",
    "text/xml": "xml_processor"
}

from playwright.sync_api import sync_playwright

def render_html_with_playwright(filepath: str):
    """Render HTML using Playwright"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"file://{os.path.abspath(filepath)}", wait_until="networkidle")
            rendered_html = page.content()
            browser.close()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        return filepath
    except Exception as e:
        logger.error(f"Failed to render HTML with Playwright: {e}")
        return None

from pathlib import Path

def load_documents(args):
    """Load documents from various sources"""
    documents = []
    xml_processor = XMLProcessor(config.get("xml_processing", {}))

    if args.youtube:
        try:
            loader = YoutubeLoader.from_youtube_url(args.youtube)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source": args.youtube,
                    "mime_type": "video/youtube"
                })
            logger.info(f"Loaded {len(documents)} documents from YouTube")
            return documents
        except Exception as e:
            logger.error(f"Failed to load YouTube video: {e}")
            return []

    if args.file:
        filepath = Path(args.file).absolute()
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return []

        filename = filepath.name
        mime_type = detect_file_type(str(filepath))
        loader_class = LOADER_MAPPING.get(mime_type)

        if not loader_class:
            logger.warning(f"Skipping unsupported MIME type: {mime_type} ({filename})")
            return []

        logger.info(f"Loading {filename} (MIME: {mime_type})")

        try:
            # Special handling for XML
            if mime_type in ["application/xml", "text/xml"]:
                docs = xml_processor.load_xml_document(str(filepath))
                for doc in docs:
                    processed_docs = smart_postprocess_document(doc)
                    documents.extend(processed_docs)
                logger.warning(f"Loaded {len(docs)} XML documents from {filename}")
                return documents

            # Special handling for HTML
            elif mime_type == "text/html":
                rendered_path = render_html_with_playwright(str(filepath))
                if not rendered_path:
                    logger.warning(f"Skipping {filename} due to rendering failure.")
                    return []

                with open(rendered_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                visible_text = soup.get_text(separator="\n", strip=True)
                if not visible_text.strip():
                    logger.warning(f"No visible text in {filename}")
                    return []
                doc = Document(page_content=visible_text, metadata={"source": str(filepath), "mime_type": mime_type})
                processed_docs = smart_postprocess_document(doc)
                documents.extend(processed_docs)
                return documents

            # Special handling for Excel
            elif filename.endswith(('.xlsx', '.xls')):
                try:
                    excel_data = pd.read_excel(filepath, sheet_name=None)
                except Exception as e:
                    logger.error(f"Failed to read Excel file {filename}: {e}")
                    return []
                for sheet_name, df in excel_data.items():
                    if df.empty:
                        continue
                    for index, row in df.iterrows():
                        content_parts = []
                        for col in df.columns:
                            val = row[col]
                            if pd.notna(val):
                                if isinstance(val, float) and val.is_integer():
                                    val = int(val)
                                content_parts.append(f"{col}: {val}")
                        if not content_parts:
                            continue
                        content = "\n".join(content_parts)
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(filepath),
                                "sheet": sheet_name,
                                "row": index + 1,
                                "file_type": "excel",
                                "mime_type": "application/vnd.ms-excel"
                            }
                        )
                        processed_docs = smart_postprocess_document(doc)
                        documents.extend(processed_docs)
                return documents

            # Photo analyzing
            elif mime_type in ["image/jpeg", "image/png", "image/jpg"]:
                loader = loader_class(str(filepath))
            
            # CSV Loader
            elif mime_type == "text/csv":
                loader = loader_class(str(filepath), encoding="utf-8")
            
            
            # General loader path
            # PDF, JSON, TXT, MarkDown Loaders
            else:
                loader = loader_class(str(filepath))
            
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(filepath)
                doc.metadata["mime_type"] = mime_type
                processed_docs = smart_postprocess_document(doc)
                documents.extend(processed_docs)

            logger.warning(f"Loaded {len(docs)} doc(s) from {filename}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return []

    return documents

def semantic_chunker_need(doc, wants_deep_search: bool) -> bool:
    """Determine if semantic chunking should be used"""
    content_length = len(doc.page_content)
    if wants_deep_search:
        return True
    return 5000 < content_length < 35000

def get_text_splitter(text_length, wants_deep_search: bool):
    """Get appropriate text splitter based on content length"""
    if wants_deep_search or 5000 <= text_length < 30000:
        return SemanticChunker(GeminiEmbedding())
    elif text_length < 1000:
        return RecursiveCharacterTextSplitter(chunk_size=100*3, chunk_overlap=50)       # high chunk_size and lower overlap will yield better results.
    elif 1000 <= text_length < 5000:
        return RecursiveCharacterTextSplitter(chunk_size=300*3, chunk_overlap=100)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=750*3, chunk_overlap=250)

def split_text(documents: List[Document]):
    """Split documents into chunks"""
    if not documents:
        logger.warning("No documents provided for splitting")
        return []

    all_chunks = []
    for doc in documents:
        try:
            use_semantic = semantic_chunker_need(doc, False)
            splitter = get_text_splitter(len(doc.page_content), use_semantic)
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
            continue

    logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    if all_chunks:
        logger.debug(f"Sample chunk: {all_chunks[0].page_content[:200]}...")
    return all_chunks

def save(context_dict: Dict, chunks: List[Document], output_dir: str):
    """Ensure output is valid JSON for the backend"""
    result = {
        "context": context_dict,
        "sources": list({c.metadata.get("source", "unknown") for c in chunks}),
        "status": "success"
    }

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "context_language.json")

    save_json(result, out_path)

    print(json.dumps(result, ensure_ascii=False))  # Only JSON goes to stdout
    return result

import argparse
def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--youtube", type=str, help="YouTube video URL to process")
    parser.add_argument("--file", type=str, help="Path to a file to process")
    parser.add_argument("--output", "-o", type=str, default="saved_data", help="Output folder")
    args = parser.parse_args()

    if not args.youtube and not args.file:
        error_result = {
            "status": "error",
            "message": "You must specify either --youtube or --file"
        }
        print(json.dumps(error_result))
        return

    try:
        logger.info("=== Document Ingestion Pipeline ===")

        logger.info("1. Loading documents...")
        documents = load_documents(args=args)
        if not documents:
            error_result = {"status": "error", "message": "No documents loaded"}
            print(json.dumps(error_result))
            return

        logger.info("2. Cleaning documents...")
        cleaned_docs = clean_documents(documents)

        logger.info("3. Splitting documents...")
        chunks = split_text(cleaned_docs)

        logger.info("4. Saving to ChromaDB...")
        save_to_chroma(chunks)

        logger.info("5. Generating output...")
        context_dict = {
            f"chunk-{i+1:03d}": chunk.page_content
            for i, chunk in enumerate(chunks)
        }
        return save(context_dict, chunks, args.output)

    except Exception as e:
        error_result = {
            "status": "error",
            "message": str(e)
        }
        print(json.dumps(error_result))
        logger.exception("Pipeline failed")
        raise

if __name__ == "__main__":
    main()