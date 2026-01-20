from pydoc import text
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="EKET",
    version="0.1.6",
    packages=find_packages(),
    license="GPL-3.0-or-later",
        classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    description="EKET: Educational Knowledge Extraction Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        # Google Generative AI
        "google-generativeai==0.8.4",
        "google-ai-generativelanguage==0.6.15",
        "google-auth==2.38.0",
        "google-auth-httplib2==0.2.0",
        "google-auth-oauthlib==1.2.1",
        "google-api-core==2.15.0",
        "googleapis-common-protos==1.70.0",
        "protobuf==4.25.3",
        "httplib2==0.22.0",
        
        # LangChain Ecosystem
        "langchain==0.3.26",
        "langchain-chroma==0.2.4",
        "langchain-community==0.3.27",
        "langchain-core==0.3.68",
        "langchain-experimental==0.3.4",
        "langchain-google-genai==2.0.6",
        "langchain-text-splitters==0.3.8",
        
        # ChromaDB
        "chromadb==1.0.15",
        
        # Loaders
        "unstructured==0.18.3",
        "pytube==15.0.0",
        "pypdf==5.7.0",
        "PyMuPDF==1.23.8",  # Replaces 'fitz' requirement
        "markdown==3.7",
        "openpyxl==3.1.5",
        "python-pptx==1.0.2",
        "lxml==5.4.0",
        "pillow==11.1.0",
        "xmltodict==0.14.2",
        "beautifulsoup4==4.13.3",
        "pandas==2.2.3",
        
        # Utilities
        "scikit-learn==1.6.1",
        "libmagic==1.0",
        "dataclasses-json==0.6.7",
        "requests==2.32.3",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "python-magic==0.4.27",
        "playwright==1.54.0",
        
        # Language detection
        "langdetect==1.0.9",
        "langcodes==3.5.0"
    ],
    python_requires=">=3.8",
)