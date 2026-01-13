# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2026 Emir Kızılçim, Emir Turgut
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

from typing import List
from langchain.schema import Document
from google import genai
from EKET.utils import get_config, save_json
from pathlib import Path
import json

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Summarizes documents using Gemini Pro with config from utils.py"""
    
    def __init__(self, config):
        self.config = config or get_config()
        self.client = genai.Client(api_key=self.config['API_KEY'])
        self.model_name = self.config['CHAT_MODEL']

        self.summary_prompt = """
        Please provide a brief and on-the-point summary of the following document contents combined.
        Focus on:3
        - Key themes and main points across all documents
        - Important facts/figures that stand out
        - Overall conclusions/recommendations
        - Notable patterns/trends that emerge
        
        Structure your summary with clear sections if appropriate.
        
        Combined Document Content:
        {text}
        """

    def summarize_combined_documents(self, documents: List[Document]):
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.summary_prompt.format(text=combined_text)
            )
            
            sources = []
            for doc in documents:
                src = str(doc.metadata.get("source", "unknown"))
                if src not in sources:
                    sources.append(src)
                
            return {
                "summary": response.text.strip() if response.text else "",
                "sources": sources,
                "total_chunks": len(documents)
            }
        except Exception as e:
            logger.error(f"Error summarizing combined documents: {str(e)}")
            return {
                "error": str(e),
                "sources": [str(doc.metadata.get("source", "unknown")) for doc in documents]
            }

def load_context_chunks(path: str):
    """Load chunks from a given JSON path and convert to Document objects"""
    try:
        context_path = Path(path)
        
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found at {context_path}")
        
        with open(context_path, "r", encoding="utf-8") as f:
            context_data = json.load(f)
        
        chunks = []
        for chunk_id, content in context_data["context"].items():
            chunks.append(Document(
                page_content=content,
                metadata={
                    "source": context_data.get("sources", ["unknown"])[0],
                    "chunk_id": chunk_id
                }
            ))
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error loading context chunks: {str(e)}")
        raise

def get_combined_summary(input_path: str, output_dir: str):
    summarizer = DocumentSummarizer(config=get_config())

    logger.info(f"Loading chunks from {input_path}...")
    chunks = load_context_chunks(input_path)
    
    logger.info(f"\nGenerating combined summary of {len(chunks)} chunks...")
    combined_summary = summarizer.summarize_combined_documents(chunks)
    if "summary" in combined_summary:
        logger.info(combined_summary["summary"])
    else:
        logger.error(combined_summary.get("error", "Unknown summarization error"))

    
    results = {
        "combined_summary": combined_summary,
        "total_chunks": len(chunks)
    }

    # Ensure output folder exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always save as combined_summary.json inside output_dir
    output_file = output_dir / "combined_summary.json"
    save_json(results, output_file)

    logger.info(f"Saved combined summary to {output_file}")
    
    return results


import argparse
def main():
    """Automated summarization pipeline for context JSON"""
    parser = argparse.ArgumentParser(description="Summarize document chunks")
    parser.add_argument("--input", type=str, required=True, help="Path to context JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save results")
    args = parser.parse_args()

    try:
        logger.info("=== Starting Document Summarization ===")
        results = get_combined_summary(args.input, args.output)
    except Exception as e:
        logger.error(f"\nError in summarization pipeline: {str(e)}")
        raise
    
    return results



if __name__ == "__main__":
    main()