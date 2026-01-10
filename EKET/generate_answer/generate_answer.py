from EKET.query import get_context_and_language
from EKET.utils import save_json, get_config
from google import genai
from langchain_community.vectorstores import Chroma
import os
import json
from datetime import datetime

import logging
import sys
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


def generate_answer(query, input_path, output_dir):
    """Generate answer and save results as JSON"""

    config = get_config()
    client = genai.Client(api_key=config["API_KEY"])

    # Get matching documents
    matching_docs, matching_ids, language, sources, context_chunks = get_context_and_language(
        query,
        path=input_path
    )

    if not matching_docs:
        result = {
            "question": query,
            "answer": "No relevant information found.",
            "sources": [],
            "used_chunks": [],
            "timestamp": datetime.now().isoformat()
        }
        save_results(result, output_dir)
        return result

    # Build numbered chunks (SAFE mapping)
    numbered_chunks = []
    for i, doc in enumerate(matching_docs):
        chunk_id = f"chunk-{i+1:03d}"
        numbered_chunks.append({
            "chunk_id": chunk_id,
            "text": doc,
            "source": sources[i] if i < len(sources) else "unknown"
        })

    # Prepare context
    context_text = "\n\n---\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in numbered_chunks
    )

    prompt = (
        f"Answer the question in {language} using ONLY the context below.\n"
        f"Do not mention chunk numbers or say 'according to the context'.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}"
    )

    response = client.models.generate_content(
        model=config["CHAT_MODEL"],
        contents=prompt
    )

    answer = response.text.strip() if response.text else ""

    result = {
        "question": query,
        "answer": answer,
        "language": language,
        "sources": list(set(chunk["source"] for chunk in numbered_chunks)),
        "used_chunks": numbered_chunks,
        "timestamp": datetime.now().isoformat()
    }

    save_results(result, output_dir)
    return result


def save_results(result, output_dir):
    """Save results to JSON file"""

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "query_answer.json")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def generate_answer_from_question():
    """CLI entry point"""

    parser = argparse.ArgumentParser(description="Ask a question and get an answer.")
    parser.add_argument("--query", "-q", required=True, help="Your question")
    parser.add_argument("--input", "-i", default=None, help="Optional context JSON")
    parser.add_argument("--output", "-o", default="saved_data", help="Output directory")

    args = parser.parse_args()

    result = generate_answer(args.query, args.input, args.output)

    logger.info(f"\n--- ANSWER ---\n{result['answer']}")

    if result["used_chunks"]:
        logger.info(f"\n--- USED CONTEXT CHUNKS ({len(result['used_chunks'])}) ---")
        for chunk in result["used_chunks"]:
            logger.info(f"\n{chunk['chunk_id']} (from {chunk['source']}):")
            logger.info(chunk["text"][:200] + "...")

    if result["sources"]:
        logger.info(f"\n--- SOURCES ---")
        for source in result["sources"]:
            logger.info(f"- {source}")

    return result
