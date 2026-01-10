from langdetect import detect
from langcodes import Language
from google import genai
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_config, save_json, load_json

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


def get_context_and_language(query_text, path: str | None = None):
    """
    Retrieve relevant context for a query.
    If `path` is provided, load context from that JSON file instead of ChromaDB.
    """
    config = get_config()
    client = genai.Client(api_key=config["API_KEY"])

    # Detect language
    try:
        detected_code = detect(query_text)
        language = Language.get(detected_code).display_name()
    except Exception:
        language = "English"

    # Load context from JSON if provided
    if path:
        try:
            path = Path(path)
            data = load_json(str(path))
            if "context" in data:
                context_chunks = data["context"]
                if isinstance(context_chunks, dict):
                    matching_docs = list(context_chunks.values())
                    matching_ids = list(context_chunks.keys())
                else:
                    matching_docs = context_chunks
                    matching_ids = [f"chunk-{i+1}" for i in range(len(context_chunks))]
                sources = data.get("sources", ["json_file"])
                return matching_docs, matching_ids, language, sources, context_chunks
        except Exception as e:
            logger.error(f"Error loading context from {path}: {e}")
            return None, None, None, None, None

    # ---- ChromaDB retrieval ----
    try:
        db = Chroma(persist_directory=str(config["CHROMA_PATH"]))

        embedding_response = client.models.embed_content(
            model=config["EMBEDDING_MODEL"],
            contents=query_text
        )

        query_embedding = np.array(embedding_response.embeddings[0].values)

        if np.isnan(query_embedding).any():
            raise ValueError("Query embedding contains NaN values")

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None, None, None, None, None

    try:
        collection_data = db._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )

        stored_embeddings = np.array(collection_data["embeddings"])
        valid_indices = [i for i, emb in enumerate(stored_embeddings) if not np.isnan(emb).any()]

        if not valid_indices:
            logger.warning("No valid embeddings found")
            return None, None, None, None, None

        stored_embeddings = stored_embeddings[valid_indices]
        documents = [collection_data["documents"][i] for i in valid_indices]
        metadatas = [collection_data["metadatas"][i] for i in valid_indices]
        ids = (
            [collection_data["ids"][i] for i in valid_indices]
            if "ids" in collection_data else list(range(len(documents)))
        )

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), stored_embeddings
        )[0]

        threshold = 0.55
        indices = np.where(similarities >= threshold)[0]

        if len(indices) == 0:
            return None, None, None, None, None

        matching_docs = [documents[i] for i in indices]
        matching_ids = [ids[i] for i in indices]
        matching_sources = [metadatas[i].get("source", "unknown") for i in indices]

        context_chunks = {
            f"chunk-{id_}": doc for id_, doc in zip(matching_ids, matching_docs)
        }

        return (
            matching_docs,
            matching_ids,
            language,
            list(set(matching_sources)),
            context_chunks
        )

    except Exception as e:
        logger.error(f"Error retrieving from ChromaDB: {e}")
        return None, None, None, None, None


def main(query_text, output_dir: str, path=None):
    matching_docs, matching_ids, language, sources, num_sources_check = get_context_and_language(
        query_text, path=path
    )

    if not matching_docs:
        logger.info("No context found.")
        return None

    context_chunks = {f"chunk-{i+1:03d}": doc for i, doc in enumerate(matching_docs)}
    full_context_text = "\n\n---\n\n".join(matching_docs)

    prompt = (
        "Answer the question based only on the following context.\n"
        "Do not mention chunk numbers or the word 'context'.\n\n"
        f"Context:\n{full_context_text}\n\n"
        f"Question: {query_text}\n"
        f"Answer in {language}:"
    )

    client = genai.Client(api_key=get_config()["API_KEY"])
    response = client.models.generate_content(
        model=get_config()["CHAT_MODEL"],
        contents=prompt
    )

    answer = response.text.strip() if response.text else ""

    output_data = {
        "question": query_text,
        "answer": answer,
        "context": context_chunks,
        "sources": sources or [],
        "language": language,
        "num_sources": num_sources_check,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "context_question_answer.json"
    save_json(output_data, output_file)

    logger.info(f"Results saved to {output_file}")

    return output_data
