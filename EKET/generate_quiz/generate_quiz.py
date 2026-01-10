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

from EKET.create import generate_quiz_from_context, parse_mcq, parse_open_ended
from EKET.utils import load_json, save_json, get_config
from datetime import datetime
from pathlib import Path

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


def get_context_from_file(input_path=None):
    """Extract context chunks from a given JSON file or from default saved_data"""
    try:
        config = get_config()
        saved_data_path = config['SAVE_DATA_DIR']

        if input_path:
            import json
            context_file = Path(input_path)
            if not context_file.exists():
                raise FileNotFoundError(f"Context file not found: {context_file}")

            logger.info(f"Loading context from provided file: {context_file}")
            with open(context_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Detect which file type it is
            if "used_chunks" in data:  # query_answer.json structure
                context_chunks = [chunk["text"] for chunk in data["used_chunks"]]
                return {
                    "context": "\n\n".join(context_chunks),
                    "language": data.get("language", "English"),
                    "source": "query_answer",
                    "question": data.get("question"),
                    "saved_data_path": context_file.parent
                }
            elif "context" in data:  # context_language.json structure
                return {
                    "context": "\n\n".join(data["context"].values()),
                    "language": data.get("language", "English"),
                    "source": "context_language",
                    "question": None,
                    "saved_data_path": context_file.parent
                }
            else:
                raise ValueError("Unsupported JSON structure")

        # === Fallback: use default directory ===
        qa_path = saved_data_path / "query_answer.json"
        cl_path = saved_data_path / "context_language.json"

        logger.info(f"Looking for context files in: {saved_data_path}")
        if qa_path.exists():
            logger.info("Found query_answer.json")
            qa = load_json("query_answer.json")
            context_chunks = [chunk["text"] for chunk in qa["used_chunks"]]
            return {
                "context": "\n\n".join(context_chunks),
                "language": qa["language"],
                "source": "query_answer",
                "question": qa["question"],
                "saved_data_path": saved_data_path
            }
        elif cl_path.exists():
            logger.info("Found context_language.json")
            context_data = load_json("context_language.json")
            return {
                "context": "\n\n".join(context_data["context"].values()),
                "language": context_data.get("language", "English"),
                "source": "context_language",
                "question": None,
                "saved_data_path": saved_data_path
            }
        else:
            raise FileNotFoundError(
                f"No context files found in {saved_data_path}\n"
                "Please ensure you've either:\n"
                "1. Run the previous pipeline steps to generate context files, or\n"
                "2. Placed your own query_answer.json or context_language.json in the saved_data folder"
            )

    except Exception as e:
        logger.error(f"Error loading context: {str(e)}")
        return None
    

def generate_quiz():
    """CLI entrypoint for quiz generation"""
    import argparse
    parser = argparse.ArgumentParser(description="Generate a quiz from context JSON")
    parser.add_argument("--input", "-i", type=str, help="Path to JSON file (query_answer.json or context_language.json)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder to save the quiz")
    args = parser.parse_args()

    logger.info("\nStarting quiz generation...")
    context_data = get_context_from_file(args.input)

    if not context_data:
        logger.info("Error: Could not load context data")
        return None

    mcq_text, open_ended_text = generate_quiz_from_context(
        context_data["context"],
        context_data["language"]
    )

    quiz = {
        "source": context_data["source"],
        "mcq": parse_mcq(mcq_text) if mcq_text else [],
        "open_ended": parse_open_ended(open_ended_text) if open_ended_text else [],
        "language": context_data["language"],
        "timestamp": datetime.now().isoformat()
    }

    if context_data["question"]:
        quiz["based_on_question"] = context_data["question"]

    # Ensure output folder exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "generated_quiz.json"
    save_json(quiz, output_file)

    logger.info(f"\nSuccessfully generated quiz:")
    logger.info(f"- Source: {context_data['source']}")
    logger.info(f"- Multiple Choice: {len(quiz['mcq'])} questions")
    logger.info(f"- Open Ended: {len(quiz['open_ended'])} questions")
    logger.info(f"- Saved to: {output_file}")

    return quiz


if __name__ == "__main__":
    generate_quiz()
