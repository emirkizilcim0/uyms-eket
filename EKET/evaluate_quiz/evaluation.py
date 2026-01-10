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

from EKET.answer import evaluate_answers
from EKET.utils import load_json, save_json

import logging
import sys
import argparse
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def evaluate_quiz(input_file: str, output_dir: str):
    """Evaluate user's quiz answers"""
    try:
        quiz = load_json(input_file)
        results = evaluate_answers(quiz)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "evaluation_results.json")
        save_json(results, out_path)

        logger.info("\n=== EVALUATION RESULTS ===")
        logger.info(f"MCQ Score: {results['score']['correct_mcq']}/{results['score']['total_mcq']}")
        logger.info(f"Open-Ended Score: {results['score']['correct_open_ended']}/{results['score']['total_open_ended']}")
        logger.info(f"Results saved to: {out_path}")

        return results
    except FileNotFoundError:
        logger.error(f"No quiz file found at {input_file}. Run generate_quiz first.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a quiz JSON file")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="generated_quiz.json",
        help="Path to the quiz JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output folder where evaluation results will be stored"
    )
    args = parser.parse_args()

    evaluate_quiz(args.input, args.output)


if __name__ == "__main__":
    main()