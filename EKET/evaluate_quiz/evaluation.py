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