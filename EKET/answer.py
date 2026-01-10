import json
import re
from google import genai
from .utils import get_config, save_json, load_json


def evaluate_answers(questions_data):
    """Evaluate user answers against correct answers"""

    config = get_config()
    client = genai.Client(api_key=config["API_KEY"])

    language = questions_data.get("language", "English")

    results = {
        "language": language,
        "score": {
            "total_mcq": len(questions_data["mcq"]),
            "correct_mcq": 0,
            "total_open_ended": len(questions_data["open_ended"]),
            "correct_open_ended": 0
        },
        "mcq_results": [],
        "open_ended_results": []
    }

    # ------------------
    # MCQ Evaluation
    # ------------------
    for mcq in questions_data["mcq"]:
        q_key = [k for k in mcq if k.startswith("Q")][0]
        q_num = q_key[1:]

        # Find correct option
        correct_answer = None
        for i in range(1, 6):
            if mcq.get(f"A{q_num}{i}", {}).get("correct", False):
                correct_answer = chr(64 + i)  # A, B, C...
                break

        # TODO: Replace with real user input
        user_answer = "B"

        is_correct = user_answer == correct_answer
        if is_correct:
            results["score"]["correct_mcq"] += 1

        results["mcq_results"].append({
            "question_number": q_num,
            "question": mcq[q_key],
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": "Correct" if is_correct else "Incorrect",
            "explanation": mcq.get(f"E{q_num}", ""),
            "options": {
                "A": mcq.get(f"A{q_num}1", {}).get("text", ""),
                "B": mcq.get(f"A{q_num}2", {}).get("text", ""),
                "C": mcq.get(f"A{q_num}3", {}).get("text", ""),
                "D": mcq.get(f"A{q_num}4", {}).get("text", ""),
                "E": mcq.get(f"A{q_num}5", {}).get("text", "")
            }
        })

    # ------------------
    # Open-Ended Evaluation
    # ------------------
    for oeq in questions_data["open_ended"]:
        q_key = [k for k in oeq if k.startswith("Q")][0]
        q_num = q_key[1:]

        # TODO: Replace with real user input
        user_answer = "Menkul kıymetlerin gelişimi sebep olmuştur."

        try:
            evaluation_prompt = f"""
You are an evaluator. Only reply with valid JSON.

Evaluate the following open-ended question in {language}:

Question: {oeq[q_key]}
Model Answer: {oeq[f"E{q_num}"]}
User Answer: {user_answer}

Return a JSON object with this format:
{{
  "rating": "Correct" | "Partially Correct" | "Incorrect",
  "explanation": "Short evaluation in {language}"
}}

ONLY return the JSON. No other text.
"""

            response = client.models.generate_content(
                model=config["CHAT_MODEL"],
                contents=evaluation_prompt
            )

            response_text = response.text.strip() if response.text else ""

            # Strip markdown fences if present
            if response_text.startswith("```"):
                match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```",
                    response_text,
                    re.DOTALL
                )
                if match:
                    response_text = match.group(1).strip()
                else:
                    raise ValueError("Invalid markdown-wrapped JSON")

            evaluation = json.loads(response_text)

            if evaluation.get("rating") == "Correct":
                results["score"]["correct_open_ended"] += 1

            result = {
                "question_number": q_num,
                "question": oeq[q_key],
                "model_answer": oeq[f"E{q_num}"],
                "user_answer": user_answer,
                "rating": evaluation.get("rating"),
                "explanation": evaluation.get("explanation")
            }

        except Exception as e:
            result = {
                "question_number": q_num,
                "question": oeq[q_key],
                "model_answer": oeq.get(f"E{q_num}", ""),
                "user_answer": user_answer,
                "rating": "Evaluation Error",
                "explanation": f"Could not evaluate: {str(e)}"
            }

        results["open_ended_results"].append(result)

    return results


def answer_checker(questions_json_path, output_dir="saved_data"):
    questions_data = load_json(questions_json_path)
    results = evaluate_answers(questions_data)
    save_json(results, "answer_check_results.json", output_dir)
    return results
