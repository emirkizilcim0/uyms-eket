import os
import re
from google import genai
from .utils import get_config, save_json, load_json

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def generate_quiz_from_context(context, language):
    """Generate quiz questions from context"""
    config = get_config()

    # Create Gemini client (NEW SDK)
    client = genai.Client(api_key=config["API_KEY"])

    prompt = f"""
Based only on the following context, generate a quiz.
Context:
{context}

---

Return:
- Create the questions and explanations in the language {language}
- If the context is not related to any learnable material or is insufficient, do not generate any questions. Give a warning according to the reason!
- True options should not be referenced directly from the context.
- Questions shouldn't be similar with each other.
- 10 multiple-choice questions with options A-E, each marked (True) or (False), only one option is True. Make sure not all answers are the same letter.
- 5 open-ended questions, give explanations for the answers for each.

Format exactly like:

Multiple Choice:
1. Question?
   A) Option A (True XOR False)
   B) Option B (True XOR False)
   C) Option C (True XOR False)
   D) Option D (True XOR False)
   E) option E (True XOR False)
   Explanation: Explanation of the answer. Why is it true and other options are not true.

Open-Ended:
1. Question?
Explanation: Explanation of the answer. Why is it true and other options are not true.
2. Question?
Explanation: Explanation of the answer. Why is it true and other options are not true.    
"""

    try:
        response = client.models.generate_content(
            model=config["CHAT_MODEL"],
            contents=prompt
        )

        text = response.text.strip() if response.text else ""
        text = text.replace("Open-Ended:", "Open-Ended:")

        if "Open-Ended:" in text:
            parts = text.split("Open-Ended:")
            mcq_part = parts[0].replace("Multiple Choice:", "").strip()
            open_part = parts[1].strip()
        else:
            logger.warning("Could not find 'Open-Ended:' section. Returning raw text.")
            mcq_part = text.strip()
            open_part = ""

        return mcq_part, open_part

    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return "", ""


# In tutor/create.py
def parse_mcq(text):
    mcq_questions = []
    # More flexible pattern that handles various whitespace scenarios
    pattern = re.compile(
        r'(\d+)\.\s*(.*?)\s+'  # Question number and text
        r'A\)\s*(.*?)\s*\((True|False)\)\s*'  # Option A
        r'B\)\s*(.*?)\s*\((True|False)\)\s*'  # Option B
        r'C\)\s*(.*?)\s*\((True|False)\)\s*'  # Option C
        r'(?:D\)\s*(.*?)\s*\((True|False)\)\s*)?'  # Optional Option D
        r'(?:E\)\s*(.*?)\s*\((True|False)\)\s*)?'  # Optional Option E
        r'Explanation:\s*(.*?)(?=\n\d+\.|\Z)',  # Explanation
        re.DOTALL
    )
    
    matches = pattern.findall(text)
    for match in matches:
        question_num = match[0]
        question = {
            f'Q{question_num}': match[1].strip(),
            f'A{question_num}1': {'text': match[2].strip(), 'correct': match[3] == 'True'},
            f'A{question_num}2': {'text': match[4].strip(), 'correct': match[5] == 'True'},
            f'A{question_num}3': {'text': match[6].strip(), 'correct': match[7] == 'True'},
            f'E{question_num}': match[-1].strip()  # Explanation is always last
        }
        
        # Handle optional D and E options
        if len(match) > 8 and match[8]:
            question[f'A{question_num}4'] = {'text': match[8].strip(), 'correct': match[9] == 'True'}
        if len(match) > 10 and match[10]:
            question[f'A{question_num}5'] = {'text': match[10].strip(), 'correct': match[11] == 'True'}
            
        mcq_questions.append(question)
    
    return mcq_questions

def parse_open_ended(text):
    open_questions = []
    pattern = re.compile(
        r'(\d+)\.\s*(.*?)\s*Explanation:\s*(.*?)(?=\n\d+\.|\Z)',
        re.DOTALL
    )
    
    matches = pattern.findall(text)
    for match in matches:
        question_num = match[0]
        question = {
            f'Q{question_num}': match[1].strip(),
            f'E{question_num}': match[2].strip()
        }
        open_questions.append(question)
    
    return open_questions

def main():
    data = load_json("context_language.json")
    if not data:
        return

    context = data.get("context")
    language = data.get("language", "English")

    mcq_text, open_ended_text = generate_quiz_from_context(context, language)

    # Parse questions into structured format
    mcq_questions = parse_mcq(mcq_text)
    open_ended_questions = parse_open_ended(open_ended_text)

    # Save questions JSON
    save_json({
        "mcq": mcq_questions,
        "open_ended": open_ended_questions,
        "language": language
    }, "questions.json")

    logger.info("--- MULTIPLE CHOICE QUESTIONS ---")
    for question in mcq_questions:
        q_num = [k for k in question.keys() if k.startswith('Q')][0][1:]

        logger.info(f"{q_num}. {question[f'Q{q_num}']}")
        logger.info(f"   A) {question[f'A{q_num}1']['text']} ({'True' if question[f'A{q_num}1']['correct'] else 'False'})")
        logger.info(f"   B) {question[f'A{q_num}2']['text']} ({'True' if question[f'A{q_num}2']['correct'] else 'False'})")
        logger.info(f"   C) {question[f'A{q_num}3']['text']} ({'True' if question[f'A{q_num}3']['correct'] else 'False'})")
        logger.info(f"   D) {question[f'A{q_num}4']['text']} ({'True' if question[f'A{q_num}4']['correct'] else 'False'})")
        logger.info(f"   E) {question[f'A{q_num}5']['text']} ({'True' if question[f'A{q_num}5']['correct'] else 'False'})")
        logger.info(f"   Explanation: {question[f'E{q_num}']}\n")

    
    logger.info("--- OPEN ENDED QUESTIONS ---")
    for question in open_ended_questions:
        q_num = [k for k in question.keys() if k.startswith('Q')][0][1:]

        logger.info(f"{q_num}. {question[f'Q{q_num}']}")
        logger.info(f"   Explanation: {question[f'E{q_num}']}\n")


if __name__ == "__main__":
    main()