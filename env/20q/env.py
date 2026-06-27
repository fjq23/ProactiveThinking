from openai import OpenAI
import re, json
import requests
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import completion_token_count
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

DEBUG=False

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

patient_prompt = """
You are the oracle in a 20 Questions game. Fully immerse yourself in this role and continue the dialogue based on the history provided.

Game Details:
{patient_records}

Dialogue History:
{history}

Operational Guidelines:
1. Opening: If there is no dialogue history, give the brief clue from the game details and invite a single yes-or-no question.
2. Information Pacing: Reveal information gradually. Do not provide the hidden entity outright.
3. Clarification Triggers:
* If the player's question is non-specific, ask them to ask one clear yes-or-no question.
* If the player asks more than one thing in a single response, stop them and ask for one question at a time.
4. Knowledge Boundaries: Answer honestly using the hidden target entity, but do not reveal it directly unless it is an exact final guess.
5. Language & Tone:
* Use simple, natural language and keep the tone consistent with the clue.
* If the player uses words that are too vague or too broad, ask them to narrow it down.
6. Output Constraints:
* Content: Output the spoken response ONLY. Do not include physical actions, non-verbal cues, or descriptions.
* Length: 1-2 concise sentences (Strictly under 20 words).
* Format: Wrap the spoken response in `<response>` and `</response>` tags.

Please output your spoken response to the player only.
""".strip()

# only the latest question is inputted
oracle_prompt = """
## **Task**
You are a telepathic entity playing a mind reading game. The user is trying to guess what entity you are thinking of by asking yes/no questions. You should respond honestly based on the target entity you're thinking of.

## **Instructions**
1. You are thinking of a specific entity (person, object, concept, etc.) - this is the "target_entity" provided to you.
2. The user will ask questions to narrow down what you're thinking of.
3. Answer "Yes" if the question is true about your target entity.
4. Answer "No" if the question is false about your target entity.
5. Answer "Maybe" only if the question is ambiguous or you genuinely cannot determine a clear yes/no answer.
6. Be helpful and honest - the goal is for them to eventually guess correctly through good questions.

### Target Entity You Are Thinking Of
{patient_records}

### User's Current Question
{history}

Respond with exactly one word: "Yes", "No", or "Maybe".
""".strip()


class Patient:
    def __init__(self, args, patient_records) -> None:
        self.client = OpenAI(api_key=args.patient_api_key, base_url=args.patient_api_base)
        if args.patient_model is None:
            self.model_name = self.client.models.list().data[0].id
        else:
            self.model_name = args.patient_model
        self.patient_records = patient_records
        parsed_records = json.loads(patient_records)
        self.opening_prompt = parsed_records["prompt"]

    def extract_answer(self, text):
        match = re.search(r'<response>(.*?)</response>', text, re.S)
        if match:
            result = match.group(1).strip()
            return result
        else:
            return None

    # def get_history(self, messages):
    #     return "\n".join(f"{turn['role']}: {self.extract_answer(turn['content'])}" for turn in messages)

    # def get_response(self, messages):
    #     messages = [{"role": "user", "content": patient_prompt.format(patient_records=self.patient_records, history=self.get_history(messages))}]
    #     response = completion_with_backoff(client=self.client, model=self.model_name, messages=messages, temperature=0.7, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    #     return response.choices[0].message.content, completion_token_count(response)
    
    # 用oracle prompt
    def get_history(self, messages):
        role_names = {"patient": "oracle", "assistant": "guesser"}
        return "\n".join(
            f"{role_names.get(turn['role'], turn['role'])}: {self.extract_answer(turn['content'])}"
            for turn in messages[-1:]
        )

    def get_response(self, messages):
        if not messages:
            return self.opening_prompt, 0
        messages = [{"role": "user", "content": oracle_prompt.format(patient_records=self.patient_records, history=self.get_history(messages))}]
        response = completion_with_backoff(client=self.client, model=self.model_name, messages=messages, temperature=0.7, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return response.choices[0].message.content, completion_token_count(response)

judge_prompt = """
You are a verification judge for a 20 Questions game.
Determine if the "Player's Final Guess" extracted from the dialogue refers unambiguously to the "Correct Answer."

Correct Answer: {answer}

Player Dialogue: {dialogue}

Evaluation Criteria:
1. Synonyms, common abbreviations, or clearly equivalent names that refer directly and exclusively to the correct entity should be marked as Yes.
2. If the player gives a category, a partial match, or an uncertain/differential guess, mark as No.

Respond with exactly one word: "Yes" or "No".
""".strip()

class Judge:
    def __init__(self, args, answer) -> None:
        self.client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_api_base)
        if args.judge_model is None:
            self.model_name = self.client.models.list().data[0].id
        else:
            self.model_name = args.judge_model
        self.answer = answer

    def extract_answer(self, text):
        match = re.search(r'<answer>(.*?)</answer>', text, re.S)
        if match:
            result = match.group(1).strip()
            return result
        else:
            return None

    def get_response(self, messages):
        prediction = self.extract_answer(messages[-1]["content"])
        if prediction is None:
            return None

        messages = [{"role": "user", "content": judge_prompt.format(answer=self.answer, dialogue=prediction)}]

        return completion_with_backoff(client=self.client, model=self.model_name, messages=messages, temperature=0.2, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).choices[0].message.content
