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
You are the patient (or their companion) described in the records below. Fully immerse yourself in this persona, setting aside your identity as an AI. Continue the dialogue based on the history provided.

Patient Records:
{patient_records}

Dialogue History:
{history}

Operational Guidelines:
1. Opening: If there is no dialogue history, the conversation has just begun. Briefly greet the doctor and describe your most prominent symptoms.
2. Information Pacing: Reveal details or symptoms gradually. Do not provide a full medical history at once; only answer what is specifically asked.
3. Clarification Triggers:
* If the doctor's question is non-specific (e.g., "Tell me about your pain" or "What did the scan say?"), do not answer. Ask them exactly what they want to know.
* If the doctor asks more than one thing in a single response—even if they are in the same sentence (e.g., "Where is the pain and when did it start?")—you must stop them. Respond by saying you can only handle one question at a time or that they are moving too fast.
4. Knowledge Boundaries: Do not reveal your diagnosis or ED disposition, as a real patient would not have this information yet.
5. Language & Tone:
* Use informal, everyday language and a tone that matches the patient's background.
* If the doctor uses words exceeding the patient's proficiency, ask for rephrasing or simpler terms.
6. Output Constraints:
* Content: Output the spoken response ONLY. Do not include physical actions, non-verbal cues, or descriptions.
* Length: 1-2 concise sentences (Strictly under 20 words).
* Consistency: Ensure all responses align with the patient's profile and previous dialogue history.

Please output your spoken response to the doctor only.
""".strip()

class Patient:
    def __init__(self, args, patient_records) -> None:
        self.client = OpenAI(api_key=args.patient_api_key, base_url=args.patient_api_base)
        if args.patient_model is None:
            self.model_name = self.client.models.list().data[0].id
        else:
            self.model_name = args.patient_model
        self.patient_records = patient_records

    def extract_answer(self, text):
        match = re.search(r'<response>(.*?)</response>', text, re.S)
        if match:
            result = match.group(1).strip()
            return result
        else:
            return None

    def get_history(self, messages):
        return "\n".join(f"{turn['role']}: {self.extract_answer(turn['content'])}" for turn in messages)

    def get_response(self, messages):
        messages = [{"role": "user", "content": patient_prompt.format(patient_records=self.patient_records, history=self.get_history(messages))}]
        response = completion_with_backoff(
            client=self.client,
            model=self.model_name,
            messages=messages,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        tokens = completion_token_count(response)
        return content, tokens

judge_prompt = """
You are a medical terminology expert specializing in diagnostic verification.
Determine if the "Doctor's Diagnosis" extracted from the dialogue refers unambiguously to the "Correct Diagnosis."

Correct Diagnosis: {answer}

Doctor Dialogue: {dialogue}

Evaluation Criteria:
1. Synonyms, common medical abbreviations, or layperson terms that map directly and exclusively to the correct diagnosis (e.g., "Hypertension" vs. "High blood pressure") should be marked as Yes.
2. If the doctor suggests a differential diagnosis (e.g., "It could be X or Y"), or provides only a symptom (e.g., "Cough") instead of the specific disease, mark as No.

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
