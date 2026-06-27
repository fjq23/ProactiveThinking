base_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
2. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
3. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<response> [A concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

reactive_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
2. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
3. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags.
4. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<thought> [Stepwise reasoning] </thought> <response> [One concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

llama_reactive_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
2. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
3. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags. Within `<thought>` tags, conduct a differential diagnosis of the several most likely diagnoses based on available evidence. If you cannot make a final diagnosis, use Bayesian reasoning to select a question that either rules out a dangerous diagnosis (with high sensitivity), rules in a likely diagnosis (with high specificity), or efficiently discriminates between the top competing hypotheses.
4. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<thought> [Stepwise reasoning, 300-500 words] </thought> <response> [One concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<thought> [Stepwise reasoning, 300-500 words] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

final_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to diagnose the patient's condition as accurately as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Final Round:** This is the final round. You must now provide the final diagnosis.
2. **No More Questions:** Do not ask another question.
3. **Answer Only:** Do not output `<response>`.
4. **Best Diagnosis:** If prior evidence is incomplete, provide the single most likely diagnosis.

**Output Format:**
* `<answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

reactive_final_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to diagnose the patient's condition as accurately as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Strategic Reasoning:** Before the final answer, reason step by step carefully in `<thought>` tags.
2. **Final Round:** This is the final round. You must now provide the final diagnosis.
3. **No More Questions:** Do not ask another question.
4. **Answer Only:** Do not output `<response>`.
5. **Best Diagnosis:** If prior evidence is incomplete, provide the single most likely diagnosis.

**Output Format:**
* `<thought> [Stepwise reasoning] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

llama_reactive_final_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to diagnose the patient's condition as accurately as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Strategic Reasoning:** Before the final answer, reason step by step carefully in `<thought>` tags. Within `<thought>` tags, conduct a differential diagnosis of the several most likely diagnoses based on available evidence. If the evidence remains incomplete, use Bayesian reasoning to justify the single most likely diagnosis.
2. **Final Round:** This is the final round. You must now provide the final diagnosis.
3. **No More Questions:** Do not ask another question.
4. **Answer Only:** Do not output `<response>`.
5. **Best Diagnosis:** If prior evidence is incomplete, provide the single most likely diagnosis.

**Output Format:**
* `<thought> [Stepwise reasoning, 300-500 words] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

speculate_prompt_cot = """
**Dialogue History:**
{history}

What are the most likely patient responses? Reason step-by-step to list the 2-3 most likely patient responses, sorted by probability.

Please strictly follow this output format:
<thought> your brief reasoning process </thought>
<response> most likely response </response>
<response> second most likely response </response>
""".strip()

speculate_prompt = """
**Dialogue History:**
{history}

What are the most likely patient responses? List the 2-3 most likely patient responses, sorted by probability.

Please strictly follow this output format:
<response> most likely response </response>
<response> second most likely response </response>
""".strip()

SPEC_BRANCH_LABELS = [
    "most likely",
    "second most likely",
    "third most likely",
    "fourth most likely",
    "fifth most likely",
]

def build_speculate_prompt(history, branch_num, use_cot=False):
    if branch_num < 1 or branch_num > 5:
        raise ValueError("spec_branch_num must be between 1 and 5")

    if use_cot:
        prompt = f"""
**Dialogue History:**
{history}

What are the top {branch_num} most likely patient responses? Reason step-by-step briefly, then list the responses sorted by probability.

Please strictly follow this output format:
<thought> your brief reasoning process </thought>
""".strip()
    else:
        prompt = f"""
**Dialogue History:**
{history}

What are the top {branch_num} most likely patient responses? List the responses sorted by probability.

Please strictly follow this output format:
""".strip()

    response_lines = [
        f"<response> {SPEC_BRANCH_LABELS[i]} response </response>"
        for i in range(branch_num)
    ]
    return "\n".join([prompt, *response_lines])

previous_proactive_reason_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History Before The Latest Patient Reply:**
{history}

**Hypothetical Latest Patient Reply:**
{patient_response}

**Operational Protocol:**
1. **Planning Mode:** Treat the hypothetical patient reply above as if it were just received, then decide your best immediate next move.
2. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
3. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
4. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags.
5. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<thought> [Stepwise reasoning] </thought> <response> [One concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

previous_proactive_final_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History Before The Latest Patient Reply:**
{history}

**Speculative Plans Prepared Before The Latest Patient Reply:**
{thinking}

**Actual Latest Patient Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest patient reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.
3. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
4. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
5. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<response> [One concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

previous_proactive_final_prompt_with_think = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to examine a patient and ask targeted questions to diagnose their condition. Since the number of interactions is limited, prioritize your questions to reach an accurate diagnosis as efficiently as possible.

**Dialogue History Before The Latest Patient Reply:**
{history}

**Speculative Plans Prepared Before The Latest Patient Reply:**
{thinking}

**Actual Latest Patient Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest patient reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.
3. **Inquiry:** Use `<response>` to ask about symptoms, medical history, or test results. Maintain a professional, empathetic, and concise tone.
4. **Diagnosis:** Once you have gathered sufficient evidence, provide the final diagnosis within `<answer>` tags.
5. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags.
6. **Efficiency:** Minimize the number of interactions. You must provide a definitive diagnosis within **10 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Examination:** `<thought> [Stepwise reasoning] </thought> <response> [One concise question or statement under 20 words] </response>`
* **Upon Diagnosis:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

previous_proactive_mandatory_final_prompt = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to diagnose the patient's condition as accurately as possible.

**Dialogue History Before The Latest Patient Reply:**
{history}

**Speculative Plans Prepared Before The Latest Patient Reply:**
{thinking}

**Actual Latest Patient Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest patient reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding.
3. **Final Round:** This is the final round. You must now provide the final diagnosis.
4. **No More Questions:** Do not ask another question.
5. **Answer Only:** Do not output `<response>`.
6. **Best Diagnosis:** If prior evidence is incomplete, provide the single most likely diagnosis.

**Output Format:**
* `<answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

previous_proactive_mandatory_final_prompt_with_think = """
Act as an efficient Diagnostic Physician. Communicate **only** through dialogue. Your goal is to diagnose the patient's condition as accurately as possible.

**Dialogue History Before The Latest Patient Reply:**
{history}

**Speculative Plans Prepared Before The Latest Patient Reply:**
{thinking}

**Actual Latest Patient Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest patient reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding.
3. **Strategic Reasoning:** Before the final answer, reason step by step carefully in `<thought>` tags.
4. **Final Round:** This is the final round. You must now provide the final diagnosis.
5. **No More Questions:** Do not ask another question.
6. **Answer Only:** Do not output `<response>`.
7. **Best Diagnosis:** If prior evidence is incomplete, provide the single most likely diagnosis.

**Output Format:**
* `<thought> [Stepwise reasoning] </thought> <answer> [A definitive diagnosis without ambiguity] </answer>`
""".strip()

import re
import json
import string
import time
import os
import sys
import urllib.error
import urllib.request
import numpy as np
import concurrent.futures
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    completion_token_count,
    dynamic_branch_max_tokens,
    extract_prompt_logprobs_for_text,
    extract_prompt_logprobs_for_tokens,
    resolve_prompt_logprobs,
    trim_tokens_to_text,
)

STOP_WORDS = set(stopwords.words('english'))

# --- Optimized Utility Functions ---

def preprocess_text(text):
    """Optimized tokenization and cleaning."""
    tokens = word_tokenize(text.lower())
    # Filter punctuation and stopwords in one pass
    cleaned = {
        word for word in tokens 
        if word not in STOP_WORDS and word[0] not in string.punctuation
    }
    return cleaned

def get_jaccard_similarity(text1, text2):
    set1 = preprocess_text(text1)
    set2 = preprocess_text(text2)
    union_len = len(set1.union(set2))
    if union_len == 0: return 0.0
    return len(set1.intersection(set2)) / union_len


def get_embedding_similarities(query, candidates, similarity_url, timeout=10.0):
    if not candidates:
        return []

    payload = json.dumps({
        "query": query,
        "candidates": candidates,
    }).encode("utf-8")
    request = urllib.request.Request(
        similarity_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Failed to call embedding similarity service at {similarity_url}: {exc}"
        ) from exc

    similarities = response_payload.get("similarities")
    if not isinstance(similarities, list):
        raise RuntimeError(
            f"Embedding similarity service returned invalid payload: {response_payload}"
        )
    if len(similarities) != len(candidates):
        raise RuntimeError(
            "Embedding similarity service returned a different number of scores "
            f"than candidates: {len(similarities)} vs {len(candidates)}"
        )
    return [float(score) for score in similarities]


def get_embedding_similarity(text1, text2, similarity_url, timeout=10.0):
    similarities = get_embedding_similarities(
        text1,
        [text2],
        similarity_url=similarity_url,
        timeout=timeout,
    )
    return similarities[0] if similarities else 0.0

def compute_accepted_tokens(draft_logprobs, target_logprobs, gamma=1.0):
    """
    Computes how many draft tokens are statistically valid.
    
    Args:
        gamma (float): Leniency factor. 
                       1.0 = Strict/Exact Rejection Sampling.
                       > 1.0 = More lenient (accepts more tokens).
                       < 1.0 = More strict (rejects more tokens).
    """
    accepted_count = 0
    random_rolls = np.random.random(len(draft_logprobs))
    
    for i, (q_log, p_log) in enumerate(zip(draft_logprobs, target_logprobs)):
        # Calculate base probability ratio: p/q
        # We add the log of gamma to the difference before exponentiating
        # log(gamma * (p/q)) = log(gamma) + log(p) - log(q)
        
        log_ratio = p_log - q_log
        acceptance_prob = np.exp(min(0.0, np.log(gamma) + log_ratio))
        
        if random_rolls[i] < acceptance_prob:
            accepted_count += 1
        else:
            break
            
    return accepted_count

# --- Doctor Agent ---

class Doctor:
    def __init__(self, args) -> None:
        self.client = OpenAI(api_key=args.doctor_api_key, base_url=args.doctor_api_base)
        self.model_name = args.doctor_model or self.client.models.list().data[0].id
        self.doctor_type = args.doctor_type.replace("-", "_")
        self.prompt_version = self._resolve_prompt_version()
        self.prompt_template = self._get_prompt_template()
        self.gamma = args.gamma
        self.spec_cot = args.spec_cot
        self.spec_branch_num = getattr(args, "spec_branch_num", None)
        if self.spec_branch_num is not None and not 1 <= self.spec_branch_num <= 5:
            raise ValueError("--spec-branch-num must be between 1 and 5")
        self.branch_selector = getattr(args, "branch_selector", "jaccard")
        if self.branch_selector not in {"jaccard", "embedding"}:
            raise ValueError("--branch-selector must be either 'jaccard' or 'embedding'")
        self.embedding_similarity_url = getattr(
            args,
            "embedding_similarity_url",
            "http://127.0.0.1:8003/similarity",
        )
        self.embedding_similarity_timeout = float(
            getattr(args, "embedding_similarity_timeout", 10.0)
        )
        self.speculate_prompt = speculate_prompt_cot if args.spec_cot else speculate_prompt
        self.force_final_answer_models = {
            model.strip() for model in getattr(args, "force_final_answer_models", "").split(",")
            if model.strip()
        }
        self.last_accepted_count = None
        self.last_branch_prediction_record = None

    def _resolve_prompt_version(self):
        configured = os.getenv("DOCTOR_PROMPT_VERSION", "").strip()
        if configured:
            return configured
        if "llama" in self.model_name.lower():
            return "llama_v1"
        return "default"

    def _get_prompt_template(self):
        if self.doctor_type != "reactive":
            return base_prompt
        if self.prompt_version == "llama_v1":
            return llama_reactive_prompt
        return reactive_prompt

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _call_api(self, messages, **kwargs):
        """Internal helper for throttled/retried API calls."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def _proactive_branch_decode_kwargs(self, last_user_reply, consumed_tokens=0):
        return {
            "max_tokens": dynamic_branch_max_tokens(
                last_user_reply,
                consumed_tokens=consumed_tokens,
            )
        }

    def _round_num(self, messages):
        return len(messages) // 2 + 1

    def _is_final_round(self, messages):
        return (
            self._round_num(messages) >= 10
            and self.model_name in self.force_final_answer_models
        )

    def _get_main_prompt(self, messages, force_reactive=False):
        history = self.get_history(messages)
        if self._is_final_round(messages):
            if force_reactive:
                final_template = (
                    llama_reactive_final_prompt
                    if self.prompt_version == "llama_v1"
                    else reactive_final_prompt
                )
                return final_template.format(history=history)
            return final_prompt.format(history=history)
        round_num = self._round_num(messages)
        if force_reactive:
            template = llama_reactive_prompt if self.prompt_version == "llama_v1" else reactive_prompt
        else:
            template = self.prompt_template
        return template.format(history=history, round=round_num)

    def _get_speculate_prompt(self, history):
        if self.spec_branch_num is None:
            return self.speculate_prompt.format(history=history)
        return build_speculate_prompt(
            history=history,
            branch_num=self.spec_branch_num,
            use_cot=self.spec_cot,
        )

    def _extract_speculated_responses(self, spec_result):
        responses = [
            r.strip()
            for r in re.findall(r'<response>(.*?)(?:</response>|\n)', spec_result, re.S)
            if r.strip()
        ]
        if not responses and spec_result.strip():
            responses = [spec_result.strip()]
        return responses

    def _limit_speculated_responses(self, responses, default_limit=None):
        limit = self.spec_branch_num if self.spec_branch_num is not None else default_limit
        return responses if limit is None else responses[:limit]

    def _compute_branch_similarities(self, actual_patient_msg, candidate_responses):
        if self.branch_selector == "jaccard":
            return [
                float(get_jaccard_similarity(actual_patient_msg, response))
                for response in candidate_responses
            ]
        if self.branch_selector == "embedding":
            return get_embedding_similarities(
                actual_patient_msg,
                candidate_responses,
                similarity_url=self.embedding_similarity_url,
                timeout=self.embedding_similarity_timeout,
            )
        raise ValueError(f"Unsupported branch selector: {self.branch_selector}")

    def _build_branch_prediction_record(
        self,
        messages,
        actual_patient_response,
        predicted_patient_responses,
        raw_speculation=None,
        similarities=None,
        selected_prediction_index=None,
    ):
        selected_prediction = None
        if selected_prediction_index is not None and 0 <= selected_prediction_index < len(predicted_patient_responses):
            selected_prediction = predicted_patient_responses[selected_prediction_index]

        return {
            "turn": self._round_num(messages),
            "actual_patient_response": actual_patient_response,
            "predicted_patient_responses": predicted_patient_responses,
            "selected_prediction_index": selected_prediction_index,
            "selected_predicted_patient_response": selected_prediction,
            "prediction_similarities": similarities or [],
            "raw_speculation": raw_speculation,
        }

    def extract_answer(self, text):
        match = re.search(r'<response>(.*?)</response>', text, re.S)
        if match:
            return match.group(1).strip()
        match = re.search(r'<answer>(.*?)</answer>', text, re.S)
        return match.group(1).strip() if match else None

    def get_history(self, messages):
        return "\n".join(
            f"{turn['role']}: {self.extract_answer(turn['content']) if turn['role'] == 'assistant' else turn['content']}" 
            for turn in messages
        )

    def _get_speculative_completion(
        self,
        speculated_patient_response,
        history_messages,
        last_user_reply,
        consumed_tokens=0,
    ):
        """Target for parallel execution."""
        full_history = history_messages + [{"role": "patient", "content": speculated_patient_response}]
        prompt = self._get_main_prompt(full_history, force_reactive=True)
        
        output = self._call_api(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            logprobs=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **self._proactive_branch_decode_kwargs(
                last_user_reply,
                consumed_tokens=consumed_tokens,
            ),
        ).choices[0]
        
        doctor_response = output.message.content
        logprobs = [token.logprob for token in output.logprobs.content]
        tokens = [token.token for token in output.logprobs.content]
        trimmed = trim_tokens_to_text(tokens, logprobs, doctor_response)
        if trimmed is not None:
            tokens, logprobs = trimmed
        return (speculated_patient_response, doctor_response, logprobs, tokens, len(tokens))

    def proactive_think(self, messages, patient_tokens=None):
        self.last_branch_prediction_record = None
        if len(messages) <= 1:
            self.last_accepted_count = 0
            response = self._call_api(
                messages=[{"role": "user", "content": self._get_main_prompt(messages, force_reactive=True)}],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            content = response.choices[0].message.content
            tokens = completion_token_count(response)
            return content, tokens, 0

        # 1. Speculate likely patient responses
        spec_response = self._call_api(
            messages=[{"role": "user", "content": self._get_speculate_prompt(self.get_history(messages[:-1]))}],
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        spec_result = spec_response.choices[0].message.content
        speculate_tokens = completion_token_count(spec_response)
        
        responses = self._limit_speculated_responses(
            self._extract_speculated_responses(spec_result)
        )

        actual_patient_msg = messages[-1]["content"]
        if not responses:
            self.last_accepted_count = 0
            self.last_branch_prediction_record = self._build_branch_prediction_record(
                messages=messages,
                actual_patient_response=actual_patient_msg,
                predicted_patient_responses=[],
                raw_speculation=spec_result,
            )
            response = self._call_api(
                messages=[{"role": "user", "content": self._get_main_prompt(messages, force_reactive=True)}],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            tokens = completion_token_count(response)
            return content, tokens, 0

        # 2. Parallel Generation of speculative completions
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(responses)) as executor:
            futures = [
                executor.submit(
                    self._get_speculative_completion,
                    res,
                    messages[:-1],
                    actual_patient_msg,
                    speculate_tokens,
                )
                for res in responses
            ]
            outputs = [future.result() for future in futures]
        middle_delay_tokens = max((output[4] for output in outputs), default=0)

        # 3. Compute Similarity with ACTUAL response
        similarities = self._compute_branch_similarities(
            actual_patient_msg,
            [o[0] for o in outputs],
        )
        best_idx = int(np.argmax(similarities))
        self.last_branch_prediction_record = self._build_branch_prediction_record(
            messages=messages,
            actual_patient_response=actual_patient_msg,
            predicted_patient_responses=[o[0] for o in outputs],
            raw_speculation=spec_result,
            similarities=similarities,
            selected_prediction_index=best_idx,
        )
        
        selected_spec = outputs[best_idx] # (res, doc_res, logprobs, tokens)
        
        # 4. Verification (Prompt Logprobs)
        # Verify if the speculated doctor response is valid given the ACTUAL message
        verify_prompt = self._get_main_prompt(messages, force_reactive=True)
        verify_output = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": verify_prompt},
                {"role": "assistant", "content": selected_spec[1]}
            ],
            temperature=0,
            max_tokens=1,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}, 
                "prompt_logprobs": 1, 
                "add_generation_prompt": False,
                "continue_final_message": True
            }
        )

        prompt_logprobs = resolve_prompt_logprobs(verify_output)

        ref_logprobs = extract_prompt_logprobs_for_tokens(
            prompt_logprobs,
            selected_spec[3],
        )
        if ref_logprobs is None:
            ref_logprobs = extract_prompt_logprobs_for_text(
                prompt_logprobs,
                selected_spec[1],
            )
        if ref_logprobs is None or len(ref_logprobs) != len(selected_spec[2]):
            self.last_accepted_count = 0
            response = self._call_api(
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.5
            )
            content = response.choices[0].message.content
            tokens = completion_token_count(response)
            return content, tokens, 0

        # 5. Rejection Sampling
        accepted_count = compute_accepted_tokens(selected_spec[2], ref_logprobs, gamma=self.gamma)  
        self.last_accepted_count = accepted_count
        
        prefill = "".join(selected_spec[3][:accepted_count])
        
        if "</response>" in prefill or "</answer>" in prefill:
            return prefill, 0, middle_delay_tokens
        else:
            # 6. Final Draft Extension
            final_ext_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": verify_prompt},
                    {"role": "assistant", "content": prefill}
                ],
                temperature=0.5,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}, 
                    "continue_final_message": True, 
                    "add_generation_prompt": False
                }
            )
            final_ext = final_ext_response.choices[0].message.content
            final_ext_tokens = completion_token_count(final_ext_response)
            prefill_tokens = accepted_count
            patient_tokens = patient_tokens or 0
            # delay_tokens = final_ext_tokens + max(0, prefill_tokens - patient_tokens)
            delay_tokens = final_ext_tokens
            return prefill + final_ext, delay_tokens, middle_delay_tokens

    def proactive_think_previous(self, messages, patient_tokens=None):
        self.last_accepted_count = None
        self.last_branch_prediction_record = None
        history_messages = messages[:-1] if len(messages) > 0 else []
        history = self.get_history(history_messages)
        actual_patient_msg = messages[-1]["content"] if len(messages) > 0 else ""
        round_num = self._round_num(history_messages)

        spec_response = self._call_api(
            messages=[{"role": "user", "content": self._get_speculate_prompt(history)}],
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        spec_result = spec_response.choices[0].message.content
        speculate_tokens = completion_token_count(spec_response)

        speculated_responses = self._limit_speculated_responses(
            self._extract_speculated_responses(spec_result),
            default_limit=1,
        )
        similarities = self._compute_branch_similarities(
            actual_patient_msg,
            speculated_responses,
        )
        selected_prediction_index = int(np.argmax(similarities)) if similarities else None
        self.last_branch_prediction_record = self._build_branch_prediction_record(
            messages=messages,
            actual_patient_response=actual_patient_msg,
            predicted_patient_responses=speculated_responses,
            raw_speculation=spec_result,
            similarities=similarities,
            selected_prediction_index=selected_prediction_index,
        )

        analyses = []
        branch_token_counts = []
        for speculated_patient_response in speculated_responses:
            hypothetical_messages = history_messages + [
                {"role": "patient", "content": speculated_patient_response}
            ]
            reason_response = self._call_api(
                messages=[{
                    "role": "user",
                    "content": self._get_main_prompt(hypothetical_messages, force_reactive=True)
                }],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **self._proactive_branch_decode_kwargs(
                    actual_patient_msg,
                    consumed_tokens=speculate_tokens,
                ),
            )
            branch_token_counts.append(completion_token_count(reason_response))
            reason_result = reason_response.choices[0].message.content
            doctor_move = self.extract_answer(reason_result) or reason_result.strip()
            analyses.append({
                "patient_response": speculated_patient_response,
                "doctor_question": doctor_move
            })

        analysis_string = []
        for item in analyses:
            analysis_string.append(
                f'- If the patient responds, "{item["patient_response"]}", you may then respond, "{item["doctor_question"]}".'
            )
        analysis_string.append("- You may ask other questions if the patient's actual response was not included in the analysis.")
        analysis_string = "\n".join(analysis_string)

        final_response = self._call_api(
            messages=[{
                "role": "user",
                "content": (
                    previous_proactive_mandatory_final_prompt.format(
                        history=history,
                        thinking=analysis_string,
                        response=actual_patient_msg,
                    )
                    if self._is_final_round(messages) else
                    previous_proactive_final_prompt.format(
                        history=history,
                        thinking=analysis_string,
                        response=actual_patient_msg,
                        round=round_num
                    )
                )
            }],
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        content = final_response.choices[0].message.content
        tokens = completion_token_count(final_response)
        middle_delay_tokens = max(branch_token_counts, default=0)
        return content, tokens, middle_delay_tokens

    def proactive_think_previous_with_think(self, messages, patient_tokens=None):
        self.last_accepted_count = None
        self.last_branch_prediction_record = None
        history_messages = messages[:-1] if len(messages) > 0 else []
        history = self.get_history(history_messages)
        actual_patient_msg = messages[-1]["content"] if len(messages) > 0 else ""
        round_num = self._round_num(history_messages)

        spec_response = self._call_api(
            messages=[{"role": "user", "content": self._get_speculate_prompt(history)}],
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        spec_result = spec_response.choices[0].message.content
        speculate_tokens = completion_token_count(spec_response)

        speculated_responses = self._limit_speculated_responses(
            self._extract_speculated_responses(spec_result),
            default_limit=1,
        )
        similarities = self._compute_branch_similarities(
            actual_patient_msg,
            speculated_responses,
        )
        selected_prediction_index = int(np.argmax(similarities)) if similarities else None
        self.last_branch_prediction_record = self._build_branch_prediction_record(
            messages=messages,
            actual_patient_response=actual_patient_msg,
            predicted_patient_responses=speculated_responses,
            raw_speculation=spec_result,
            similarities=similarities,
            selected_prediction_index=selected_prediction_index,
        )

        analyses = []
        branch_token_counts = []
        for speculated_patient_response in speculated_responses:
            hypothetical_messages = history_messages + [
                {"role": "patient", "content": speculated_patient_response}
            ]
            reason_response = self._call_api(
                messages=[{
                    "role": "user",
                    "content": self._get_main_prompt(hypothetical_messages, force_reactive=True)
                }],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **self._proactive_branch_decode_kwargs(
                    actual_patient_msg,
                    consumed_tokens=speculate_tokens,
                ),
            )
            branch_token_counts.append(completion_token_count(reason_response))
            reason_result = reason_response.choices[0].message.content
            doctor_move = self.extract_answer(reason_result) or reason_result.strip()
            analyses.append({
                "patient_response": speculated_patient_response,
                "doctor_question": doctor_move
            })

        analysis_string = []
        for item in analyses:
            analysis_string.append(
                f'- If the patient responds, "{item["patient_response"]}", you may then respond, "{item["doctor_question"]}".'
            )
        analysis_string.append("- You may ask other questions if the patient's actual response was not included in the analysis.")
        analysis_string = "\n".join(analysis_string)

        final_response = self._call_api(
            messages=[{
                "role": "user",
                "content": (
                    previous_proactive_mandatory_final_prompt_with_think.format(
                        history=history,
                        thinking=analysis_string,
                        response=actual_patient_msg,
                    )
                    if self._is_final_round(messages) else
                    previous_proactive_final_prompt_with_think.format(
                        history=history,
                        thinking=analysis_string,
                        response=actual_patient_msg,
                        round=round_num
                    )
                )
            }],
            temperature=0.5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        content = final_response.choices[0].message.content
        tokens = completion_token_count(final_response)
        middle_delay_tokens = max(branch_token_counts, default=0)
        return content, tokens, middle_delay_tokens

    def get_response(self, messages, patient_tokens=None):
        if self.doctor_type in ["base", "basic", "reactive"]:
            self.last_accepted_count = None
            self.last_branch_prediction_record = None
            prompt = self._get_main_prompt(messages, force_reactive=self.doctor_type == "reactive")
            response = self._call_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            content = response.choices[0].message.content
            tokens = completion_token_count(response)
            return content, tokens, 0
        if self.doctor_type == "previous_proactive":
            return self.proactive_think_previous(messages, patient_tokens=patient_tokens)
        if self.doctor_type == "previous_proactive_with_think":
            return self.proactive_think_previous_with_think(messages, patient_tokens=patient_tokens)
        return self.proactive_think(messages, patient_tokens=patient_tokens)
