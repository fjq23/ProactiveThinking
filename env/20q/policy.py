base_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity from the clue and the dialogue history. Since the number of interactions is limited, prioritize your questions to reach the correct guess as efficiently as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Inquiry:** Use `<response>` to ask one yes-or-no question at a time. Keep the question concise and focused.
2. **Guess:** Once you are confident, provide the final entity within `<answer>` tags.
3. **Efficiency:** Minimize the number of interactions. You must provide a definitive guess within **20 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Questioning:** `<response> [A concise yes-or-no question under 20 words] </response>`
* **Upon Guessing:** `<answer> [A definitive entity without ambiguity] </answer>`
""".strip()

reactive_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity from the clue and the dialogue history. Since the number of interactions is limited, prioritize your questions to reach the correct guess as efficiently as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Inquiry:** Use `<response>` to ask one yes-or-no question at a time. Keep the question concise and focused.
2. **Guess:** Once you are confident, provide the final entity within `<answer>` tags.
3. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags. But do not make long enumerative reasoning.
4. **Efficiency:** Minimize the number of interactions. You must provide a definitive guess within **20 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Questioning:** `<thought> [Stepwise reasoning] </thought> <response> [One concise yes-or-no question under 20 words] </response>`
* **Upon Guessing:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive entity without ambiguity] </answer>`
""".strip()

final_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity as accurately as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Final Round:** This is the final round. You must now provide your final guess.
2. **No More Questions:** Do not ask another question.
3. **Answer Only:** Do not output `<response>`.
4. **Best Guess:** If prior evidence is incomplete, provide the single most likely entity.

**Output Format:**
* `<answer> [A definitive entity without ambiguity] </answer>`
""".strip()

reactive_final_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity as accurately as possible.

**Dialogue History:**
{history}

**Operational Protocol:**
1. **Strategic Reasoning:** Before the final answer, reason step by step carefully in `<thought>` tags.
2. **Final Round:** This is the final round. You must now provide your final guess.
3. **No More Questions:** Do not ask another question.
4. **Answer Only:** Do not output `<response>`.
5. **Best Guess:** If prior evidence is incomplete, provide the single most likely entity.

**Output Format:**
* `<thought> [Stepwise reasoning] </thought> <answer> [A definitive entity without ambiguity] </answer>`
""".strip()

speculate_prompt_cot = """
**Dialogue History:**
{history}

What are the most likely user replies to your latest yes-or-no question? Reason step-by-step to list the 2-3 most likely replies, sorted by probability.

Please strictly follow this output format:
<thought> your brief reasoning process </thought>
<response> most likely response </response>
<response> second most likely response </response>
""".strip()

speculate_prompt = """
**Dialogue History:**
{history}

What are the most likely user replies to your latest yes-or-no question? List the 2-3 most likely replies, sorted by probability.

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

What are the top {branch_num} most likely user replies to your latest yes-or-no question? Reason step-by-step briefly, then list the replies sorted by probability.

Please strictly follow this output format:
<thought> your brief reasoning process </thought>
""".strip()
    else:
        prompt = f"""
**Dialogue History:**
{history}

What are the top {branch_num} most likely user replies to your latest yes-or-no question? List the replies sorted by probability.

Please strictly follow this output format:
""".strip()

    response_lines = [
        f"<response> {SPEC_BRANCH_LABELS[i]} response </response>"
        for i in range(branch_num)
    ]
    return "\n".join([prompt, *response_lines])

previous_proactive_reason_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity from the clue and the dialogue history. Since the number of interactions is limited, prioritize your questions to reach the correct guess as efficiently as possible.

**Dialogue History Before The Latest User Reply:**
{history}

**Hypothetical Latest User Reply:**
{patient_response}

**Operational Protocol:**
1. **Planning Mode:** Treat the hypothetical user reply above as if it were just received, then decide your best immediate next move.
2. **Inquiry:** Use `<response>` to ask one yes-or-no question at a time. Keep the question concise and focused.
3. **Guess:** Once you are confident, provide the final entity within `<answer>` tags.
4. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags.
5. **Efficiency:** Minimize the number of interactions. You must provide a definitive guess within **20 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Questioning:** `<thought> [Stepwise reasoning] </thought> <response> [One concise yes-or-no question under 20 words] </response>`
* **Upon Guessing:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive entity without ambiguity] </answer>`
""".strip()

previous_proactive_final_prompt = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity from the clue and the dialogue history. Since the number of interactions is limited, prioritize your questions to reach the correct guess as efficiently as possible.

**Dialogue History Before The Latest User Reply:**
{history}

**Speculative Plans Prepared Before The Latest User Reply:**
{thinking}

**Actual Latest User Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest user reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.
3. **Inquiry:** Use `<response>` to ask one yes-or-no question at a time. Keep the question concise and focused.
4. **Guess:** Once you are confident, provide the final entity within `<answer>` tags.
5. **Efficiency:** Minimize the number of interactions. You must provide a definitive guess within **20 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Questioning:** `<response> [One concise yes-or-no question under 20 words] </response>`
* **Upon Guessing:** `<answer> [A definitive entity without ambiguity] </answer>`
""".strip()

previous_proactive_final_prompt_with_think = """
Act as an efficient 20 Questions player. Communicate **only** through dialogue. Your goal is to identify the hidden entity from the clue and the dialogue history. Since the number of interactions is limited, prioritize your questions to reach the correct guess as efficiently as possible.

**Dialogue History Before The Latest User Reply:**
{history}

**Speculative Plans Prepared Before The Latest User Reply:**
{thinking}

**Actual Latest User Reply:**
{response}

**Operational Protocol:**
1. **Use The Actual Reply:** Treat the actual latest user reply above as the newest turn in the dialogue.
2. **Use Speculation Carefully:** The speculative plans may help, but they are not binding. Revise them freely if the actual reply points elsewhere.
3. **Inquiry:** Use `<response>` to ask one yes-or-no question at a time. Keep the question concise and focused.
4. **Guess:** Once you are confident, provide the final entity within `<answer>` tags.
5. **Strategic Reasoning:** Before each <response> or <answer>, reason step by step carefully in `<thought>` tags.
6. **Efficiency:** Minimize the number of interactions. You must provide a definitive guess within **20 total rounds** of interaction. Current round count: {round}.

**Output Format:**
* **During Questioning:** `<thought> [Stepwise reasoning] </thought> <response> [One concise yes-or-no question under 20 words] </response>`
* **Upon Guessing:** `<thought> [Stepwise reasoning] </thought> <answer> [A definitive entity without ambiguity] </answer>`
""".strip()

import re
import json
import string
import time
import os
import sys
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

STOP_WORDS = set(stopwords.words('english')) - {"yes", "no"}

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
        self.prompt_template = reactive_prompt if self.doctor_type == "reactive" else base_prompt
        self.gamma = args.gamma
        self.spec_cot = args.spec_cot
        self.spec_branch_num = getattr(args, "spec_branch_num", None)
        if self.spec_branch_num is not None and not 1 <= self.spec_branch_num <= 5:
            raise ValueError("--spec-branch-num must be between 1 and 5")
        self.speculate_prompt = speculate_prompt_cot if args.spec_cot else speculate_prompt
        self.last_accepted_count = None
        self.last_branch_prediction_record = None
        self.force_final_answer_models = {
            model.strip()
            for model in args.force_final_answer_models.split(",")
            if model.strip()
        }

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
            self._round_num(messages) >= 20
            and self.model_name in self.force_final_answer_models
        )

    def _get_main_prompt(self, messages, force_reactive=False):
        history = self.get_history(messages)
        use_reactive_prompt = force_reactive or self.doctor_type == "reactive"
        if self._is_final_round(messages):
            if use_reactive_prompt:
                return reactive_final_prompt.format(history=history)
            return final_prompt.format(history=history)
        template = reactive_prompt if force_reactive else self.prompt_template
        return template.format(history=history, round=self._round_num(messages))

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
        role_names = {"patient": "oracle", "assistant": "guesser"}
        return "\n".join(
            f"{role_names.get(turn['role'], turn['role'])}: {self.extract_answer(turn['content']) if turn['role'] == 'assistant' else turn['content']}" 
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
        # print(f"speculative prompt:\n{prompt}")
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
            return response.choices[0].message.content, completion_token_count(response), 0

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
            return response.choices[0].message.content, completion_token_count(response), 0

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
        similarities = [get_jaccard_similarity(actual_patient_msg, o[0]) for o in outputs]
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
        # print("\n[continuation] before verify")
        # print(f"actual_patient_msg: {actual_patient_msg}")
        # print(f"responses: {responses}")
        # print(f"similarities: {similarities}")
        # print(f"selected_spec_patient_response: {selected_spec[0]}")
        # print(f"selected_spec_doctor_response: {selected_spec[1]}")
        # print(f"selected_spec_tokens: {selected_spec[3]}")

        # 4. Verification (Prompt Logprobs)
        # Verify if the speculated doctor response is valid given the ACTUAL message
        # assert(len(selected_spec[1]) > 0)
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
        # print("\n[continuation] after verify")
        # print(f"verify_prompt: {verify_prompt}")
        # print(f"draft_logprobs_len: {len(selected_spec[2])}")
        # print(f"ref_logprobs_len: {len(ref_logprobs) if ref_logprobs is not None else None}")
        # print(f"draft_logprobs: {selected_spec[2]}")
        # print(f"ref_logprobs: {ref_logprobs}")
        # assert(ref_logprobs is not None)
        # print(selected_spec[1])

        # if ref_logprobs is None:
        #     print("sssss")
        #     assert(-1)
        if ref_logprobs is None or len(ref_logprobs) != len(selected_spec[2]):
            self.last_accepted_count = 0
            response = self._call_api(
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content, completion_token_count(response), 0

        # 5. Rejection Sampling
        accepted_count = compute_accepted_tokens(selected_spec[2], ref_logprobs, gamma=self.gamma)
        self.last_accepted_count = accepted_count

        # 6. Final Draft Extension
        prefill = "".join(selected_spec[3][:accepted_count])
        if "</response>" in prefill or "</answer>" in prefill:
            return prefill, 0, middle_delay_tokens
        else:
            # print("\n[continuation] before extend")
            # print(f"gamma: {self.gamma}")
            # print(f"accepted_count: {accepted_count}")
            # print(f"accepted_tokens: {selected_spec[3][:accepted_count]}")
            # print(f"prefill: {prefill}")
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
                    "add_generation_prompt": False,
                },
                # max_tokens = 512
            )
            final_ext = final_ext_response.choices[0].message.content
            # print("\n[continuation] after extend")
            # print(f"final_ext: {final_ext}")
            # print(f"final_text: {prefill + final_ext}")
            final_ext_tokens = completion_token_count(final_ext_response)
            patient_tokens = patient_tokens or 0
            # delay_tokens = final_ext_tokens + max(0, accepted_count - patient_tokens)
            delay_tokens = final_ext_tokens
            return prefill + final_ext, delay_tokens, middle_delay_tokens

    def proactive_think_previous(self, messages, patient_tokens=None):
        self.last_accepted_count = None
        self.last_branch_prediction_record = None
        history_messages = messages[:-1] if len(messages) > 0 else []
        history = self.get_history(history_messages)
        actual_patient_msg = messages[-1]["content"] if len(messages) > 0 else ""
        round_num = len(history_messages) // 2 + 1

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
        similarities = [
            get_jaccard_similarity(actual_patient_msg, response)
            for response in speculated_responses
        ]
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
                {"role": "user", "content": speculated_patient_response}
            ]
            reason_response = self._call_api(
                messages=[{
                    "role": "user",
                    "content": reactive_prompt.format(
                        history=self.get_history(hypothetical_messages),
                        round=round_num
                    )
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
                "content": previous_proactive_final_prompt.format(
                    history=history,
                    thinking=analysis_string,
                    response=actual_patient_msg,
                    round=round_num
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
        round_num = len(history_messages) // 2 + 1

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
        similarities = [
            get_jaccard_similarity(actual_patient_msg, response)
            for response in speculated_responses
        ]
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
                {"role": "user", "content": speculated_patient_response}
            ]
            reason_response = self._call_api(
                messages=[{
                    "role": "user",
                    "content": reactive_prompt.format(
                        history=self.get_history(hypothetical_messages),
                        round=round_num
                    )
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
                "content": previous_proactive_final_prompt_with_think.format(
                    history=history,
                    thinking=analysis_string,
                    response=actual_patient_msg,
                    round=round_num
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
            prompt = self._get_main_prompt(messages)
            response = self._call_api(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            return response.choices[0].message.content, completion_token_count(response), 0
        if self.doctor_type == "previous_proactive":
            return self.proactive_think_previous(messages, patient_tokens=patient_tokens)
        if self.doctor_type == "previous_proactive_with_think":
            return self.proactive_think_previous_with_think(messages, patient_tokens=patient_tokens)
        return self.proactive_think(messages, patient_tokens=patient_tokens)
