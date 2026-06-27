import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENV_DIR = Path(__file__).resolve().parents[1]
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from utils import (
    extract_prompt_logprobs_for_text as extract_prompt_logprobs_for_text_raw,
    extract_prompt_logprobs_for_tokens as extract_prompt_logprobs_for_tokens_raw,
    trim_tokens_to_text,
)

from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


@dataclass
class Tau2AgentState:
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


AGENT_TOKENS_PER_SEC = 100.0
TAU2_BRANCH_MAX_TOKENS_CAP = 1024
TAU2_AGENT_TOOL_DELAY_SEC = 0.5
TAU2_USER_TOOL_DELAY_SEC = 5.0


def normalize_method_name(method: str) -> str:
    normalized = (method or "basic").replace("-", "_")
    if normalized in {"basic", "base", "no_cot"}:
        return normalized
    if normalized == "reactive":
        return normalized
    if normalized == "previous_proactive":
        return normalized
    if normalized == "previous_proactive_with_think":
        return normalized
    if normalized in {"proactive", "speculative_proactive"}:
        return "speculative_proactive"
    return normalized


def simple_tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or "", re.UNICODE))


def tau2_branch_max_tokens_for_message(message: Message) -> int:
    if isinstance(message, UserMessage):
        if getattr(message, "tool_calls", None):
            return int(TAU2_USER_TOOL_DELAY_SEC * AGENT_TOKENS_PER_SEC)
        return min(
            TAU2_BRANCH_MAX_TOKENS_CAP,
            int((0.5 + word_count(message.content or "")) * AGENT_TOKENS_PER_SEC),
        )

    if isinstance(message, ToolMessage):
        if getattr(message, "requestor", "assistant") == "user":
            return int(TAU2_USER_TOOL_DELAY_SEC * AGENT_TOKENS_PER_SEC)
        return int(TAU2_AGENT_TOOL_DELAY_SEC * AGENT_TOKENS_PER_SEC)

    return int(TAU2_AGENT_TOOL_DELAY_SEC * AGENT_TOKENS_PER_SEC)


def get_jaccard_similarity(text1: str, text2: str) -> float:
    set1 = simple_tokenize(text1)
    set2 = simple_tokenize(text2)
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def render_message(message: Message) -> str:
    role = getattr(message, "role", "unknown")
    if isinstance(message, ToolMessage):
        return f"tool: {message.content or ''}".strip()
    if getattr(message, "tool_calls", None):
        tool_descriptions = []
        for tool_call in message.tool_calls:
            tool_descriptions.append(f"{tool_call.name}({tool_call.arguments})")
        return f"{role}: [tool_calls] " + "; ".join(tool_descriptions)
    return f"{role}: {message.content or ''}".strip()


def render_history(messages: list[Message]) -> str:
    return "\n".join(render_message(message) for message in messages)


def completion_tokens_from_message(message: Optional[AssistantMessage]) -> int:
    if message is None or not message.usage:
        return 0
    return int(message.usage.get("completion_tokens", 0) or 0)


def attach_protocol_metadata(
    message: AssistantMessage,
    *,
    delay_tokens: int,
    middle_delay_tokens: int = 0,
    accepted_count: Optional[int] = None,
    speculative_used: bool = False,
    selected_reply: Optional[str] = None,
) -> AssistantMessage:
    raw_data = dict(message.raw_data or {})
    raw_data["benchmark_tau2"] = {
        "delay_tokens": int(delay_tokens),
        "middle_delay_tokens": int(middle_delay_tokens),
        "accepted_count": accepted_count,
        "speculative_used": speculative_used,
        "selected_reply": selected_reply,
    }
    message.raw_data = raw_data
    return message


def extract_content_logprobs(
    message: AssistantMessage,
) -> Optional[tuple[list[str], list[float]]]:
    raw_data = message.raw_data or {}
    choices = raw_data.get("choices") or []
    if not choices:
        return None
    logprobs = (choices[0] or {}).get("logprobs") or {}
    content = logprobs.get("content")
    if not isinstance(content, list):
        return None

    content_text = message.content or ""
    tokens: list[str] = []
    logprob_values: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        token = item.get("token")
        logprob = item.get("logprob")
        if token is None or logprob is None:
            continue
        try:
            logprob_values.append(float(logprob))
        except (TypeError, ValueError):
            continue
        tokens.append(str(token))
    if not tokens or len(tokens) != len(logprob_values):
        return None
    trimmed = trim_tokens_to_text(tokens, logprob_values, content_text)
    if trimmed is not None:
        return trimmed
    return tokens, logprob_values


def extract_prompt_logprobs_for_tokens(
    message: AssistantMessage, target_tokens: list[str]
) -> Optional[list[float]]:
    raw_data = message.raw_data or {}
    logprobs = extract_prompt_logprobs_for_tokens_raw(
        raw_data.get("prompt_logprobs"), target_tokens
    )
    return logprobs or None


def extract_prompt_logprobs_for_text(
    message: AssistantMessage, target_text: str
) -> Optional[list[float]]:
    raw_data = message.raw_data or {}
    logprobs = extract_prompt_logprobs_for_text_raw(
        raw_data.get("prompt_logprobs"), target_text
    )
    return logprobs or None


def compute_accepted_tokens(
    draft_logprobs: list[float],
    target_logprobs: list[float],
    gamma: float,
    rng: random.Random,
) -> int:
    accepted_count = 0
    for draft_logprob, target_logprob in zip(draft_logprobs, target_logprobs):
        acceptance_prob = math.exp(
            min(0.0, math.log(gamma) + target_logprob - draft_logprob)
        )
        if rng.random() < acceptance_prob:
            accepted_count += 1
        else:
            break
    return accepted_count
