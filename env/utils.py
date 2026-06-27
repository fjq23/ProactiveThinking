import os
import re
from typing import Optional


DEFAULT_AGENT_TOKENS_PER_SEC = 100.0
DEFAULT_USER_BASE_DELAY_SEC = 0.5
DEFAULT_USER_TYPING_WPM = 60.0


def _read_env_float(name, default, *, min_value, allow_equal):
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return float(default)

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc

    is_valid = value >= min_value if allow_equal else value > min_value
    comparator = ">=" if allow_equal else ">"
    if not is_valid:
        raise ValueError(f"{name} must be {comparator} {min_value}, got {value}")
    return value


def get_latency_config():
    return {
        "agent_tokens_per_sec": _read_env_float(
            "AGENT_TOKENS_PER_SEC",
            DEFAULT_AGENT_TOKENS_PER_SEC,
            min_value=0.0,
            allow_equal=False,
        ),
        "user_base_delay_sec": _read_env_float(
            "USER_BASE_DELAY_SEC",
            DEFAULT_USER_BASE_DELAY_SEC,
            min_value=0.0,
            allow_equal=True,
        ),
        "user_typing_wpm": _read_env_float(
            "USER_TYPING_WPM",
            DEFAULT_USER_TYPING_WPM,
            min_value=0.0,
            allow_equal=False,
        ),
    }


def word_count(text):
    return len(re.findall(r"\w+", text or "", re.UNICODE))


def dynamic_branch_max_tokens(
    user_reply,
    upper_cap=512,
    consumed_tokens=0,
    min_tokens=1,
    agent_tokens_per_sec=None,
    user_base_delay_sec=None,
    user_typing_wpm=None,
):
    config = get_latency_config()
    agent_tokens_per_sec = (
        config["agent_tokens_per_sec"]
        if agent_tokens_per_sec is None
        else float(agent_tokens_per_sec)
    )
    user_base_delay_sec = (
        config["user_base_delay_sec"]
        if user_base_delay_sec is None
        else float(user_base_delay_sec)
    )
    user_typing_wpm = (
        config["user_typing_wpm"]
        if user_typing_wpm is None
        else float(user_typing_wpm)
    )

    user_typing_words_per_sec = user_typing_wpm / 60.0
    available_seconds = (
        user_base_delay_sec
        + word_count(user_reply) / user_typing_words_per_sec
    )
    base_budget = min(upper_cap, int(available_seconds * agent_tokens_per_sec))
    remaining_budget = base_budget - max(0, int(consumed_tokens or 0))
    return max(min_tokens, remaining_budget)


def completion_token_count(value):
    if isinstance(value, str):
        return len(re.findall(r"\w+|[^\w\s]", value, re.UNICODE)) if value else 0

    usage = getattr(value, "usage", None)
    if usage and getattr(usage, "completion_tokens", None) is not None:
        return usage.completion_tokens

    content = getattr(getattr(value.choices[0], "message", None), "content", "") or ""
    if not content:
        return 0

    return len(re.findall(r"\w+|[^\w\s]", content, re.UNICODE))


def trim_tokens_to_text(
    tokens: list[str], logprobs: list[float], text: str
) -> Optional[tuple[list[str], list[float]]]:
    if len(tokens) != len(logprobs):
        return None
    if not text:
        return [], []

    consumed = ""
    trimmed_tokens: list[str] = []
    trimmed_logprobs: list[float] = []
    for token, logprob in zip(tokens, logprobs):
        candidate = consumed + token
        if not text.startswith(candidate):
            break
        trimmed_tokens.append(token)
        trimmed_logprobs.append(logprob)
        consumed = candidate
        if consumed == text:
            return trimmed_tokens, trimmed_logprobs
    return None


def normalize_token(token: Optional[str]) -> str:
    return token or ""


def prompt_logprobs_to_candidate_sets(raw_prompt_logprobs) -> list[dict[str, float]]:
    if not isinstance(raw_prompt_logprobs, list):
        return []

    candidate_sets: list[dict[str, float]] = []
    for pos in raw_prompt_logprobs:
        if pos is None or not isinstance(pos, dict):
            continue

        candidates: dict[str, float] = {}
        values = [pos] if ("decoded_token" in pos or "token" in pos) else pos.values()
        for item in values:
            if not isinstance(item, dict):
                continue
            tok = normalize_token(item.get("decoded_token") or item.get("token"))
            lp = item.get("logprob")
            if not tok or lp is None:
                continue
            try:
                lp_value = float(lp)
            except (TypeError, ValueError):
                continue
            if tok not in candidates or lp_value > candidates[tok]:
                candidates[tok] = lp_value
        if candidates:
            candidate_sets.append(candidates)
    return candidate_sets


def _suffix_biased_start_positions(total_positions: int, target_len: int) -> list[int]:
    if total_positions <= 0:
        return []

    recent_window = 16
    approx_start = max(0, total_positions - target_len)
    preferred_start = max(0, approx_start - recent_window)
    preferred_end = (
        min(total_positions - target_len, approx_start)
        if target_len <= total_positions
        else -1
    )

    start_positions = list(range(preferred_start, preferred_end + 1))
    start_positions.reverse()
    seen = set(start_positions)
    start_positions.extend(
        start
        for start in range(total_positions - target_len, -1, -1)
        if start not in seen
    )
    return start_positions


def extract_prompt_logprobs_for_tokens(
    raw_prompt_logprobs, target_tokens: list[str]
) -> Optional[list[float]]:
    candidate_sets = prompt_logprobs_to_candidate_sets(raw_prompt_logprobs)
    target_tokens = [
        normalize_token(token) for token in target_tokens if normalize_token(token)
    ]
    if not candidate_sets or not target_tokens or len(target_tokens) > len(candidate_sets):
        return None

    for start in _suffix_biased_start_positions(len(candidate_sets), len(target_tokens)):
        if start < 0 or start + len(target_tokens) > len(candidate_sets):
            continue
        ref_logprobs = []
        for offset, target_token in enumerate(target_tokens):
            candidates = candidate_sets[start + offset]
            if target_token not in candidates:
                ref_logprobs = []
                break
            ref_logprobs.append(candidates[target_token])
        if ref_logprobs:
            return ref_logprobs
    return None


def _match_text_from_position(
    candidate_sets: list[dict[str, float]], start: int, target_text: str
) -> Optional[list[float]]:
    stack: list[tuple[int, int, list[float]]] = [(start, 0, [])]
    visited: set[tuple[int, int]] = set()

    while stack:
        pos, offset, matched = stack.pop()

        if offset == len(target_text):
            return matched
        if pos >= len(candidate_sets):
            continue

        state = (pos, offset)
        if state in visited:
            continue
        visited.add(state)

        items = sorted(
            candidate_sets[pos].items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        next_states: list[tuple[int, int, list[float]]] = []
        for token, logprob in items:
            if token and target_text.startswith(token, offset):
                next_states.append((pos + 1, offset + len(token), matched + [logprob]))

        stack.extend(reversed(next_states))

    return None


def extract_prompt_logprobs_for_text(
    raw_prompt_logprobs, target_text: str
) -> Optional[list[float]]:
    candidate_sets = prompt_logprobs_to_candidate_sets(raw_prompt_logprobs)
    target_text = target_text or ""
    if not candidate_sets or not target_text:
        return None

    for start in range(len(candidate_sets) - 1, -1, -1):
        matched = _match_text_from_position(candidate_sets, start, target_text)
        if matched:
            return matched
    return None


def resolve_prompt_logprobs(response):
    prompt_logprobs = getattr(response, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        return prompt_logprobs

    if not hasattr(response, "model_dump"):
        return None

    response_dict = response.model_dump()
    return response_dict.get("prompt_logprobs") or response_dict.get("choices", [{}])[0].get(
        "prompt_logprobs"
    )
