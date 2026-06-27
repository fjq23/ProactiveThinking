import sys
import jsonlines
import json
import re
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parents[1] / "env"
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from utils import get_latency_config, word_count


LATENCY_CONFIG = get_latency_config()
AGENT_TOKENS_PER_SEC = LATENCY_CONFIG["agent_tokens_per_sec"]
USER_BASE_DELAY_SEC = LATENCY_CONFIG["user_base_delay_sec"]
USER_TYPING_WPM = LATENCY_CONFIG["user_typing_wpm"]
USER_TYPING_WORDS_PER_SEC = USER_TYPING_WPM / 60.0


def count_tau2_steps(raw_messages):
    steps = 0
    prev_role = None
    for message in raw_messages:
        role = message.get("role")
        if role in {"assistant", "user", "doctor", "patient"}:
            steps += 1
        elif role == "tool":
            # Consecutive tool responses belong to the same env step.
            if prev_role != "tool":
                steps += 1
        prev_role = role
    return steps


def count_instance_turns(instance):
    raw_messages = instance.get("raw_messages")
    if raw_messages and any(message.get("role") == "tool" for message in raw_messages):
        return count_tau2_steps(raw_messages)
    return len(instance["messages"]) // 2


def is_tau2_instance(instance):
    if "tau2" in data_fpath:
        return True
    raw_messages = instance.get("raw_messages")
    return bool(raw_messages and any(message.get("role") == "tool" for message in raw_messages))


def user_messages(instance):
    if is_tau2_instance(instance):
        return []
    return [
        message
        for message in instance["messages"]
        if message.get("role") in {"user", "patient"}
    ]


def count_words(text):
    return word_count(str(text or ""))


def user_latency_seconds(messages):
    user_turns = len(messages)
    user_words = sum(count_words(message.get("content", "")) for message in messages)
    return (
        USER_BASE_DELAY_SEC * user_turns
        + user_words / USER_TYPING_WORDS_PER_SEC
    )


def tau2_input_messages(instance):
    raw_messages = instance.get("raw_messages") or []
    return [
        message
        for message in raw_messages
        if message.get("role") in {"user", "tool"}
    ]


data_fpath = sys.argv[1]
with jsonlines.open(data_fpath) as reader:
    dataset = list(reader)

corr = 0
total = 0
turn = 0
word = []
doctor_delay_total = 0
doctor_middle_delay_total = 0
doctor_latency_sec_total = 0
doctor_middle_latency_sec_total = 0
user_turn_total = 0
user_word_total = 0
user_latency_sec_total = 0
simulated_latency_sec_total = 0
has_user_latency = False
save_token = []
doctor_delay_by_turn = []
for instance in dataset:
    instance_turn = count_instance_turns(instance)
    turn += instance_turn

    word += [
        len(turn["content"].split())
        for turn in instance["messages"]
        if turn["role"] in {"assistant", "doctor"}
    ]

    if instance["judgement"].lower().strip() == "yes":
        corr += 1
    total += 1

    save_token += instance.get("accepted_counts", [])
    instance_doctor_delay = instance.get("doctor_delay_total", 0)
    instance_doctor_middle_delay = instance.get("doctor_middle_delay_total", 0)
    for turn_idx, delay_tokens in enumerate(instance.get("doctor_delay_per_turn", [])):
        if turn_idx == len(doctor_delay_by_turn):
            doctor_delay_by_turn.append([])
        doctor_delay_by_turn[turn_idx].append(delay_tokens)
    instance_doctor_latency_sec = instance_doctor_delay / AGENT_TOKENS_PER_SEC
    instance_doctor_middle_latency_sec = (
        instance_doctor_middle_delay / AGENT_TOKENS_PER_SEC
    )
    doctor_delay_total += instance_doctor_delay
    doctor_middle_delay_total += instance_doctor_middle_delay
    doctor_latency_sec_total += instance_doctor_latency_sec
    doctor_middle_latency_sec_total += instance_doctor_middle_latency_sec

    if is_tau2_instance(instance):
        instance_user_messages = tau2_input_messages(instance)
        instance_user_latency_sec = len(instance_user_messages) * USER_BASE_DELAY_SEC
        if instance_user_messages:
            has_user_latency = True
        user_turn_total += len(instance_user_messages)
        user_latency_sec_total += instance_user_latency_sec
    else:
        instance_user_messages = user_messages(instance)
        instance_user_latency_sec = user_latency_seconds(instance_user_messages)
        if instance_user_messages:
            has_user_latency = True
        user_turn_total += len(instance_user_messages)
        user_word_total += sum(
            count_words(message.get("content", "")) for message in instance_user_messages
        )
        user_latency_sec_total += instance_user_latency_sec
    simulated_latency_sec_total += (
        instance_doctor_delay / AGENT_TOKENS_PER_SEC
        + instance_user_latency_sec
    )
    
print("Agent Tokens / Sec:", AGENT_TOKENS_PER_SEC)
print("User Typing WPM:", USER_TYPING_WPM)
print("User Base Delay Sec:", USER_BASE_DELAY_SEC)
print("Accuracy:", corr / total)
print("Turn Num:", turn / total)
print("Word Num:", sum(word) / len(word) if word else 0.0)
if turn:
    print("Doctor Delay / Turn Tokens:", doctor_delay_total / turn)
    print("Doctor Delay / Turn Seconds:", doctor_latency_sec_total / turn)
    print("Doctor Middle Delay / Turn Tokens:", doctor_middle_delay_total / turn)
    print("Doctor Middle Delay / Turn Seconds:", doctor_middle_latency_sec_total / turn)
if has_user_latency and user_turn_total:
    print("User Delay / Turn Seconds:", user_latency_sec_total / user_turn_total)
    if not any(is_tau2_instance(instance) for instance in dataset):
        print("User Words / Turn:", user_word_total / user_turn_total)
if turn:
    print("Total Simulated Latency / Turn Seconds:", simulated_latency_sec_total / turn)
if total:
    print("Total Simulated Latency / Instance Seconds:", simulated_latency_sec_total / total)
for turn_idx, delay_tokens in enumerate(doctor_delay_by_turn, start=1):
    print(
        f"Doctor Delay / Dialogue Turn {turn_idx} Seconds:",
        sum(delay_tokens) / len(delay_tokens) / AGENT_TOKENS_PER_SEC,
    )
if save_token:
    print("Save Token Num:", sum(save_token) / len(save_token))
