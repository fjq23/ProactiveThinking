import argparse
import concurrent.futures
import copy
import functools
import json
import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
TAU2_SRC = ROOT_DIR / "third_party" / "tau2-bench" / "src"
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

try:
    from policy import create_benchmark_tau2_agent
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import get_tasks, run_single_task
except ModuleNotFoundError as exc:
    missing_module = exc.name or "unknown"
    raise SystemExit(
        "Missing tau2 dependency: "
        f"{missing_module}. Install dependencies from "
        f"{ROOT_DIR / 'third_party' / 'tau2-bench' / 'pyproject.toml'} first."
    ) from exc


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run tau2 benchmark through local benchmark interface"
    )

    parser.add_argument("--patient-api-base", type=str, default="http://localhost:2345/v1")
    parser.add_argument("--patient-api-key", type=str, default="EMPTY")
    parser.add_argument("--patient-model", type=str, default="openai/gpt-4.1-mini")

    parser.add_argument("--doctor-api-base", type=str, default="http://localhost:2345/v1")
    parser.add_argument("--doctor-api-key", type=str, default="EMPTY")
    parser.add_argument("--doctor-model", type=str, default="openai/gpt-4.1-mini")

    parser.add_argument("--judge-api-base", type=str, default="http://localhost:2345/v1")
    parser.add_argument("--judge-api-key", type=str, default="EMPTY")
    parser.add_argument("--judge-model", type=str, default=None)

    parser.add_argument("--max-turn", type=int, default=200)
    parser.add_argument("--min-turn", type=int, default=0)
    parser.add_argument("--enable-thinking", type=str_to_bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-sample", type=int, default=1)
    parser.add_argument("--run-number", "--num-trials", dest="num_trials", type=int, default=1)
    parser.add_argument("--static-strategy", type=str_to_bool, default=False)
    parser.add_argument("--patient-type", type=str, default="basic")
    parser.add_argument("--doctor-type", type=str, default="basic")
    parser.add_argument("--max-think-token", type=int, default=-1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--spec-cot", type=str_to_bool, default=False)
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--domain", type=str, default="retail")
    parser.add_argument("--task-set-name", type=str, default=None)
    parser.add_argument("--task-split-name", type=str, default="base")
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_agent_registered():
    agent_name = "benchmark_tau2_agent"
    if registry.get_agent_factory(agent_name) is None:
        registry.register_agent_factory(create_benchmark_tau2_agent, agent_name)
    return agent_name


def normalize_litellm_model(model_name: str) -> str:
    if not model_name:
        return model_name
    if "/" in model_name or model_name.startswith("ft:"):
        return model_name
    return f"openai/{model_name}"


def normalize_full_message(message):
    role = getattr(message, "role", "unknown")
    content = message.content or ""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        content = "; ".join(
            f"{tool_call.name}({tool_call.arguments})" for tool_call in tool_calls
        )
    return {
        "role": role,
        "content": content,
    }


def normalize_visible_messages(messages):
    normalized = []
    for message in messages:
        if isinstance(message, UserMessage):
            normalized.append({"role": "user", "content": message.content or ""})
        elif isinstance(message, AssistantMessage):
            if message.tool_calls:
                continue
            normalized.append({"role": "assistant", "content": message.content or ""})
        elif isinstance(message, ToolMessage):
            continue
    return normalized


def completion_tokens(message) -> int:
    if not getattr(message, "usage", None):
        return 0
    return int(message.usage.get("completion_tokens", 0) or 0)


def extract_protocol_metadata(message) -> dict:
    raw_data = getattr(message, "raw_data", None) or {}
    metadata = raw_data.get("benchmark_tau2")
    if isinstance(metadata, dict):
        return metadata
    return {}


def extract_reactive_metadata(message) -> dict:
    raw_data = getattr(message, "raw_data", None) or {}
    metadata = raw_data.get("benchmark_tau2_reactive")
    if isinstance(metadata, dict):
        return metadata
    return {}


def assistant_delay_total(messages) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        metadata = extract_protocol_metadata(message)
        if "delay_tokens" in metadata:
            total += int(metadata["delay_tokens"] or 0)
        else:
            total += completion_tokens(message)
    return total


def assistant_middle_delay_total(messages) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        metadata = extract_protocol_metadata(message)
        total += int(metadata.get("middle_delay_tokens", 0) or 0)
    return total


def user_delay_total(messages) -> int:
    total = 0
    for message in messages:
        if isinstance(message, UserMessage):
            total += completion_tokens(message)
    return total


def accepted_counts(messages) -> list[int]:
    counts = []
    for message in messages:
        if not isinstance(message, AssistantMessage):
            continue
        metadata = extract_protocol_metadata(message)
        accepted_count = metadata.get("accepted_count")
        if accepted_count is not None:
            counts.append(int(accepted_count))
    return counts


def reactive_traces(messages) -> list[dict]:
    traces = []
    for turn, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        metadata = extract_reactive_metadata(message)
        if metadata:
            traces.append({"turn": turn, **metadata})
    return traces


def warn_ignored_args(args):
    ignored = [
        "--min-turn",
        "--static-strategy",
        "--patient-type",
        "--max-think-token",
        "--input",
    ]
    print(
        "Ignoring clinic-compatibility args in tau2 backend: " + ", ".join(ignored),
        file=sys.stderr,
    )


def configure_judge_llm(args) -> None:
    """Route evaluator-side LLM calls through the configured judge endpoint."""
    from tau2.evaluator import (
        auth_classifier,
        evaluator_nl_assertions,
        hallucination_reviewer,
        review_llm_judge,
        review_llm_judge_user_only,
    )

    judge_model = normalize_litellm_model(args.judge_model or args.patient_model)
    judge_api_base = args.judge_api_base
    judge_api_key = args.judge_api_key

    def wrap_generate(module_generate):
        @functools.wraps(module_generate)
        def wrapped_generate(*, model, messages, **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("api_base", judge_api_base)
            kwargs.setdefault("api_key", judge_api_key)
            return module_generate(
                model=judge_model,
                messages=messages,
                **kwargs,
            )

        return wrapped_generate

    evaluator_nl_assertions.generate = wrap_generate(evaluator_nl_assertions.generate)
    auth_classifier.generate = wrap_generate(auth_classifier.generate)
    review_llm_judge.generate = wrap_generate(review_llm_judge.generate)
    review_llm_judge_user_only.generate = wrap_generate(
        review_llm_judge_user_only.generate
    )
    hallucination_reviewer.generate = wrap_generate(hallucination_reviewer.generate)


def build_run_specs(tasks, num_trials: int, seed: int):
    random.seed(seed)
    trial_seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    run_specs = []
    for trial, trial_seed in enumerate(trial_seeds):
        for task in tasks:
            run_specs.append((trial, trial_seed, copy.deepcopy(task)))
    return run_specs


def main():
    args = parse_arguments()
    agent_name = ensure_agent_registered()
    configure_judge_llm(args)
    warn_ignored_args(args)

    task_set_name = args.task_set_name or args.domain
    tasks = get_tasks(
        task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks or args.max_sample,
    )

    if args.max_sample < len(tasks):
        tasks = tasks[: args.max_sample]

    config = TextRunConfig(
        domain=args.domain,
        task_set_name=task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        llm_user=normalize_litellm_model(args.patient_model),
        llm_args_user={
            "temperature": args.temperature,
            "api_base": args.patient_api_base,
            "api_key": args.patient_api_key,
        },
        agent=agent_name,
        llm_agent=normalize_litellm_model(args.doctor_model),
        llm_args_agent={
            "temperature": args.temperature,
            "api_base": args.doctor_api_base,
            "api_key": args.doctor_api_key,
            "method": args.doctor_type,
            "gamma": args.gamma,
            "spec_cot": args.spec_cot,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
            },
        },
        max_steps=args.max_turn,
        max_errors=args.max_errors,
        seed=args.seed,
        max_concurrency=max(1, args.max_workers),
        user="user_simulator",
    )

    run_specs = build_run_specs(tasks, num_trials=args.num_trials, seed=args.seed)

    def run_once(run_spec):
        trial, trial_seed, task = run_spec
        simulation = run_single_task(config, task, seed=trial_seed)
        reward = simulation.reward_info.reward if simulation.reward_info else 0
        messages = simulation.messages or []
        result = {
            "task_id": task.id,
            "trial": trial,
            "seed": trial_seed,
            "domain": args.domain,
            "messages": normalize_visible_messages(messages),
            "raw_messages": [normalize_full_message(message) for message in messages],
            "judgement": "Yes" if reward and reward > 0 else "No",
            "reward": reward,
            "patient_delay_total": user_delay_total(messages),
            "doctor_delay_total": assistant_delay_total(messages),
            "doctor_middle_delay_total": assistant_middle_delay_total(messages),
            "accepted_counts": accepted_counts(messages),
            "termination_reason": str(simulation.termination_reason),
        }
        traces = reactive_traces(messages)
        if traces:
            result["reactive_traces"] = traces
        return result

    max_workers = max(1, args.max_workers)
    if max_workers == 1:
        outputs = [run_once(run_spec) for run_spec in run_specs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            outputs = list(executor.map(run_once, run_specs))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if outputs:
            f.write(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in outputs) + "\n"
            )

    print(f"Results successfully written to {output_path}")


if __name__ == "__main__":
    main()
