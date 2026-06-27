#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


COVERAGE_PROMPT = """
You judge whether an actual patient reply is semantically covered by a list of predicted patient replies.

Actual patient reply:
{actual_patient_response}

Predicted patient replies as a JSON list:
{predicted_patient_responses}

Criteria:
1. Mark covered=true if at least one prediction captures the main clinical meaning of the actual reply.
2. Wording can differ. Synonyms, paraphrases, matching yes/no meaning, matching denial/affirmation, and matching symptoms/timing/test details should count as covered.
3. Mark covered=false if the closest prediction misses the main meaning, is materially narrower, or contradicts the actual reply.
4. Multiple predictions may match the same actual reply. Return every matching zero-based index in matched_prediction_indices.
5. Use an empty list for matched_prediction_indices when covered=false.

Return JSON only:
{{"covered": true, "matched_prediction_indices": [0, 2], "reason": "short explanation"}}
""".strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Use an LLM to judge whether actual patient replies are covered by branch predictions."
    )
    parser.add_argument("--input", required=True, help="Input clinic result JSONL file.")
    parser.add_argument("--output", default=None, help="Output JSONL file for turn-level coverage judgements.")
    parser.add_argument("--api-base", default=os.getenv("BRANCH_JUDGE_API_BASE", "http://localhost:2345/v1"))
    parser.add_argument("--api-key", default=os.getenv("BRANCH_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("BRANCH_JUDGE_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--record-field", default="branch_prediction_records")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many turn records.")
    parser.add_argument("--include-empty", action="store_true", help="Also write turns with no predicted replies.")
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Disable vLLM-style chat_template_kwargs in case the judge endpoint rejects extra_body.",
    )
    return parser.parse_args()


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    if ext:
        return f"{root}.branch_coverage{ext}"
    return f"{input_path}.branch_coverage.jsonl"


def iter_turn_records(input_path, record_field):
    with open(input_path, "r", encoding="utf-8") as f:
        for instance_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            for record_index, record in enumerate(instance.get(record_field, [])):
                yield {
                    "instance_index": instance_index,
                    "record_index": record_index,
                    "turn": record.get("turn"),
                    "actual_patient_response": record.get("actual_patient_response", ""),
                    "predicted_patient_responses": record.get("predicted_patient_responses") or [],
                    "selected_prediction_index": record.get("selected_prediction_index"),
                    "selected_predicted_patient_response": record.get("selected_predicted_patient_response"),
                    "prediction_similarities": record.get("prediction_similarities") or [],
                }


def extract_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text}")
    return json.loads(match.group(0))


def normalize_matched_indices(value, prediction_count):
    if value is None:
        values = []
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        # Keep accepting legacy scalar outputs during the transition.
        values = [value]

    indices = []
    for item in values:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < prediction_count and index not in indices:
            indices.append(index)
    return sorted(indices)


def judge_record(client, model, record, temperature, use_extra_body):
    predictions = record["predicted_patient_responses"]
    result = dict(record)
    if not predictions:
        result.update({
            "covered": None,
            "matched_prediction_index": None,
            "matched_prediction_indices": [],
            "coverage_reason": "No predicted replies were recorded for this turn.",
            "raw_model_output": None,
            "parse_error": None,
        })
        return result

    prompt = COVERAGE_PROMPT.format(
        actual_patient_response=record["actual_patient_response"],
        predicted_patient_responses=json.dumps(predictions, ensure_ascii=False),
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if use_extra_body:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    response = completion_with_backoff(client=client, **kwargs)
    raw_output = response.choices[0].message.content.strip()
    try:
        judgement = extract_json_object(raw_output)
        covered = judgement.get("covered")
        if isinstance(covered, str):
            covered = covered.strip().lower() == "true"
        else:
            covered = bool(covered)
        matched_indices = normalize_matched_indices(
            judgement.get(
                "matched_prediction_indices",
                judgement.get("matched_prediction_index"),
            ),
            len(predictions),
        )
        if not covered:
            matched_indices = []
        elif not matched_indices:
            raise ValueError("covered=true but no valid matched_prediction_indices were returned")
        result.update({
            "covered": covered,
            "matched_prediction_index": matched_indices[0] if matched_indices else None,
            "matched_prediction_indices": matched_indices,
            "coverage_reason": judgement.get("reason", ""),
            "raw_model_output": raw_output,
            "parse_error": None,
        })
    except Exception as exc:
        result.update({
            "covered": None,
            "matched_prediction_index": None,
            "matched_prediction_indices": [],
            "coverage_reason": "",
            "raw_model_output": raw_output,
            "parse_error": str(exc),
        })
    return result


def summarize(results, skipped_empty):
    judged = [item for item in results if item.get("covered") is not None]
    covered = [item for item in judged if item["covered"]]
    top1_covered = [
        item for item in covered
        if 0 in item.get("matched_prediction_indices", [])
    ]
    selected_covered = [
        item for item in covered
        if item.get("selected_prediction_index") in item.get("matched_prediction_indices", [])
    ]
    parse_errors = [item for item in results if item.get("parse_error")]
    return {
        "evaluated_turns": len(judged),
        "covered_turns": len(covered),
        "coverage_rate": len(covered) / len(judged) if judged else None,
        "top1_covered_turns": len(top1_covered),
        "top1_coverage_rate": len(top1_covered) / len(judged) if judged else None,
        "selected_covered_turns": len(selected_covered),
        "selected_coverage_rate": len(selected_covered) / len(judged) if judged else None,
        "parse_error_turns": len(parse_errors),
        "skipped_empty_prediction_turns": skipped_empty,
    }


def main():
    args = parse_arguments()
    output_path = args.output or default_output_path(args.input)

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    model = args.model or client.models.list().data[0].id

    records = list(iter_turn_records(args.input, args.record_field))
    if args.limit is not None:
        records = records[:args.limit]

    skipped_empty = 0
    records_to_write = []
    records_to_judge = []
    for record in records:
        if record["predicted_patient_responses"] or args.include_empty:
            records_to_judge.append(record)
        else:
            skipped_empty += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                judge_record,
                client,
                model,
                record,
                args.temperature,
                not args.no_extra_body,
            )
            for record in records_to_judge
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            records_to_write.append(future.result())

    records_to_write.sort(key=lambda item: (item["instance_index"], item["record_index"]))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in records_to_write:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = summarize(records_to_write, skipped_empty=skipped_empty)
    summary["output"] = output_path
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
