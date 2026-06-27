import traceback

from openai import OpenAI
from env import *
from policy import *
import json
import concurrent.futures
from threading import Lock
from tqdm import tqdm
import argparse
import sys
import copy
import random

DEBUG = False

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run 20 Questions dialogue simulation')
    
    # Patient API arguments
    parser.add_argument('--patient-api-base', type=str, default="http://localhost:2345/v1",
                       help='Base URL for patient OpenAI API')
    parser.add_argument('--patient-api-key', type=str, default="EMPTY",
                       help='API key for patient OpenAI API')
    parser.add_argument('--patient-model', type=str, default=None,
                       help='Model name for patient (if not provided, will auto-detect)')
    
    # Doctor API arguments
    parser.add_argument('--doctor-api-base', type=str, default="http://localhost:2345/v1",
                       help='Base URL for doctor OpenAI API')
    parser.add_argument('--doctor-api-key', type=str, default="EMPTY",
                       help='API key for doctor OpenAI API')
    parser.add_argument('--doctor-model', type=str, default=None,
                       help='Model name for doctor (if not provided, will auto-detect)')
    
    # Judge API arguments
    parser.add_argument('--judge-api-base', type=str, default="http://localhost:2345/v1",
                       help='Base URL for judge OpenAI API')
    parser.add_argument('--judge-api-key', type=str, default="EMPTY",
                       help='API key for judge OpenAI API')
    parser.add_argument('--judge-model', type=str, default=None,
                       help='Model name for judge (if not provided, will auto-detect)')

    # Execution parameters
    parser.add_argument('--max-turn', type=int, default=20,
                       help='Maximum number of dialogue turns')
    parser.add_argument('--min-turn', type=int, default=0,
                       help='Minimum number of dialogue turns')
    parser.add_argument('--enable-thinking', type=str_to_bool, default=False,
                       help='Whether to adopt thinking mode')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Generation temperature')
    parser.add_argument('--max-workers', type=int, default=64,
                       help='Maximum number of worker threads')
    parser.add_argument('--max-sample', type=int, default=1,
                       help='Maximum number of samples to test')
    parser.add_argument('--run-number', type=int, default=1,
                       help='Run dataset multiple times')
    parser.add_argument('--static-strategy', type=str_to_bool, default=False,
                       help='Use static inquiry strategy')
    parser.add_argument('--patient-type', type=str, default="basic",
                       help='Patient prompting type, can be basic, mcqa, and maque')
    parser.add_argument('--doctor-type', type=str, default="basic",
                       help='Doctor prompting type, can be basic, reactive, proactive, previous-proactive, and previous-proactive-with-think')
    parser.add_argument('--max-think-token', type=int, default=-1,
                       help='Max token number for thinking')
    parser.add_argument('--gamma', type=float, default=1,
                       help='Max token number for thinking')
    parser.add_argument('--spec-cot', type=str_to_bool, default=False,
                       help='Whether to adopt thinking mode')
    parser.add_argument('--spec-branch-num', type=int, choices=range(1, 6), default=None,
                       help='Experimental: number of speculative user-response branches to request and execute, from 1 to 5')
    parser.add_argument('--force-final-answer-models', type=str, default="",
                       help='Compatibility flag accepted by the shared runner; currently unused in 20Q')


    # File paths
    parser.add_argument('--input', type=str, default="data/test_20q.json",
                       help='Input data file path')
    parser.add_argument('--output', type=str, default="data/test_20q_results.jsonl",
                       help='Output file path')
    
    return parser.parse_args()

class ScenarioTwentyQ:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.prompt = scenario_dict["prompt"]
        self.answer = scenario_dict["answer"]
        self.category = scenario_dict.get("category", "")
    
    def patient_information(self) -> dict:
        patient_info = {
            "prompt": self.prompt,
            "answer": self.answer,
        }
        if self.category:
            patient_info["category"] = self.category
        return patient_info
    
    def diagnosis_information(self) -> str:
        return self.answer


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    stripped_data = raw_data.strip()
    if not stripped_data:
        return []

    try:
        parsed = json.loads(stripped_data)
    except json.JSONDecodeError:
        dataset = []
        for line_number, line in enumerate(raw_data.splitlines(), start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                dataset.append(ScenarioTwentyQ(json.loads(stripped_line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e
        return dataset

    if isinstance(parsed, list):
        return [ScenarioTwentyQ(instance) for instance in parsed]
    if isinstance(parsed, dict):
        return [ScenarioTwentyQ(parsed)]
    raise ValueError(f"Unsupported JSON root type: {type(parsed).__name__}")

def main():
    args = parse_arguments()

    def process_instance(instance):
        # Extract target answer and clean instance
        patient_information = json.dumps(instance.patient_information(), ensure_ascii=False)
        judge_information = instance.diagnosis_information()

        # Initialize agents with thread-specific client
        patient_agent = Patient(args, patient_information)
        judge_agent = Judge(args, judge_information)
        doctor_agent = Doctor(args)

        # retry 3 times
        for retry in range(3):
            try:
                messages = [] # 只含有对话，不含 system prompt
                patient_delay_tokens = []
                doctor_delay_tokens = []
                doctor_middle_delay_tokens = []
                accepted_counts = []
                branch_prediction_records = []
                judge_result = "No"
                cnt = 0
                while cnt < args.max_turn:
                    cnt += 1

                    patient_turn, patient_delay = patient_agent.get_response(messages)
                    messages.append({"role": "patient", "content": patient_turn})
                    patient_delay_tokens.append(patient_delay)

                    doctor_turn, doctor_delay, doctor_middle_delay = doctor_agent.get_response(messages, patient_tokens=patient_delay)
                    messages.append({"role": "assistant", "content": doctor_turn})
                    doctor_delay_tokens.append(doctor_delay)
                    doctor_middle_delay_tokens.append(doctor_middle_delay)
                    if doctor_agent.last_accepted_count is not None:
                        accepted_counts.append(doctor_agent.last_accepted_count)

                    branch_prediction_record = copy.deepcopy(doctor_agent.last_branch_prediction_record)
                    if branch_prediction_record is None:
                        branch_prediction_record = {
                            "turn": cnt,
                            "actual_patient_response": patient_turn,
                            "predicted_patient_responses": [],
                            "selected_prediction_index": None,
                            "selected_predicted_patient_response": None,
                            "prediction_similarities": [],
                            "raw_speculation": None,
                        }
                    branch_prediction_records.append(branch_prediction_record)
                    
                    # 只有检查到诊断才会 Judge
                    result = judge_agent.get_response(messages)
                    if result is not None:
                        judge_result = result
                        break
                    
                final_output = copy.deepcopy(instance.scenario_dict)
                final_output["messages"] = messages
                final_output["judgement"] = judge_result
                final_output["patient_delay_total"] = sum(patient_delay_tokens)
                final_output["doctor_delay_total"] = sum(doctor_delay_tokens)
                final_output["doctor_middle_delay_total"] = sum(doctor_middle_delay_tokens)
                final_output["accepted_counts"] = accepted_counts
                final_output["branch_prediction_records"] = branch_prediction_records
                if doctor_agent.spec_branch_num is not None:
                    final_output["doctor_spec_branch_num"] = doctor_agent.spec_branch_num
                # print("done")
                
                break
            except Exception as e:
                print(f"An exception occurs, retry this session: {e}")
                traceback.print_exc()

            final_output = copy.deepcopy(instance.scenario_dict)
            final_output["messages"] = None
            final_output["judgement"] = "No"
            final_output["patient_delay_total"] = 0
            final_output["doctor_delay_total"] = 0
            final_output["doctor_middle_delay_total"] = 0
            final_output["accepted_counts"] = []
            final_output["branch_prediction_records"] = []
            if doctor_agent.spec_branch_num is not None:
                final_output["doctor_spec_branch_num"] = doctor_agent.spec_branch_num
        return final_output

    # Read dataset
    try:
        dataset = load_dataset(args.input)
        if args.max_sample < len(dataset):
            dataset = random.sample(dataset, k=args.max_sample)
        if args.run_number > 1:
            extended_dataset = []
            for _ in range(args.run_number):
                for instance in dataset:
                    extended_dataset.append(copy.deepcopy(instance))
            dataset = extended_dataset
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Process instances concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        processed_instances = list(tqdm(executor.map(process_instance, dataset), total=len(dataset)))

    # Write results
    try:
        with open(args.output, "w") as f:
            f.write("\n".join(json.dumps(instance) for instance in processed_instances))
        print(f"Results successfully written to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
