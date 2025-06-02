import argparse
import os
import sys
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from utils.util import timestamped_print, save_json, print_args
from .aggregation_util import load_and_merge_json_files_to_hf_dataset
from transformers import AutoTokenizer
from math_verify import parse

def parse_args():
    parser = argparse.ArgumentParser(description="Process data framework.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model.")
    parser.add_argument("--policy_budget", type=int, nargs='+', required=True, help="Budget for policy inference.")
    parser.add_argument("--prm_strategy", type=str, default=None, help="Strategy for PRM (default: none).")
    # parser.add_argument("--reward_budget", type=int, default=1, help="Budget for reward inference.")
    parser.add_argument(
        "--process_module",
        type=str,
        nargs='+',
        required=True,
        help="Path to the user's processor module (e.g., 'user_logic.my_test_processor')."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory containing input JSON files to process."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print_args(args, program_name="Aggregation logic", version="1.0")

    modules = []
    for module_name in args.process_module:
        timestamped_print(f"Attempting to import processor module: {module_name}")
        try:
            module = importlib.import_module(module_name)
            select_function = getattr(module, 'select_function')
            modules.append(select_function)
            timestamped_print(f"Successfully imported '{module_name}'.")
        except ImportError as e:
            timestamped_print(f"ERROR: Could not import processor module '{module_name}'. Details: {e}", "ERROR")
            sys.exit(1)
        except Exception as e:
            timestamped_print(f"ERROR: An error occurred during import of '{module_name}'. Details: {e}", "ERROR")
            sys.exit(1)
    
    if len(args.policy_modules) != len(args.policy_budgets):
        timestamped_print("ERROR: Number of modules must match number of budgets", "ERROR")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    timestamped_print(f"Loaded tokenizer from: {args.tokenizer_path}")

    data = load_and_merge_json_files_to_hf_dataset(
        directory_path=args.input_path,
        features=None,  # Assuming no specific features are needed
        encoding='utf-8',
        max_workers=16  # Adjust as needed for your system
    )
    timestamped_print(f"Loaded data: {data}")

    results = {}
    for i, (module, budget) in enumerate(zip(modules, args.policy_budgets)):
        for record in data:
            responses = record['responses'][:budget]
            steps = record['steps'][:budget]
            correctness = record['correctness'][:budget]
            rewards = record['rewards'][:budget]
            conversations = record['conversations'][:budget]

            extracted_answers = [
                parse(response)[1]
                for response in responses
            ]

            module_name, select_correctness, token_cost = module(
                args=args,
                steps=steps,
                rewards=rewards,
                extracted_answers=extracted_answers,
                correctness=correctness,
                conversations = conversations,
                tokenizer=tokenizer
            )

            if budget not in results:
                results[budget] = {}
            if module_name not in results[budget]:
                results[budget][module_name] = {}
            
            results[budget][module_name]["accuracy"] = select_correctness
            results[budget][module_name]["token_cost"] = token_cost
    
    save_json(results, args.output_path)
    timestamped_print(f"Results saved to: {args.output_path}")
