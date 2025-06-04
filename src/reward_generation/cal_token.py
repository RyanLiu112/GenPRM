import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple # Added Tuple
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sample JSON files from subdirectories and perform analysis.")
    parser.add_argument(
        "--main_directory",
        type=str,
        required=True,
        help="The main directory containing subfolders with JSON files."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the Hugging Face tokenizer (e.g., model name or local path)."
    )
    parser.add_argument(
        "--json_filename",
        type=str,
        default="sample.json",
        help="The name of the JSON file to look for in each subfolder (default: sample.json)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of subfolders (and their JSON files) to randomly sample (default: 100)."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed for sampling."
    )
    return parser.parse_args()

def calculate_steps_stats(steps_data: List[List[str]], tokenizer) -> Tuple[float, float]:
    """
    Calculates:
    1. Average token count per primary path in the 'steps' field.
    2. Average number of steps per primary path in the 'steps' field.

    'steps' is a List of paths, where each path is a List of step strings.
    Returns a tuple: (average_tokens_per_path, average_steps_per_path)
    """
    if not steps_data or not isinstance(steps_data, list):
        return 0.0, 0.0 # Return tuple for consistency

    total_tokens_all_paths = 0
    total_steps_all_paths = 0 # New accumulator for total steps
    num_primary_paths = 0 # Count valid primary paths

    for path in steps_data:
        if isinstance(path, list) and path: # Ensure path is a non-empty list
            num_primary_paths += 1 # Increment for each valid path encountered
            path_text = " ".join(step_str for step_str in path if isinstance(step_str, str))
            total_tokens_all_paths += len(tokenizer.encode(path_text))
            total_steps_all_paths += len(path) # Number of steps in this path
        # else:
            # print(f"Warning: Encountered an item in 'steps' that is not a valid list of steps: {path}")

    if num_primary_paths == 0: # Avoid division by zero if no valid paths
        return 0.0, 0.0

    avg_tokens = total_tokens_all_paths / num_primary_paths
    avg_steps = total_steps_all_paths / num_primary_paths

    return avg_tokens, avg_steps


def get_metrics_sum(metrics_data: Dict[str, Any]) -> int:
    if not metrics_data or not isinstance(metrics_data, dict):
        return 0
    correct_num = metrics_data.get("correct_num", 0)
    incorrect_num = metrics_data.get("incorrect_num", 0)
    invalid_num = metrics_data.get("invalid_num", 0)
    if not isinstance(correct_num, (int, float)): correct_num = 0
    if not isinstance(incorrect_num, (int, float)): incorrect_num = 0
    if not isinstance(invalid_num, (int, float)): invalid_num = 0
    return int(correct_num + incorrect_num + invalid_num)


def map_completor_score(score: float) -> int:
    if not isinstance(score, (int, float)):
        return 0
    if 0 <= score <= 0.1:
        return 128
    elif 0.1 < score <= 0.9:
        return 64
    elif 0.9 < score <= 1.0:
        return 32
    else:
        return 0

def main():
    args = parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"Using random seed: {args.random_seed}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print(f"Successfully loaded tokenizer from: {args.tokenizer_path}")
    except Exception as e:
        print(f"Error loading tokenizer from '{args.tokenizer_path}': {e}")
        return

    if not os.path.isdir(args.main_directory):
        print(f"Error: Main directory '{args.main_directory}' not found or is not a directory.")
        return

    all_subfolders = [
        f for f in os.listdir(args.main_directory)
        if os.path.isdir(os.path.join(args.main_directory, f))
    ]

    if not all_subfolders:
        print(f"No subfolders found in '{args.main_directory}'.")
        return
    print(f"Found {len(all_subfolders)} subfolders in '{args.main_directory}'.")

    num_to_sample = min(args.num_samples, len(all_subfolders))
    if num_to_sample < args.num_samples:
        print(f"Warning: Requested {args.num_samples} samples, but only {len(all_subfolders)} subfolders are available. Sampling {num_to_sample}.")
    
    sampled_subfolders = random.sample(all_subfolders, num_to_sample)
    print(f"Randomly sampled {len(sampled_subfolders)} subfolders.")

    total_avg_steps_tokens_across_files = 0
    total_avg_num_steps_in_path_across_files = 0 # New accumulator
    total_metrics_sum_across_files = 0
    total_mapped_completor_score_across_files = 0
    valid_files_processed_for_steps_stats = 0 # Renamed for clarity, used for both token and step count
    valid_files_processed_for_metrics = 0
    valid_files_processed_for_completor_score = 0

    for folder_name in tqdm(sampled_subfolders, desc="Processing sampled folders"):
        json_file_path = os.path.join(args.main_directory, folder_name, args.json_filename)

        if not os.path.isfile(json_file_path):
            continue

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from '{json_file_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Could not read file '{json_file_path}': {e}. Skipping.")
            continue

        # Task 1 & New Task: Steps stats (tokens and count)
        steps_data = data.get("steps")
        if steps_data is not None:
            avg_tokens_for_file, avg_steps_for_file = calculate_steps_stats(steps_data, tokenizer)
            if avg_tokens_for_file > 0 or avg_steps_for_file > 0: # Check if any valid path was processed
                total_avg_steps_tokens_across_files += avg_tokens_for_file
                total_avg_num_steps_in_path_across_files += avg_steps_for_file # Accumulate avg steps
                valid_files_processed_for_steps_stats += 1
        
        metrics_data = data.get("metrics")
        if metrics_data is not None:
            metrics_sum_for_file = get_metrics_sum(metrics_data)
            total_metrics_sum_across_files += metrics_sum_for_file
            valid_files_processed_for_metrics += 1

        completor_score_value = data.get("completor_score")
        if completor_score_value is not None:
            mapped_score_for_file = map_completor_score(completor_score_value)
            total_mapped_completor_score_across_files += mapped_score_for_file
            valid_files_processed_for_completor_score += 1
            
    print("\n--- Aggregated Results ---")

    if valid_files_processed_for_steps_stats > 0:
        final_avg_steps_tokens = total_avg_steps_tokens_across_files / valid_files_processed_for_steps_stats
        final_avg_num_steps = total_avg_num_steps_in_path_across_files / valid_files_processed_for_steps_stats # Calculate final average
        print(f"Average tokens per path in 'steps' (across {valid_files_processed_for_steps_stats} files): {final_avg_steps_tokens:.2f}")
        print(f"Average number of steps per path in 'steps' (across {valid_files_processed_for_steps_stats} files): {final_avg_num_steps:.2f}") # Print new stat
    else:
        print("No valid 'steps' data found in sampled files to calculate average tokens or step counts.")

    if valid_files_processed_for_metrics > 0:
        final_avg_metrics_sum = total_metrics_sum_across_files / valid_files_processed_for_metrics
        print(f"Average sum of 'correct_num', 'incorrect_num', 'invalid_num' (across {valid_files_processed_for_metrics} files): {final_avg_metrics_sum:.2f}")
    else:
        print("No valid 'metrics' data found in sampled files.")

    if valid_files_processed_for_completor_score > 0:
        final_avg_mapped_completor_score = total_mapped_completor_score_across_files / valid_files_processed_for_completor_score
        print(f"Average mapped 'completor_score' (across {valid_files_processed_for_completor_score} files): {final_avg_mapped_completor_score:.2f}")
    else:
        print("No valid 'completor_score' data found in sampled files.")

if __name__ == "__main__":
    main()