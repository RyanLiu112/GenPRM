import argparse
import os
import sys
from typing import List, Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from tqdm import tqdm
from utils.util import timestamped_print, save_json, print_args
from aggregation_util import load_and_merge_json_files_to_hf_dataset
from datasets import Dataset # For type hinting

DEFAULT_ALIGN_KEYS = [
    "policy_responses",
    "steps",
    "correctness",
    "rewards",
    "conversations"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Process and align data from multiple directories for specified keys.")
    parser.add_argument(
        "--input_paths",
        type=str,
        required=True,
        nargs='+',
        help="Space-separated list of directory paths containing input JSON files to process."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Base path to save the output aligned JSON files (turn_X/Y.json structure will be created here)."
    )
    parser.add_argument(
        "--align_keys",
        type=str,
        nargs='+',
        default=DEFAULT_ALIGN_KEYS,
        help=f"Space-separated list of keys to align. Defaults to: {' '.join(DEFAULT_ALIGN_KEYS)}"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of workers for loading JSON files."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print_args(args, program_name="Multi-Key Alignment Logic", version="1.2")

    loaded_datasets: List[Dataset] = []
    for input_dir_path in args.input_paths:
        timestamped_print(f"Loading data from directory: {input_dir_path}...")
        dataset = load_and_merge_json_files_to_hf_dataset(
            directory_path=input_dir_path,
            features=None,
            encoding='utf-8',
            max_workers=args.max_workers
        )
        if len(dataset) == 0:
            timestamped_print(f"Warning: No data loaded from {input_dir_path}. Skipping this directory.", "WARNING")
        else:
            loaded_datasets.append(dataset)
            timestamped_print(f"Loaded {len(dataset)} items from {input_dir_path}.")

    if not loaded_datasets:
        timestamped_print("No data loaded from any input directory. Exiting.", "ERROR")
        return

    num_data_points_to_align = min(len(ds) for ds in loaded_datasets)
    timestamped_print(f"Aligning up to {num_data_points_to_align} data points (based on the shortest dataset).")

    if num_data_points_to_align == 0:
        timestamped_print("No data points to align. Exiting.")
        return

    max_num_turns = 0
    for dataset_idx, dataset in enumerate(loaded_datasets):
        for item_idx in range(min(len(dataset), num_data_points_to_align)):
            item = dataset[item_idx]
            for key_to_align in args.align_keys:
                values_list = item.get(key_to_align)
                if isinstance(values_list, list):
                    max_num_turns = max(max_num_turns, len(values_list))
                elif values_list is not None: # Key exists but is not a list
                     timestamped_print(f"Warning: Dataset {dataset_idx}, Item {item_idx}, key '{key_to_align}' is not a list. Value: {str(values_list)[:100]}", "WARNING")


    if max_num_turns == 0:
        timestamped_print(f"No alignable list data found for keys {args.align_keys} in the first {num_data_points_to_align} items. Exiting.")
        return

    timestamped_print(f"Found a maximum of {max_num_turns} turns to process across specified keys.")

    for turn_idx in tqdm(range(max_num_turns), desc="Processing Turns"):
        turn_output_dir = os.path.join(args.output_path, f"turn_{turn_idx}")
        os.makedirs(turn_output_dir, exist_ok=True)

        for item_idx in tqdm(range(num_data_points_to_align), desc=f"Aligning items for turn_{turn_idx}", leave=False):
            output_data_point: Dict[str, Any] = {}
            
            # Copy other keys from the first dataset's item
            # (assuming they are identical across datasets for this item_idx)
            if loaded_datasets: # Should always be true if we reached here
                base_item = loaded_datasets[0][item_idx]
                for key, value in base_item.items():
                    if key not in args.align_keys:
                        output_data_point[key] = value
            
            # Align the specified keys
            for key_to_align in args.align_keys:
                aligned_list_for_key: List[Any] = []
                for dataset in loaded_datasets:
                    # dataset[item_idx] is safe due to num_data_points_to_align
                    item_data_for_key = dataset[item_idx].get(key_to_align)
                    
                    if isinstance(item_data_for_key, list) and turn_idx < len(item_data_for_key):
                        aligned_list_for_key.append(item_data_for_key[turn_idx])
                    else:
                        aligned_list_for_key.append(None) # Placeholder for missing data
                output_data_point[key_to_align] = aligned_list_for_key
            
            output_file_path = os.path.join(turn_output_dir, f"{item_idx}.json")
            save_json(output_data_point, output_file_path)

    timestamped_print("Alignment processing complete.")
    timestamped_print(f"Output saved under: {args.output_path}")

if __name__ == "__main__":
    main()