import argparse
import os
import sys
import importlib
import multiprocessing as mp
import math # For ceiling division

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from tqdm import tqdm
from functools import partial
from utils.util import timestamped_print, save_json, print_args
from aggregation_util import load_and_merge_json_files_to_hf_dataset
from transformers import AutoTokenizer
from math_verify import parse # Assuming this is your custom parsing function

# Define the list of fields that are expected to be lists and are chunked
LIST_FIELDS_FOR_CHUNKING = ['policy_responses', 'steps', 'correctness', 'rewards', 'conversations']

def parse_args():
    parser = argparse.ArgumentParser(description="Process data framework.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model.")
    parser.add_argument("--policy_budget", type=int, nargs='+', required=True, help="Budget for policy inference.")
    parser.add_argument("--prm_strategy", type=str, default=None, help="Strategy for PRM (default: none).")
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

def process_chunk(chunk_data, module, args, tokenizer):
    """
    Processes a single chunk of data based on the provided module.
    chunk_data fields like 'steps', 'rewards', etc., might be lists of Nones if original data was missing.
    """
    # policy_responses is crucial, others can be None-lists
    responses = chunk_data['policy_responses']
    steps = chunk_data.get('steps') # Will be None if not generated or passed
    correctness = chunk_data.get('correctness')
    rewards = chunk_data.get('rewards')
    conversations = chunk_data.get('conversations')

    extracted_answers = []
    if responses: # Ensure responses is not None
        for response in responses:
            # If response itself is None (due to padding from a missing field), parse would fail
            if response is not None:
                try:
                    parsed_response, answer = parse(response) # Assuming parse returns two values
                    extracted_answers.append(answer)
                except Exception as e:
                    # Handle cases where a response (even if not None) cannot be parsed
                    # timestamped_print(f"Warning: Could not parse response in chunk: '{str(response)[:50]}...'. Error: {e}", "WARNING")
                    extracted_answers.append(None) # Or some other default
            else:
                extracted_answers.append(None) # If response was None
    else: # If responses itself is a list of Nones or empty
        # Create a list of Nones matching the expected chunk size if possible,
        # but select_function should handle len(extracted_answers) correctly.
        # This case should ideally be caught before calling process_chunk if policy_responses are essential.
        pass


    # The module's select_function will now operate on this chunk.
    # It MUST be robust to receiving lists that might contain None, or entirely be a list of Nones
    # for fields other than 'policy_responses'.
    module_name, select_correctness, token_cost = module(
        args=args,
        steps=steps, # Could be a list of (lists of steps) or list of Nones
        rewards=rewards, # Could be a list of (lists of rewards) or list of Nones
        extracted_answers=extracted_answers, # List of extracted answers, may contain Nones
        correctness=correctness, # Could be a list of (lists of correctness) or list of Nones
        conversations=conversations, # Could be a list of (lists of conversations) or list of Nones
        tokenizer=tokenizer
    )
    
    return module_name, int(select_correctness), token_cost


def process_record_with_chunking(record_tuple, module, budget_value, args, tokenizer):
    record_idx, record = record_tuple

    # 1. Pre-process record: Generate 'steps' if missing
    if 'steps' not in record and 'policy_responses' in record and isinstance(record['policy_responses'], list):
        record['steps'] = []
        for resp in record['policy_responses']:
            if isinstance(resp, str):
                record['steps'].append(resp.split('\n\n'))
            else:
                record['steps'].append(None) # Or handle non-string responses differently
        # timestamped_print(f"Generated 'steps' for record {record.get('id', record_idx)}", "DEBUG")


    # 2. Determine available_len based on 'policy_responses' as the primary driver.
    #    Other fields will be padded if shorter or missing.
    available_len = 0
    if record.get('policy_responses') and isinstance(record['policy_responses'], list):
        available_len = len(record['policy_responses'])
    
    if available_len == 0:
        timestamped_print(f"Warning: Record {record.get('id', record_idx)} has no 'policy_responses' or it's empty. Skipping.", 'WARNING')
        return None

    num_chunks = math.ceil(available_len / budget_value)
    if num_chunks == 0:
        return None

    record_total_correctness = 0
    record_total_token_cost = 0
    record_processed_chunks = 0
    module_name_from_chunk = None

    for i in range(num_chunks):
        start_idx = i * budget_value
        # current_chunk_size is primarily driven by policy_responses' available length for this chunk
        current_chunk_size = min(budget_value, available_len - start_idx)
        if current_chunk_size <= 0:
            continue

        chunk_data = {}
        
        # Ensure policy_responses for the chunk are valid
        policy_responses_chunk = record['policy_responses'][start_idx : start_idx + current_chunk_size]
        if not policy_responses_chunk or all(r is None for r in policy_responses_chunk): # if chunk is empty or all None
            # timestamped_print(f"Debug: Skipping chunk {i} for record {record.get('id', record_idx)} due to empty/all-None policy_responses_chunk.", "DEBUG")
            continue
        chunk_data['policy_responses'] = policy_responses_chunk
        
        # For other fields, slice if present, otherwise pad with Nones
        for field in LIST_FIELDS_FOR_CHUNKING:
            if field == 'policy_responses': # Already handled
                continue
            
            source_list = record.get(field)
            if source_list and isinstance(source_list, list):
                # If source_list is shorter than policy_responses, effectively it will be padded by this logic:
                # We take up to current_chunk_size elements starting from start_idx for this field.
                # If len(source_list) is less than start_idx + current_chunk_size, slicing will just return what's available.
                # The select_function needs to handle lists of varying effective lengths or padded Nones.
                
                field_chunk = []
                # We need to ensure the field_chunk has `current_chunk_size` elements, padding with None if necessary
                # relative to the `policy_responses` timeline.
                for chunk_item_idx in range(current_chunk_size):
                    original_list_idx = start_idx + chunk_item_idx
                    if original_list_idx < len(source_list) and source_list[original_list_idx] is not None:
                        field_chunk.append(source_list[original_list_idx])
                    else:
                        field_chunk.append(None) # Pad with None
                chunk_data[field] = field_chunk
            else:
                # Field is missing or not a list, fill with Nones for this chunk
                chunk_data[field] = [None] * current_chunk_size
        
        try:
            m_name, chunk_correctness, chunk_token_cost = process_chunk(
                chunk_data, module, args, tokenizer
            )
            if m_name is not None:
                if module_name_from_chunk is None:
                    module_name_from_chunk = m_name
                record_total_correctness += chunk_correctness
                record_total_token_cost += chunk_token_cost
                record_processed_chunks += 1
        except Exception as e:
            timestamped_print(f"ERROR processing chunk for record {record.get('id', record_idx)}, budget {budget_value}, chunk {i}: {e}", "ERROR")

    if record_processed_chunks > 0:
        avg_correctness = record_total_correctness / record_processed_chunks
        avg_token_cost = record_total_token_cost / record_processed_chunks
        return module_name_from_chunk, avg_correctness, avg_token_cost
    else:
        return None

def main():
    args = parse_args()
    print_args(args, program_name="Chunked Budget Aggregation Logic with Fallbacks", version="1.2")

    modules = []
    for module_path_str in args.process_module:
        timestamped_print(f"Attempting to import processor module: {module_path_str}")
        try:
            actual_module = importlib.import_module(module_path_str)
            select_function = getattr(actual_module, 'select_function')
            modules.append(select_function)
            timestamped_print(f"Successfully imported 'select_function' from '{module_path_str}'.")
        except ImportError as e:
            timestamped_print(f"ERROR: Could not import processor module '{module_path_str}'. Details: {e}", "ERROR")
            sys.exit(1)
        except AttributeError:
            timestamped_print(f"ERROR: Function 'select_function' not found in module '{module_path_str}'.", "ERROR")
            sys.exit(1)
        except Exception as e:
            timestamped_print(f"ERROR: An error occurred during import of '{module_path_str}'. Details: {e}", "ERROR")
            sys.exit(1)
    
    if len(modules) != len(args.policy_budget):
        timestamped_print("ERROR: Number of successfully imported modules must match number of budgets", "ERROR")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    timestamped_print(f"Loaded tokenizer from: {args.tokenizer_path}")

    dataset = load_and_merge_json_files_to_hf_dataset(
        directory_path=args.input_path,
        features=None,
        encoding='utf-8',
        max_workers=16 
    )
    timestamped_print(f"Loaded data: {dataset}")

    results_summary = {}
    
    for i, (module_func, budget_val) in enumerate(zip(modules, args.policy_budget)):
        if budget_val <= 0:
            timestamped_print(f"Warning: Budget value {budget_val} is non-positive. Skipping.", "WARNING")
            continue

        process_func_for_pool = partial(process_record_with_chunking,
                                        module=module_func,
                                        budget_value=budget_val,
                                        args=args,
                                        tokenizer=tokenizer)
        
        processed_records_stats = []
        num_processes = min(mp.cpu_count(), len(dataset)) if len(dataset) > 0 else 1

        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(process_func_for_pool, enumerate(dataset)),
                total=len(dataset),
                desc=f"Processing records for budget {budget_val}"
            ):
                if result is not None:
                    processed_records_stats.append(result)
        
        if not processed_records_stats:
            timestamped_print(f"No records were successfully processed for budget {budget_val}. Skipping summary.", "WARNING")
            continue
        
        current_module_name_str = None
        for m_name, _, _ in processed_records_stats:
            if m_name is not None:
                current_module_name_str = m_name
                break
        
        if current_module_name_str is None:
            current_module_name_str = args.process_module[i] 
            timestamped_print(f"Warning: Could not determine module name from processing results for budget {budget_val}. Using '{current_module_name_str}'.", "WARNING")

        total_avg_correctness = sum(correct for _, correct, _ in processed_records_stats)
        total_avg_token_cost = sum(cost for _, _, cost in processed_records_stats)
        num_valid_records = len(processed_records_stats)

        if budget_val not in results_summary:
            results_summary[budget_val] = {}
        if current_module_name_str not in results_summary[budget_val]:
            results_summary[budget_val][current_module_name_str] = {}
        
        results_summary[budget_val][current_module_name_str]["accuracy"] = total_avg_correctness / num_valid_records if num_valid_records > 0 else 0.0
        results_summary[budget_val][current_module_name_str]["token_cost"] = total_avg_token_cost / num_valid_records if num_valid_records > 0 else 0.0
        results_summary[budget_val][current_module_name_str]["num_records_processed_for_avg"] = num_valid_records
    
    save_json(results_summary, args.output_path)
    timestamped_print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()
