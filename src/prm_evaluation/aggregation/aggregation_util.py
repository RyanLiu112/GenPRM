import os
import json
from typing import Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
from datasets import Dataset, Features
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_json_file(filepath: str, encoding: str = 'utf-8') -> Optional[List[Dict[str, Any]]]:
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
            if not content.strip():
                print(f"Skipping empty JSON file: {filepath}")
                return None

            data = json.loads(content)
            
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    return data
                else:
                    print(f"Warning: JSON file '{filepath}' is a list but contains non-dict items. Skipping.")
                    return None
            else:
                print(f"Warning: JSON file '{filepath}' does not contain a dict or list of dicts. Skipping.")
                return None
                
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from file: {filepath}. Skipping.")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}. Skipping.")
        return None

def load_and_merge_json_files_to_hf_dataset(
    directory_path: str,
    features: Optional[Features] = None,
    encoding: str = 'utf-8',
    max_workers: int = 4
) -> Optional[Dataset]:
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return None

    json_files = [
        os.path.join(directory_path, filename) 
        for filename in os.listdir(directory_path) 
        if filename.endswith(".json")
    ]
    
    if not json_files:
        print("No JSON files found in the directory.")
        return None

    print(f"Found {len(json_files)} JSON files. Loading with {max_workers} workers...")
    
    all_data_records: List[Dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_json_file, filepath, encoding): filepath 
            for filepath in json_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading JSON files"):
            result = future.result()
            if result:
                all_data_records.extend(result)

    if not all_data_records:
        print("No valid data records found in JSON files to create a dataset.")
        return None

    print(f"Successfully loaded {len(all_data_records)} records from JSON files.")

    try:
        if features:
            hf_dataset = Dataset.from_list(all_data_records, features=features)
        else:
            print("No features provided, attempting to infer schema from data...")
            hf_dataset = Dataset.from_list(all_data_records)
            print(f"Inferred features: {hf_dataset.features}")

        print(f"Hugging Face Dataset created successfully with {len(hf_dataset)} rows.")
        return hf_dataset

    except Exception as e:
        print(f"Error creating Hugging Face Dataset: {e}")
        return None

def count_tokens(
    text: str, 
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    add_special_tokens: bool = False
) -> int:
    encoding = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return len(encoding)
