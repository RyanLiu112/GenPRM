import argparse
import os
import sys
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from tqdm import tqdm
from utils.util import timestamped_print, save_json, print_args
from aggregation_util import load_and_merge_json_files_to_hf_dataset
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Process data for F1 and token stats, with per-ID-prefix aggregation.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model.")
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
        help="Path to save the output JSON file with F1 and token stats."
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.5,
        help="Threshold for reward values to determine correctness for F1 calculation."
    )
    parser.add_argument(
        "--id_field",
        type=str,
        default="id",
        help="The field name in the JSON data that contains the unique ID for grouping results."
    )
    parser.add_argument(
        "--id_delimiter",
        type=str,
        default="-",
        help="The delimiter used in the ID field to separate the prefix from the suffix (e.g., '-'). If not found, the whole ID is used as prefix."
    )
    return parser.parse_args()

def calculate_metrics(correct_positive, total_positive, correct_negative, total_negative):
    acc_pos = correct_positive / total_positive if total_positive > 0 else 0.0
    acc_neg = correct_negative / total_negative if total_negative > 0 else 0.0
    f1 = 0.0
    if (acc_pos + acc_neg) > 0:
        f1 = 2 * (acc_pos * acc_neg) / (acc_pos + acc_neg)
    return f1, acc_pos, acc_neg

def get_id_prefix(id_string, delimiter):
    """Extracts the prefix from an ID string before the first occurrence of the delimiter."""
    if id_string is None:
        return None
    if not isinstance(id_string, str): # Ensure it's a string before trying to split
        id_string = str(id_string)

    parts = id_string.split(delimiter, 1)
    return parts[0] # Returns the whole string if delimiter is not found

def main():
    args = parse_args()
    print_args(args, program_name="F1 & Token Stats Aggregation (Per-ID-Prefix)", version="1.3")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    timestamped_print(f"Loaded tokenizer from: {args.tokenizer_path}")

    dataset = load_and_merge_json_files_to_hf_dataset(
        directory_path=args.input_path,
        features=None,
        encoding='utf-8',
        max_workers=16
    )
    timestamped_print(f"Loaded {len(dataset)} data points.")

    if not dataset or len(dataset) == 0:
        timestamped_print("No data loaded. Exiting.", "ERROR")
        save_json({"overall_stats": {}, "per_id_prefix_stats": {}}, args.output_path)
        return

    # Overall stats
    overall_correct_positive_label_neg1 = 0
    overall_correct_negative_label_not_neg1 = 0
    overall_total_positive_label_neg1 = 0
    overall_total_negative_label_not_neg1 = 0
    overall_total_conversation_tokens = 0
    overall_num_samples_with_conversation = 0
    
    # Per-ID-Prefix stats
    results_by_id_prefix = defaultdict(lambda: {
        "correct_positive_label_neg1": 0,
        "correct_negative_label_not_neg1": 0,
        "total_positive_label_neg1": 0,
        "total_negative_label_not_neg1": 0,
        "total_conversation_tokens": 0,
        "num_samples_with_conversation": 0,
        "sample_count": 0
    })
    
    thr = args.reward_threshold

    for sample in tqdm(dataset, desc="Calculating Stats"):
        rewards = sample.get('rewards')
        label = sample.get('label')
        full_id_val = sample.get(args.id_field)
        
        id_prefix = get_id_prefix(full_id_val, args.id_delimiter)

        if rewards is None or not isinstance(rewards, list) or \
           label is None or not isinstance(label, int) or \
           id_prefix is None: # Skip if essential fields or ID prefix are missing
            # timestamped_print(f"Warning: Skipping sample due to missing 'rewards', 'label', or invalid '{args.id_field}' for prefix extraction. Sample: {str(sample)[:100]}...", "WARNING")
            continue

        current_rewards_list = rewards
        if rewards and isinstance(rewards[0], list):
            current_rewards_list = rewards[0]
            if not current_rewards_list or not isinstance(current_rewards_list[0], float):
                continue
        elif not rewards or (rewards and not isinstance(rewards[0], float)):
            continue
        
        results_by_id_prefix[id_prefix]["sample_count"] += 1

        # 1. Check for label match based on rewards
        if label == -1:
            overall_total_positive_label_neg1 += 1
            results_by_id_prefix[id_prefix]["total_positive_label_neg1"] += 1
            all_above_threshold = all(r >= thr for r in current_rewards_list)
            if all_above_threshold:
                overall_correct_positive_label_neg1 += 1
                results_by_id_prefix[id_prefix]["correct_positive_label_neg1"] += 1
        else: # label != -1
            overall_total_negative_label_not_neg1 += 1
            results_by_id_prefix[id_prefix]["total_negative_label_not_neg1"] += 1
            first_reward_below_threshold_idx = next((idx for idx, r in enumerate(current_rewards_list) if r < thr), None)
            if first_reward_below_threshold_idx == label:
                overall_correct_negative_label_not_neg1 += 1
                results_by_id_prefix[id_prefix]["correct_negative_label_not_neg1"] += 1
        
        # 2. Calculate average token count for "conversation"
        conversation = sample.get('conversations')
        if conversation:
            current_sample_tokens = 0
            if isinstance(conversation, list):
                full_conversation_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                current_sample_tokens = len(tokenizer.encode(full_conversation_text))
            elif isinstance(conversation, str):
                current_sample_tokens = len(tokenizer.encode(conversation))
            
            if current_sample_tokens > 0:
                overall_total_conversation_tokens += current_sample_tokens
                overall_num_samples_with_conversation += 1
                results_by_id_prefix[id_prefix]["total_conversation_tokens"] += current_sample_tokens
                results_by_id_prefix[id_prefix]["num_samples_with_conversation"] += 1

    # Calculate overall metrics
    overall_f1, overall_acc_pos, overall_acc_neg = calculate_metrics(
        overall_correct_positive_label_neg1, overall_total_positive_label_neg1,
        overall_correct_negative_label_not_neg1, overall_total_negative_label_not_neg1
    )
    overall_avg_tokens = overall_total_conversation_tokens / overall_num_samples_with_conversation \
        if overall_num_samples_with_conversation > 0 else 0.0

    overall_results = {
        "f1_score": overall_f1,
        "accuracy_for_label_neg1": overall_acc_pos,
        "accuracy_for_label_not_neg1": overall_acc_neg,
        "total_samples_label_neg1": overall_total_positive_label_neg1,
        "total_samples_label_not_neg1": overall_total_negative_label_not_neg1,
        "average_conversation_token_count": overall_avg_tokens,
        "num_samples_with_conversation_for_token_avg": overall_num_samples_with_conversation,
        "reward_threshold_used": thr,
        "total_data_points_processed": len(dataset)
    }
    
    # Calculate per-ID-prefix metrics
    per_id_prefix_final_stats = {}
    for id_prefix_val, stats in results_by_id_prefix.items():
        id_f1, id_acc_pos, id_acc_neg = calculate_metrics(
            stats["correct_positive_label_neg1"], stats["total_positive_label_neg1"],
            stats["correct_negative_label_not_neg1"], stats["total_negative_label_not_neg1"]
        )
        id_avg_tokens = stats["total_conversation_tokens"] / stats["num_samples_with_conversation"] \
            if stats["num_samples_with_conversation"] > 0 else 0.0
        
        per_id_prefix_final_stats[id_prefix_val] = {
            "f1_score": id_f1,
            "accuracy_for_label_neg1": id_acc_pos,
            "accuracy_for_label_not_neg1": id_acc_neg,
            "total_samples_label_neg1": stats["total_positive_label_neg1"],
            "total_samples_label_not_neg1": stats["total_negative_label_not_neg1"],
            "average_conversation_token_count": id_avg_tokens,
            "num_samples_with_conversation_for_token_avg": stats["num_samples_with_conversation"],
            "contributing_sample_count": stats["sample_count"]
        }

    final_output = {
        "overall_stats": overall_results,
        "per_id_prefix_stats": per_id_prefix_final_stats # Renamed key
    }
    
    timestamped_print("\n--- Overall Stats ---")
    timestamped_print(f"F1 Score: {overall_results['f1_score']:.4f}")
    timestamped_print(f"Accuracy for 'Correct' paths (label=-1): {overall_results['accuracy_for_label_neg1']:.4f} (Total: {overall_results['total_samples_label_neg1']})")
    timestamped_print(f"Accuracy for 'Incorrect' paths (label!=-1): {overall_results['accuracy_for_label_not_neg1']:.4f} (Total: {overall_results['total_samples_label_not_neg1']})")
    timestamped_print(f"Average Conversation Token Count: {overall_results['average_conversation_token_count']:.2f} (Samples: {overall_results['num_samples_with_conversation_for_token_avg']})")

    save_json(final_output, args.output_path)
    timestamped_print(f"\nResults saved to: {args.output_path}")

if __name__ == "__main__":
    main()