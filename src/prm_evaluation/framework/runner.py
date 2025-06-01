#####################################################           import packeges and args             ########################################################

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import argparse
import json
import random
import time
import threading
import traceback
import framework.register
from utils.util import timestamped_print, print_args


random_seed = int(time.time())
random.seed(random_seed)
timestamped_print(f"Random seed initialized to: {random_seed}")

os.environ['VLLM_USE_V1'] = '0'

TIME_LIMIT = 300  # set time limit
stop_event = threading.Event()

def heart_beat_worker(file_path):
    start_time = time.time()

    while not stop_event.is_set():
        if os.path.exists(file_path):
            try:
                os.utime(file_path)
                timestamped_print(f"Heartbeat updated: {file_path}")
            except Exception as e:
                timestamped_print(f"Update file time error: {str(e)}", 'ERROR')
        else:
            try:
                with open(file_path, 'w') as f:
                    pass
                timestamped_print(f"Created file while heart beating: {file_path}", 'ERROR')
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')

        for _ in range(6):
            if stop_event.is_set():
                timestamped_print("Heartbeat worker exiting...")
                return
            time.sleep(5)

def is_json_file_empty(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return True
    
    if os.stat(file_path).st_size == 0:
        return True
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if not data:
                return True
            if isinstance(data, (dict, list)) and len(data) == 0:
                return True
                
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True
    
    return False

def run_infer(args: argparse.Namespace):
    #####################################################           load splited dataset             ########################################################

    random.seed(int(time.time()))

    def get_shuffled_files(directory):
        full_paths = []
        if not os.path.isdir(directory):
            print(f"Error: Directory '{directory}' not found or is not a directory.")
            return full_paths

        for f_name in os.listdir(directory):
            full_path = os.path.join(directory, f_name)
            if os.path.isfile(full_path):
                full_paths.append(full_path)

        random.shuffle(full_paths)
        return full_paths
    
    def get_shuffled_folders(directory):
        folders = [f for f in os.listdir(directory) 
                if os.path.isdir(os.path.join(directory, f))]
        random.shuffle(folders)
        return folders
    

    if args.store_type == 'file':
        target_list = get_shuffled_files(args.input_path)
    elif args.store_type == 'folder':
        raise NotImplementedError("Folder input type is not supported yet.")
        
    #####################################################           processing dataset             ########################################################

    for data_path in target_list:
        base_name = os.path.basename(data_path)
        save_path = os.path.join(args.output_path, base_name)

        output_directory = os.path.dirname(save_path)
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                timestamped_print(f"Created output directory: {output_directory}")
            except Exception as e:
                timestamped_print(f"Error creating output directory {output_directory}: {e}", 'ERROR')
                continue

        if os.path.exists(save_path):
            if not is_json_file_empty(save_path):
                timestamped_print(f"Skip: File {save_path} already exists and is not empty.")
                continue
            file_modification_time = os.path.getmtime(save_path)
            current_time = time.time()
            if (current_time - file_modification_time) > TIME_LIMIT:
                timestamped_print(f"Warning: File {save_path} exists and its modification time exceeds TIME_LIMIT. Will attempt to overwrite.", 'WARNING')
            else:
                timestamped_print(f"Skip: File {save_path} already exists and is within TIME_LIMIT.")
                continue
        else:
            timestamped_print(f"Info: Target file {save_path} does not exist. Will be created.")
            pass

        try:
            # Create a thread for the heartbeat worker
            stop_event.clear()
            thread = threading.Thread(target=heart_beat_worker, args=(save_path,))
            thread.daemon = True
            thread.start()
            timestamped_print("Heartbeat thread started. Main thread continues...")

            # Run the inference process
            args.input_filepath = data_path
            args.output_filepath = save_path
            framework.register._the_user_function(args)

            timestamped_print(f"Processing {data_path} completed successfully.", 'INFO')

        except Exception as e:
            timestamped_print(f"Error processing {data_path}: {str(e)}", 'ERROR')
            traceback.print_exc()
        finally:
            stop_event.set()
            thread.join(timeout=5)

            if thread.is_alive():
                timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
            else:
                timestamped_print("Heartbeat thread exited successfully")
