#!/bin/bash

# ==============================================================================
# Script to submit DLC PyTorch jobs with configurable parameters.
#
# Usage:
#   ./submit_dlc_job.sh -c "<your_command>" -w <num_workers> -g <gpus_per_worker> [-N <num_submissions>] [-n <job_name_prefix>] [-p <priority>] [--other-dlc-options...]
#
# Example:
#   ./submit_dlc_job.sh -c "python /mnt/workspace/train.py --epochs 10" -w 2 -g 8 -N 3 -n "my-training"
#   ./submit_dlc_job.sh --command "echo 'Hello DLC'" --workers 1 --gpus_per_worker 1
# ==============================================================================

# --- Default Configuration (can be overridden by environment variables or command-line arguments) ---
DEFAULT_JOB_NAME_PREFIX="zj_unknown"
DEFAULT_PRIORITY=9
DEFAULT_NUM_SUBMISSIONS=1 # -N, default to 1 submission
DEFAULT_WORKERS=1         # Number of worker nodes
DEFAULT_GPUS_PER_WORKER=1 # GPUs per worker node

# --- Fixed DLC Configuration (can be moved to a config file or set as env vars if they change often) ---
DATA_SOURCES="d-x135sqzws1argjld1r"
RESOURCE_ID="quota1bhq0p32wuc"  # llmit6
WORKSPACE_ID="84885"
DRIVER_VERSION="535.54.03"
# Oversold type can also be made configurable if needed
# OVERSOLD_TYPE="ForbiddenQuotaOverSold"
OVERSOLD_TYPE="ForceQuotaOverSold"
# OVERSOLD_TYPE="AcceptQuotaOverSold"

# --- Default Worker Resource Specs (Consider making these configurable too if needed) ---
# These might come from environment variables set by your DLC environment or a config file
WORKER_IMAGE=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/liurunze:liurunze-sllm02 # Use env var or a default
# WORKER_CPU="${WORKER_CPU:-16}"
WORKER_MEMORY=200
WORKER_SHARED_MEMORY=200

# --- Helper Functions ---
print_usage() {
    echo "Usage: $0 -c <command> -w <num_workers> -g <gpus_per_worker> [-N <num_submissions>] [-n <job_name_prefix>] [-p <priority>] [other_dlc_options...]"
    echo ""
    echo "Required arguments:"
    echo "  -c, --command <string>            The command to execute on worker nodes."
    echo "  -w, --workers <int>               Number of worker nodes (NNODES)."
    echo "  -g, --gpus_per_worker <int>       Number of GPUs per worker node (N_GPUS_PER_WORKER_NODE)."
    echo ""
    echo "Optional arguments:"
    echo "  -N, --num_submissions <int>       Number of times to submit the job in a loop (default: $DEFAULT_NUM_SUBMISSIONS)."
    echo "  -n, --name_prefix <string>        Prefix for the DLC job name (default: $DEFAULT_JOB_NAME_PREFIX)."
    echo "  -p, --priority <int>              Job priority (default: $DEFAULT_PRIORITY)."
    echo "  --worker_image <string>           Docker image for workers (default: $WORKER_IMAGE or env WORKER_IMAGE)."
    # echo "  --worker_cpu <int>                CPUs per worker (default: $WORKER_CPU or env WORKER_CPU)."
    # echo "  --worker_memory <string>          Memory per worker, e.g., 64Gi (default: $WORKER_MEMORY or env WORKER_MEMORY)."
    # echo "  --job_max_running_time <int>      Job max running time in minutes (0 for unlimited, default: 0)."
    echo "  --data_sources <string>           DLC data sources (default: $DATA_SOURCES)."
    echo "  --resource_id <string>            DLC resource ID (default: $RESOURCE_ID)."
    echo "  --workspace_id <string>           DLC workspace ID (default: $WORKSPACE_ID)."
    echo "  --oversold_type <string>          Oversold type (default: $OVERSOLD_TYPE)."
    echo "  -h, --help                        Show this help message."
    echo ""
    echo "Any other arguments will be passed directly to 'dlc submit pytorchjob'."
    echo "Example: $0 -c 'python train.py' -w 2 -g 4 -- --custom_arg value"
}

timestamped_echo() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $@"
}

# --- Argument Parsing ---
JOB_COMMAND=""
NNODES=$DEFAULT_WORKERS
N_GPUS_PER_WORKER_NODE=$DEFAULT_GPUS_PER_WORKER
NUM_SUBMISSIONS=$DEFAULT_NUM_SUBMISSIONS
JOB_NAME_PREFIX=$DEFAULT_JOB_NAME_PREFIX
PRIORITY=$DEFAULT_PRIORITY
# JOB_MAX_RUNNING_TIME_MINUTES=0 # Default: 0 for unlimited

# Array to hold passthrough arguments for dlc submit
DLC_PASSTHROUGH_ARGS=()

# Parse named arguments first
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--command)
            JOB_COMMAND="$2"
            shift 2
            ;;
        -w|--workers)
            NNODES="$2"
            shift 2
            ;;
        -g|--gpus_per_worker)
            N_GPUS_PER_WORKER_NODE="$2"
            shift 2
            ;;
        -N|--num_submissions)
            NUM_SUBMISSIONS="$2"
            shift 2
            ;;
        -n|--name_prefix)
            JOB_NAME_PREFIX="$2"
            shift 2
            ;;
        -p|--priority)
            PRIORITY="$2"
            shift 2
            ;;
        --worker_image)
            WORKER_IMAGE="$2"
            shift 2
            ;;
        --worker_cpu)
            WORKER_CPU="$2"
            shift 2
            ;;
        --worker_memory)
            WORKER_MEMORY="$2"
            shift 2
            ;;
        --job_max_running_time)
            JOB_MAX_RUNNING_TIME_MINUTES="$2"
            shift 2
            ;;
        --data_sources)
            DATA_SOURCES="$2"
            shift 2
            ;;
        --resource_id)
            RESOURCE_ID="$2"
            shift 2
            ;;
        --workspace_id)
            WORKSPACE_ID="$2"
            shift 2
            ;;
        --oversold_type)
            OVERSOLD_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --) # End of named arguments, rest are passthrough
            shift
            DLC_PASSTHROUGH_ARGS+=("$@")
            break
            ;;
        -*) # Unknown option
            timestamped_echo "ERROR: Unknown option: $1"
            print_usage
            exit 1
            ;;
        *) # Positional argument, treat as passthrough (or error if not expected)
            DLC_PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Validate Required Arguments ---
if [[ -z "$JOB_COMMAND" ]]; then
    timestamped_echo "ERROR: Job command (-c or --command) is required."
    print_usage
    exit 1
fi
if ! [[ "$NNODES" =~ ^[0-9]+$ ]] || [[ "$NNODES" -lt 1 ]]; then
    timestamped_echo "ERROR: Number of workers (-w or --workers) must be a positive integer."
    print_usage
    exit 1
fi
if ! [[ "$N_GPUS_PER_WORKER_NODE" =~ ^[0-9]+$ ]] || [[ "$N_GPUS_PER_WORKER_NODE" -lt 0 ]]; then # Allow 0 for CPU-only
    timestamped_echo "ERROR: GPUs per worker (-g or --gpus_per_worker) must be a non-negative integer."
    print_usage
    exit 1
fi
if ! [[ "$NUM_SUBMISSIONS" =~ ^[0-9]+$ ]] || [[ "$NUM_SUBMISSIONS" -lt 1 ]]; then
    timestamped_echo "ERROR: Number of submissions (-N or --num_submissions) must be a positive integer."
    print_usage
    exit 1
fi

# --- Source bashrc if it exists (optional, can be useful for aliases or env vars) ---
if [ -f "$HOME/.bashrc" ]; then
    timestamped_echo "Sourcing $HOME/.bashrc"
    source "$HOME/.bashrc"
fi

# --- Main Submission Loop ---
timestamped_echo "Starting DLC job submission process..."
timestamped_echo "Job Command: $JOB_COMMAND"
timestamped_echo "Number of Workers (NNODES): $NNODES"
timestamped_echo "GPUs per Worker: $N_GPUS_PER_WORKER_NODE"
timestamped_echo "Number of Submissions: $NUM_SUBMISSIONS"
timestamped_echo "Job Name Prefix: $JOB_NAME_PREFIX"
timestamped_echo "Worker Image: $WORKER_IMAGE"
timestamped_echo "---"

FIXED_PREFIX=$(cat <<'EOF'
source $HOME/.bashrc
NEW_HOME=/cpfs02/user/liurunze
eval "$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)"
EOF
)
JOB_COMMAND="${FIXED_PREFIX}
${JOB_COMMAND}"

for i in $(seq 1 "$NUM_SUBMISSIONS"); do
    # JOB_NAME="${JOB_NAME_PREFIX}-run${i}-$(date +%Y%m%d-%H%M%S)-${RANDOM}"
    JOB_NAME="${JOB_NAME_PREFIX}"
    timestamped_echo "Submitting job ${i}/${NUM_SUBMISSIONS}: ${JOB_NAME}"

    # Construct the dlc submit command
    dlc submit pytorchjob \
        --name=${JOB_NAME} \
        --command="${JOB_COMMAND}" \
        --data_sources=${DATA_SOURCES} \
        --resource_id=${RESOURCE_ID} \
        --workspace_id=${WORKSPACE_ID} \
        --priority=${PRIORITY} \
        --driver=${DRIVER_VERSION} \
        --workers=${NNODES} \
        --worker_image=${WORKER_IMAGE} \
        --worker_cpu=$((N_GPUS_PER_WORKER_NODE * 16)) \
        --worker_memory=$((N_GPUS_PER_WORKER_NODE * WORKER_MEMORY))Gi \
        --worker_shared_memory=$((N_GPUS_PER_WORKER_NODE * WORKER_SHARED_MEMORY))Gi \
        --worker_gpu=${N_GPUS_PER_WORKER_NODE} \
        --oversold_type=${OVERSOLD_TYPE}



    # Add any passthrough arguments
    if [ ${#DLC_PASSTHROUGH_ARGS[@]} -gt 0 ]; then
        DLC_COMMAND+=("${DLC_PASSTHROUGH_ARGS[@]}")
    fi

    timestamped_echo "Executing: ${DLC_COMMAND[@]}"

    # Execute the command
    if "${DLC_COMMAND[@]}"; then
        timestamped_echo "Successfully submitted job: ${JOB_NAME}"
    else
        timestamped_echo "ERROR: Failed to submit job: ${JOB_NAME}"
        # Decide if you want to continue or exit on failure
        # exit 1 # Uncomment to exit immediately on first failure
    fi

    # if [[ "$i" -lt "$NUM_SUBMISSIONS" ]]; then
    #     SLEEP_DURATION=$((RANDOM % 10 + 5)) # Sleep for 5-14 seconds between submissions
    #     timestamped_echo "Sleeping for ${SLEEP_DURATION} seconds before next submission..."
    #     sleep "$SLEEP_DURATION"
    # fi
done

timestamped_echo "All $NUM_SUBMISSIONS job submissions attempted."
timestamped_echo "Script finished."