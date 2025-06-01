# Default arguments
LM=models--Qwen--Qwen2.5-Math-7B-Instruct
ORIGIN=models--Qwen--Qwen2.5-Math-7B-Instruct
RM=dummy
data_name=MATH
round=0
#start_idx=0
#end_idx=500
eager=0
batch_size=1
max_time=20
n_gpus=1
loop=0
echo_only=0
N=1
local=0
wo_check=0

params=$@
# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --LM)
        LM="$2"
        shift 2
        ;;
    --ORIGIN)
        ORIGIN_MODEL_PATH="$2"
        shift 2
        ;;
    --RM)
        RM="$2"
        shift 2
        ;;
    --task)
        data_name="$2"
        shift 2
        ;;
    --width)
        tree_max_width="$2"
        shift 2
        ;;
    --num_seq)
        num_sequence="$2"
        shift 2
        ;;
    --num_q)
        question_parallel_num="$2"
        shift 2
        ;;
    --round)
        round="$2"
        shift 2
        ;;
    --start_idx)
        start_idx="$2"
        shift 2
        ;;
    --end_idx)
        end_idx="$2"
        shift 2
        ;;
    --eager)
        eager="$2"
        shift 2
        ;;
    --bs)
        batch_size="$2"
        shift 2
        ;;
    --mt)
        max_time="$2"
        shift 2
        ;;
    --n_gpus)
        n_gpus="$2"
        shift 2
        ;;
    --loop)
        loop="$2"
        shift 2
        ;;
    --echo_only)
        echo_only="$2"
        shift 2
        ;;
    --N)
        N="$2"
        shift 2
        ;;
    --local)
        local="$2"
        shift 2
        ;;
    --wo_check)
        wo_check="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done
echo "LM: $LM, RM: $RM, task: $task_name, tree_max_width: $tree_max_width, num_sequence: $num_sequence"
echo "question_parallel_num: $question_parallel_num, round: $round, levels: $levels, start_idx: $start_idx, end_idx: $end_idx"
echo "eager: $eager, batch_size: $batch_size, max_time: $max_time, n_gpus: $n_gpus, loop: $loop, echo_only: $echo_only"

POLICY_MODEL_PATH=${NEW_HOME}/hf_models/${LM}
VALUE_MODEL_PATH=${NEW_HOME}/hf_models/${RM}

SCRIPT_FULL_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_FULL_PATH")"
PARENT_DIR_OF_SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PARENT_DIR_OF_SCRIPT_DIR}"
cd ${PYTHONPATH}

export CUDA_VISIBLE_DEVICES=0
GPU_LIST=(0 0)
if [ $n_gpus -eq 0 ]; then
    echo "Abandon GPU"
else
    if [ $n_gpus -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
        GPU_LIST=(0 1)
    elif [ $n_gpus -eq 3 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2
        GPU_LIST=(0 1 2)
    fi
    n_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, n_gpus: $n_gpus"
echo "GPU_LIST:"
echo "${GPU_LIST[@]}"

HOST_ADDR=0.0.0.0
CONTROLER_PORT=10014
WORKER_BASE_PORT=10081
controller_addr=http://${HOST_ADDR}:10014
echo "controller_ip: $HOST_ADDR, controller_addr: $controller_addr"

temperature=0.7
max_new_tokens=2048
num_worker=8
#save_dir=${NEW_HOME}/main_branch/GenPRM/output

LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR

if [ $n_gpus -eq 0 ]; then
    echo "Abandon GPU"
elif [ $n_gpus -eq 1 ]; then
    bash ${PYTHONPATH}/reward_generation/math_shepherd/vllm_fc_serve_1_v2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLER_PORT $WORKER_BASE_PORT $LOGDIR
elif [ $n_gpus -eq 2 ]; then
    bash ${PYTHONPATH}/reward_generation/math_shepherd/vllm_fc_serve_2_v2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLER_PORT $WORKER_BASE_PORT $LOGDIR
elif [ $n_gpus -eq 3 ]; then
    bash ${PYTHONPATH}/reward_generation/math_shepherd/vllm_fc_serve_3_v2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLER_PORT $WORKER_BASE_PORT $LOGDIR
else
    echo "Invalid n_gpus: $n_gpus"
    exit
fi
if [ $n_gpus -gt 0 ]; then
    echo "Wait 50 seconds ..."
    sleep 50
fi

for split in train; do
    cnt=0
    if [ "$loop" == "1" ]; then
        echo "Running ..."

        # echo "Running $method evaluation original command ..."

        echo "========================================"
        echo "Processing data_name: ${split}"
        echo "========================================"
        
        python reward_generation/math_shepherd_ray.py \
            --model_path "$POLICY_MODEL_PATH" \
            --origin_model_path $ORIGIN_MODEL_PATH \
            --split $split \
            --data_name $data_name \
            --num_paths 4 \
            --num_worker "$num_worker" \
            --controller_addr "$controller_addr" \
            --add_step_prompt \
            --round "$round" \
            --eager "$eager" \
            --batch_size "$batch_size" \
            --max_time "$max_time" \
            --only_monte_carlo
            # --close_monte_carlo


        cnt=$((cnt+1))
        if [ $cnt -ge 100 ]; then
            echo "Looping 100 times, exit."
            break
        fi
        if [ $exit_code -eq 0 ]; then
            echo "$exit_code, split finished"
            break
        fi
    fi
done

# bash reward_generation/mt_score_generate.sh --LM models--Qwen--Qwen2.5-Math-7B-Instruct --ORIGIN models--Qwen--Qwen2.5-Math-7B-Instruct --round 0 --bs 4 --mt 6000 --n_gpus 1 --task math --loop 1
