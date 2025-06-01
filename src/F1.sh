#!/bin/bash

# Initialize environment
source $HOME/.bashrc
# unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
NEW_HOME=/cpfs02/user/liurunze
eval "$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)"
which conda
conda activate zj_GenPRM
which python

model_path=${NEW_HOME}/hf_models

#================================================================================>

export PYTHONPATH=${NEW_HOME}/release_version/GenPRM/src
cd ${PYTHONPATH}

# cd prm_evaluation

name=$1
idd=$2
tp=$3

analyze_flag="" # 默认不启用
verify_flag=""  # 默认不启用
execute_flag="" # 默认不启用

analyze_flag="--analyze"
verify_flag="--verify"
execute_flag="--execute"

# # 检查参数3及之后
# for arg in "${@:3}"; do
#     case $arg in
#         --analyze)
#             analyze_flag="--analyze"
#             ;;
#         --verify)
#             verify_flag="--verify"
#             ;;
#         --execute)
#             execute_flag="--execute"
#             ;;
#     esac
# done

# judge if 'v2' in name
if [[ $name == *"v4"* ]]; then
    echo "v4"
    DIR=$NEW_HOME/release_version/GenPRM/src/_output_model
elif [[ $name == *"v2"* ]]; then
    echo "v2"
    DIR=$NEW_HOME/main_branch/GenPRM/_outputs/axo_model_v2
else
    echo "not v2"
    DIR=$NEW_HOME/main_branch/GenPRM/_outputs/axo_model
fi

python prm_evaluation/prm_evaluate.py \
    --reward_name_or_path $DIR/$name \
    --data_path $NEW_HOME/release_version/GenPRM/src/_data/split_input/ProcessBench \
    --split_out $NEW_HOME/release_version/GenPRM/src/_output/split_output/$name/ProcessBench-$idd \
    --idd $idd \
    --tensor_parallel_size $tp \
    $analyze_flag \
    $verify_flag \
    $execute_flag \
