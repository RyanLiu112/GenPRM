#!/bin/bash

# Initialize environment
source $HOME/.bashrc
NEW_HOME=/cpfs02/user/liurunze
eval "$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)"
mount_path=$NEW_HOME
PROJECT_ROOT="${mount_path}/release_version/GenPRM/src"
cd $PROJECT_ROOT

# llmit6
data_sources="d-x135sqzws1argjld1r"
resource_id=quota1bhq0p32wuc
workspace_id=84885
priority=9

# # A-ma4agi
# data_sources="d-egq3ijcv3u5f5xta1q"
# resource_id=quota1ycjvgnso74
# workspace_id=86974
# priority=6

#oversold_type=ForbiddenQuotaOverSold  # 如果不指定，则不使用闲时资源
oversold_type=ForceQuotaOverSold  # 如果设置为1，则只使用闲时资源
#oversold_type=AcceptQuotaOverSold  # 如果设置为2，则可接受使用闲时资源
echo "oversold_type: $oversold_type"

name=$1
idd=$2
tp=$3

analyze_flag="" # 默认不启用
verify_flag=""  # 默认不启用
execute_flag="" # 默认不启用

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

command="/cpfs02/user/liurunze/release_version/GenPRM/src/F1.sh ${name} ${idd} ${tp} ${analyze_flag} ${verify_flag} ${execute_flag}"
echo =============================================
echo command is $command
echo =============================================

echo_only=0
local=0
if [ $echo_only -eq 1 ]; then
    echo $command
    exit
fi
if [ $local -eq 1 ]; then
    bash $command
    exit
fi
worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/liurunze:liurunze-sllm02

N=${!#}

run_name=eval
for ((i=0; i<$N; i++)); do
    ${NEW_HOME}/packages/dlc submit pytorchjob \
        --name="zj_test_${run_name}" \
        --command="bash ${command}" \
        --data_sources=$data_sources \
        --resource_id=$resource_id \
        --workspace_id=$workspace_id \
        --priority=$priority \
        --driver=535.54.03 \
        --job_max_running_time_minutes=0 \
        --workers=1 \
        --worker_image=$worker_image \
        --worker_cpu=$((12 * tp)) \
        --worker_memory=$((200 * tp))Gi \
        --worker_shared_memory=$((200 * tp))Gi \
        --worker_gpu=$tp \
        --oversold_type=$oversold_type
done

