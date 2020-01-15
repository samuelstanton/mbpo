#!/bin/bash

while getopts e:v:t:d: option
do
  case "${option}"
  in
  e) env=${OPTARG};;
  v) variant=${OPTARG};;
  t) trial_num=${OPTARG};;
  d) device=${OPTARG};;
  esac
done

exp_name="mbpo/${env}/${variant}/trial_${trial_num}"
echo ${exp_name}
conda activate root
tmux new -d -s "${exp_name}"
tmux send-keys -t "${exp_name}" "conda activate mbpo" Enter
tmux send-keys -t "${exp_name}" "export OPT_NUM_THREADS=4" Enter
tmux send-keys -t "${exp_name}" "export CUDA_VISIBLE_DEVICES=${device}" Enter
launch_cmd="mbpo run_local examples.development --config=examples.config.${env}.${variant} --gpus=1 --trial-gpus=1"
tmux send-keys -t "$exp_name" "$launch_cmd" Enter

#printf "Enter an experiment name: "
#read -r exp_name
#mkdir -p "experiments/mbpo/${exp_name}"
#printf "Enter any experiment notes.\n"
#read -r exp_notes
#logfile="experiments/mbpo/${exp_name}/log.txt"
#current_date=$(date)
#echo "${current_date}" > "$logfile"
#git log --pretty=format:'%h' -n 1 >> "$logfile"
#echo -e "\n${env} \nNotes: ${exp_notes}" >> "$logfile"
#
#for i in $(seq 1 ${num_trials})
#do
#    trial_name="${exp_name}/trial_$i"
#    conda activate root
#    tmux new -d -s "$trial_name"
#    tmux send-keys -t "${trial_name}" "conda activate mbpo_pytorch" Enter
#    tmux send-keys -t "${trial_name}" "export CUDA_VISIBLE_DEVICES=${device}" Enter
#
#    tmux send-keys -t "$trial_name" "$launch_cmd" Enter
#done
