#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

data_root="${repo_root}/data"
log_root="${repo_root}/logs"
checkpoint_root="${repo_root}/checkpoints"
mkdir -p "${log_root}" "${checkpoint_root}"
n_gpus_per_node=8

ts="$(date +%m%d_%H%M)"
run_name="NFPO-ForwardTrace-Mask0.2-N8-RatioClip3-TraceClip0.8-1.2-RollOld-b128-Qwen3-1.7B-Base-math_${ts}"
log_path="${log_root}/${run_name}.log"
ckpt_dir="${checkpoint_root}/${run_name}"
model_repo_id="Qwen/Qwen3-1.7B-Base"
model_path="${HOME}/models/Qwen/Qwen3-1.7B-Base"
train_files="['${data_root}/math/train.parquet']"
test_files="['${data_root}/amc23/train.parquet', '${data_root}/aime2024/train.parquet', '${data_root}/aime2025/train.parquet', '${data_root}/aime2026/train.parquet']"

ensure_local_model_snapshot() {
  local repo_id="$1"
  local local_dir="$2"

  if [[ -f "${local_dir}/config.json" ]]; then
    echo "Using cached model at ${local_dir}"
    return 0
  fi

  mkdir -p "${local_dir}"
  python3 - "${repo_id}" "${local_dir}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = os.path.expanduser(sys.argv[2])

snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(local_dir)
PY
}

ensure_local_model_snapshot "${model_repo_id}" "${model_path}"

require_paths() {
  local missing=0
  local path

  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required path: ${path}" >&2
      missing=1
    fi
  done

  if (( missing )); then
    echo "Prepare the data or edit data_root/model_path in this script and retry." >&2
    exit 1
  fi
}

require_paths \
  "${data_root}/math/train.parquet" \
  "${data_root}/amc23/train.parquet" \
  "${data_root}/aime2024/train.parquet" \
  "${data_root}/aime2025/train.parquet" \
  "${data_root}/aime2026/train.parquet" \
  "${model_path}"

python3 -u -m verl.trainer.main_ppo \
  data.train_files="$train_files" \
  data.val_files="$test_files" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=8000 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.model.path="${model_path}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.clip_ratio_c=10000 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.policy_loss.loss_mode=nfpo \
  actor_rollout_ref.actor.policy_loss.mask_delta=0.2 \
  actor_rollout_ref.actor.policy_loss.n_step_forward_trace=8 \
  actor_rollout_ref.actor.policy_loss.forward_trace_ratio_clip=3 \
  actor_rollout_ref.actor.policy_loss.forward_trace_lower=0.8 \
  actor_rollout_ref.actor.policy_loss.forward_trace_upper=1.2 \
  actor_rollout_ref.actor.policy_loss.forward_trace_use_rollout_old_log_probs=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_model_len=9024 \
  actor_rollout_ref.rollout.max_num_batched_tokens=9024 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  trainer.logger='["console","file"]' \
  trainer.project_name="verl-nfpo" \
  trainer.experiment_name="${run_name}" \
  trainer.default_local_dir="${ckpt_dir}" \
  trainer.resume_mode=disable \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node=${n_gpus_per_node} \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=25 \
  trainer.total_training_steps=500 \
  2>&1 | tee -a "${log_path}"

echo "LOG: ${log_path}"
echo "CKPT_DIR: ${ckpt_dir}"
