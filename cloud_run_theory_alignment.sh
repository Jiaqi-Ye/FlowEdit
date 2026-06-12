#!/usr/bin/env bash
set -euo pipefail

EXP_YAML="${1:-SD3_theory_alignment_same_nfe.yaml}"
DATASET_YAML="${2:-edits_midpoint_eval.yaml}"
TAG="${3:-theory_alignment}"
LOAD_MODE="${FLOWEDIT_LOAD_MODE:-auto}"

echo "Using experiment YAML: ${EXP_YAML}"
echo "Using dataset YAML: ${DATASET_YAML}"
echo "Using output tag: ${TAG}"
echo "Using pipeline load mode: ${LOAD_MODE}"
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGINGFACE_HUB_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Hugging Face token detected: yes"
else
  echo "Hugging Face token detected: no"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 -m pip install -q \
  diffusers==0.30.1 \
  transformers==4.44.0 \
  accelerate==0.33.0 \
  sentencepiece \
  protobuf \
  huggingface_hub==0.36.2 \
  safetensors

mkdir -p outputs/metrics

RUN_ARGS=(
  --exp_yaml "${EXP_YAML}"
  --pipeline_load_mode "${LOAD_MODE}"
)

if [[ "${FLOWEDIT_SKIP_ACCESS_CHECK:-0}" == "1" ]]; then
  RUN_ARGS+=(--skip_model_access_check)
fi

if [[ "${FLOWEDIT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  RUN_ARGS+=(--preflight_only)
fi

python3 run_script.py "${RUN_ARGS[@]}"

if [[ "${FLOWEDIT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight-only mode finished successfully."
  exit 0
fi

python3 evaluate_clip_dino.py \
  --run_summary_csv outputs/run_summary.csv \
  --dataset_yaml "${DATASET_YAML}" \
  --out_samples "outputs/metrics/${TAG}_clip_dino_per_sample.csv" \
  --out_summary "outputs/metrics/${TAG}_clip_dino_summary.csv"

python3 evaluate_artifact_proxy.py \
  --run_summary_csv outputs/run_summary.csv \
  --out_samples "outputs/metrics/${TAG}_artifact_per_sample.csv" \
  --out_summary "outputs/metrics/${TAG}_artifact_summary.csv"

echo "Cloud run finished."
echo "Main outputs:"
echo "  outputs/run_summary.csv"
echo "  outputs/metrics/${TAG}_clip_dino_summary.csv"
echo "  outputs/metrics/${TAG}_artifact_summary.csv"
