#!/usr/bin/env bash
# Continue the verified SC physical-mouth pipeline without treating a failed
# collection or malformed dataset as success.  Run this on the TACC login node.
# It records every transition in the datagen root for morning inspection.

set -eo pipefail

readonly base=/work2/11590/satya_a/stampede3
readonly source_root="${SC_MOUTH_REPO_ROOT:-${base}/aic-sc-mouth-pose-src-20260727}"
readonly data_root="${SC_MOUTH_JOB_ROOT:-${base}/aic-sc-mouth-pose-datagen-20260727}"
readonly state_log="${data_root}/OVERNIGHT_ORCHESTRATOR.log"
readonly state_file="${data_root}/OVERNIGHT_ORCHESTRATOR_STATE.txt"
readonly training_partition="${SC_MOUTH_TRAIN_PARTITION:-h100}"
readonly training_workers="${SC_MOUTH_TRAIN_WORKERS:-4}"
readonly smoke_job="${1:?pass the already-submitted smoke job id}"
readonly first_full_job="${2:-}"
# Set to 51 after batches 1..50 are already complete.  The data is append-only
# by global trial index, so the resume path verifies what exists and submits
# only the missing ranges.
readonly resume_start="${SC_MOUTH_RESUME_START:-1}"

stamp() { date -Is; }
# Send progress to stderr so command substitutions that return Slurm job IDs do
# not accidentally capture their own log line as part of the ID.
record() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "${state_log}" >&2; }
fail() {
  record "FAILED $*"
  printf 'FAILED %s at %s\n' "$*" "$(stamp)" >"${state_file}"
  exit 1
}
complete() {
  record "COMPLETE $*"
  printf 'COMPLETE %s at %s\n' "$*" "$(stamp)" >"${state_file}"
}

job_state() {
  local job="$1" queued state
  queued=$(squeue -h -j "${job}" -o '%T' 2>/dev/null | head -1 || true)
  if [[ -n "${queued}" ]]; then
    printf '%s\n' "${queued}"
    return
  fi
  state=$(sacct -X -n -P -j "${job}" -o State 2>/dev/null | head -1 | cut -d'|' -f1 | tr -d '[:space:]')
  printf '%s\n' "${state:-UNKNOWN}"
}

wait_for_job() {
  local job="$1" last='' state
  while true; do
    state=$(job_state "${job}")
    if [[ "${state}" != "${last}" ]]; then
      record "job=${job} state=${state}"
      last="${state}"
    fi
    case "${state}" in
      COMPLETED) return 0 ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|PREEMPTED)
        fail "job=${job} terminal_state=${state}"
        ;;
    esac
    sleep 60
  done
}

image_count() { find "$1/images" -type f -name '*.png' 2>/dev/null | wc -l; }
label_count() { find "$1/labels" -type f -name '*.txt' 2>/dev/null | wc -l; }
bad_label_lines() {
  find "$1/labels" -type f -name '*.txt' -exec awk 'NF != 20 {bad++} END {print bad+0}' {} + 2>/dev/null \
    | awk '{sum+=$1} END {print sum+0}'
}

verify_dataset() {
  local root="$1" minimum_total="$2" minimum_split="$3" images labels malformed split split_images split_labels
  images=$(image_count "${root}")
  labels=$(label_count "${root}")
  malformed=$(bad_label_lines "${root}")
  record "verify root=${root} images=${images} labels=${labels} malformed_lines=${malformed}"
  [[ "${images}" -ge "${minimum_total}" ]] || fail "dataset_underfilled root=${root} images=${images} minimum=${minimum_total}"
  [[ "${images}" -eq "${labels}" ]] || fail "image_label_mismatch root=${root} images=${images} labels=${labels}"
  [[ "${malformed}" -eq 0 ]] || fail "malformed_labels root=${root} lines=${malformed}"
  test -f "${root}/aic_sc_mouth_pose.yaml" || fail "missing_dataset_yaml root=${root}"
  if (( minimum_split > 0 )); then
    for split in train val test; do
      split_images=$(find "${root}/images/${split}" -type f -name '*.png' 2>/dev/null | wc -l)
      split_labels=$(find "${root}/labels/${split}" -type f -name '*.txt' 2>/dev/null | wc -l)
      record "verify split=${split} images=${split_images} labels=${split_labels}"
      [[ "${split_images}" -ge "${minimum_split}" ]] || fail "split_underfilled split=${split} images=${split_images} minimum=${minimum_split}"
      [[ "${split_images}" -eq "${split_labels}" ]] || fail "split_image_label_mismatch split=${split}"
    done
  fi
}

submit_full_batch() {
  local start="$1" output job
  cd -- "${source_root}"
  output=$(SC_MODE=full SC_MOUTH_TRIAL_START="${start}" SC_MOUTH_TRIAL_COUNT=25 \
    sbatch --parsable .tacc/sc_mouth_pose_datagen.slurm)
  # TACC prepends a human-readable allocation banner even with --parsable.
  # Accept only the final bare job-id line, never the banner's quota decimals.
  job=$(printf '%s\n' "${output}" | awk '/^[0-9]+(;[0-9]+)?$/ {sub(/;.*/, ""); id=$0} END {print id}')
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "could_not_parse_full_job_id start=${start} output=${output}"
  record "submitted_full start=${start} job=${job}"
  printf '%s\n' "${job}"
}

submit_training() {
  local output job
  cd -- "${source_root}"
  output=$(SC_MOUTH_TRAIN_WORKERS="${training_workers}" sbatch --parsable \
    --partition="${training_partition}" .tacc/train_sc_mouth_pose.slurm)
  job=$(printf '%s\n' "${output}" | awk '/^[0-9]+(;[0-9]+)?$/ {sub(/;.*/, ""); id=$0} END {print id}')
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "could_not_parse_training_job_id output=${output}"
  record "submitted_training job=${job}"
  printf '%s\n' "${job}"
}

mkdir -p "${data_root}/logs" "${data_root}/training/logs"
record "START smoke_job=${smoke_job} first_full_job=${first_full_job:-none} resume_start=${resume_start}"
if (( resume_start <= 1 )); then
  [[ "${first_full_job}" =~ ^[0-9]+$ ]] || fail "missing_first_full_job_for_fresh_run"
  wait_for_job "${smoke_job}"
  verify_dataset "${data_root}/smoke_dataset/${smoke_job}" 300 0
  wait_for_job "${first_full_job}"
  verify_dataset "${data_root}/dataset" 600 0
  previous_images=$(image_count "${data_root}/dataset")
  record "full_batch_complete start=1 images=${previous_images}"
  starts=(26 51 76)
else
  verify_dataset "${data_root}/smoke_dataset/${smoke_job}" 300 0
  verify_dataset "${data_root}/dataset" 600 0
  previous_images=$(image_count "${data_root}/dataset")
  record "RESUME existing_images=${previous_images} next_start=${resume_start}"
  case "${resume_start}" in
    26) starts=(26 51 76) ;;
    51) starts=(51 76) ;;
    76) starts=(76) ;;
    *) fail "unsupported_resume_start=${resume_start}" ;;
  esac
fi

for start in "${starts[@]}"; do
  job=$(submit_full_batch "${start}")
  wait_for_job "${job}"
  current_images=$(image_count "${data_root}/dataset")
  delta=$((current_images - previous_images))
  record "full_batch_complete start=${start} images=${current_images} delta=${delta}"
  (( delta >= 600 )) || fail "full_batch_underfilled start=${start} delta_images=${delta}"
  verify_dataset "${data_root}/dataset" "${current_images}" 0
  previous_images="${current_images}"
done

verify_dataset "${data_root}/dataset" 3000 300
training_job=$(submit_training)
wait_for_job "${training_job}"
test -s "${data_root}/training/best_sc_mouth_pose.pt" || fail "training_missing_checkpoint"
test -s "${data_root}/training/reports/validate_sc_mouth_pose_test.json" || fail "training_missing_heldout_report"
complete "training_job=${training_job} checkpoint_and_heldout_report_present"
