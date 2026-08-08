#!/usr/bin/env bash
# Finish the physical SC-mouth dataset with three isolated simulators.  The
# ranges are disjoint, so the collector's t<global-trial> filenames and its
# deterministic train/val/test split remain collision-free.  This is intended
# only after trials 1..75 have passed their integrity gates.

set -eo pipefail

readonly base=/work2/11590/satya_a/stampede3
readonly source_root="${SC_MOUTH_REPO_ROOT:-${base}/aic-sc-mouth-pose-src-20260727}"
readonly data_root="${SC_MOUTH_JOB_ROOT:-${base}/aic-sc-mouth-pose-datagen-20260727}"
readonly dataset_root="${data_root}/dataset"
readonly state_log="${data_root}/PARALLEL_FINAL_ORCHESTRATOR.log"
readonly state_file="${data_root}/PARALLEL_FINAL_ORCHESTRATOR_STATE.txt"
readonly training_partition="${SC_MOUTH_TRAIN_PARTITION:-rtx-small}"
readonly training_workers="${SC_MOUTH_TRAIN_WORKERS:-4}"

stamp() { date -Is; }
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
  [[ "${images}" -ge "${minimum_total}" ]] || fail "dataset_underfilled images=${images} minimum=${minimum_total}"
  [[ "${images}" -eq "${labels}" ]] || fail "image_label_mismatch images=${images} labels=${labels}"
  [[ "${malformed}" -eq 0 ]] || fail "malformed_labels lines=${malformed}"
  test -f "${root}/aic_sc_mouth_pose.yaml" || fail "missing_dataset_yaml"
  for split in train val test; do
    split_images=$(find "${root}/images/${split}" -type f -name '*.png' 2>/dev/null | wc -l)
    split_labels=$(find "${root}/labels/${split}" -type f -name '*.txt' 2>/dev/null | wc -l)
    record "verify split=${split} images=${split_images} labels=${split_labels}"
    [[ "${split_images}" -ge "${minimum_split}" ]] || fail "split_underfilled split=${split} images=${split_images} minimum=${minimum_split}"
    [[ "${split_images}" -eq "${split_labels}" ]] || fail "split_image_label_mismatch split=${split}"
  done
}

parse_job_id() {
  awk '/^[0-9]+(;[0-9]+)?$/ {sub(/;.*/, ""); id=$0} END {print id}'
}

submit_chunk() {
  local start="$1" count="$2" output job
  cd -- "${source_root}"
  output=$(SC_MODE=full SC_MOUTH_TRIAL_START="${start}" SC_MOUTH_TRIAL_COUNT="${count}" \
    sbatch --parsable --partition=rtx-small .tacc/sc_mouth/collect.slurm)
  job=$(printf '%s\n' "${output}" | parse_job_id)
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "could_not_parse_chunk_job start=${start} count=${count} output=${output}"
  record "submitted_chunk start=${start} count=${count} job=${job} partition=rtx-small"
  printf '%s\n' "${job}"
}

submit_training() {
  local output job
  cd -- "${source_root}"
  output=$(SC_MOUTH_TRAIN_WORKERS="${training_workers}" sbatch --parsable \
    --partition="${training_partition}" .tacc/sc_mouth/train.slurm)
  job=$(printf '%s\n' "${output}" | parse_job_id)
  [[ "${job}" =~ ^[0-9]+$ ]] || fail "could_not_parse_training_job output=${output}"
  record "submitted_training job=${job} partition=${training_partition} workers=${training_workers}"
  printf '%s\n' "${job}"
}

wait_for_all() {
  local last_states='' snapshot='' job state remaining
  while true; do
    remaining=0
    snapshot=''
    for job in "$@"; do
      state=$(job_state "${job}")
      snapshot+="${job}:${state} "
      case "${state}" in
        COMPLETED) ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|PREEMPTED)
          fail "job=${job} terminal_state=${state}"
          ;;
        *) remaining=$((remaining + 1)) ;;
      esac
    done
    if [[ "${snapshot}" != "${last_states}" ]]; then
      record "parallel_jobs ${snapshot}"
      last_states="${snapshot}"
    fi
    (( remaining == 0 )) && return 0
    sleep 30
  done
}

mkdir -p "${data_root}/logs" "${data_root}/training/logs"
previous_images=$(image_count "${dataset_root}")
record "START previous_images=${previous_images} chunks=76:8,84:8,92:9 partition=rtx-small"
verify_dataset "${dataset_root}" 2400 0

# Refuse to overlay a partial old final batch.  Every filename embeds its global
# trial number, so this check protects against accidental duplicate trials.
for trial in $(seq 76 100); do
  frame_prefix=$(printf 't%05d_' "${trial}")
  if find "${dataset_root}" -type f -name "${frame_prefix}*" -print -quit | grep -q .; then
    fail "preexisting_final_trial_data trial=${trial}"
  fi
done

# TACC enforces two active RTX-small jobs per user.  Use both slots for the
# first two disjoint chunks, then immediately use the released slot for the
# third.  This is the maximum permitted collection parallelism without
# queuing/cancelling another user's work.
job_a=$(submit_chunk 76 8)
job_b=$(submit_chunk 84 8)
wait_for_all "${job_a}" "${job_b}"
job_c=$(submit_chunk 92 9)
wait_for_all "${job_c}"

current_images=$(image_count "${dataset_root}")
delta=$((current_images - previous_images))
record "parallel_final_complete images=${current_images} delta=${delta} jobs=${job_a},${job_b},${job_c}"
(( delta >= 600 )) || fail "parallel_final_underfilled delta_images=${delta}"
verify_dataset "${dataset_root}" 3000 300

training_job=$(submit_training)
wait_for_all "${training_job}"
test -s "${data_root}/training/best_sc_mouth_pose.pt" || fail "training_missing_checkpoint"
test -s "${data_root}/training/reports/validate_sc_mouth_pose_test.json" || fail "training_missing_heldout_report"
complete "parallel_jobs=${job_a},${job_b},${job_c} training_job=${training_job} checkpoint_and_heldout_report_present"
