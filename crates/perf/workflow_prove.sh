#! /bin/bash

# Get the current git branch.
GIT_REF=$(git rev-parse --abbrev-ref HEAD)

# Define the list of CPU workloads.
CPU_WORKLOADS=(
    "ssz-withdrawals"
    "loop-10k"
    "fibonacci-20k"
    "keccak256-300kb"
    "sha256-300kb"
    "rsp-20526626"
    "zk-email"
    "eddsa-verify"
)

# Define the list of CUDA workloads.
CUDA_WORKLOADS=(
    "ssz-withdrawals"
    "loop-10k"
    "fibonacci-20k"
    "keccak256-300kb"
    "sha256-300kb"
    "rsp-20526626"
    "zk-email"
    "eddsa-verify"
)

# Define the list of network workloads.
NETWORK_WORKLOADS=(
    "ssz-withdrawals"
    "loop-100m"
    "fibonacci-200m"
    "keccak256-3mb"
    "sha256-3mb"
    "rsp-20526626"
    "zk-email"
    "eddsa-verify"
)

# Create a JSON object with the list of workloads.
WORKLOADS=$(jq -n \
    --arg cpu "$(printf '%s\n' "${CPU_WORKLOADS[@]}" | jq -R . | jq -s 'map(select(length > 0))')" \
    --arg cuda "$(printf '%s\n' "${CUDA_WORKLOADS[@]}" | jq -R . | jq -s 'map(select(length > 0))')" \
    --arg network "$(printf '%s\n' "${NETWORK_WORKLOADS[@]}" | jq -R . | jq -s 'map(select(length > 0))')" \
    '{cpu_workloads: $cpu, cuda_workloads: $cuda, network_workloads: $network}')

# Run the workflow with the list of workloads.
echo $WORKLOADS | gh workflow run suite.yml --ref $GIT_REF --json
