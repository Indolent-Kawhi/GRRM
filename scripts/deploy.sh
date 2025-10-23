#!/bin/bash

MODEL=$1
NAME=$2
CUDA_DEVICES=${3:-"0,1,2,3"}  # Default GPU devices
TP_SIZE=${4:-1}            # Default TP size

# Store all background process PIDs
PIDS=()

# Signal handling function
cleanup() {
    echo "Stopping all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill -TERM "$pid"
        fi
    done
    
    # Wait for processes to exit
    sleep 2
    
    # Force kill any remaining running processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -KILL "$pid"
        fi
    done
    
    exit 0
}

# Set signal handling
trap cleanup SIGINT SIGTERM

# Split CUDA_DEVICES by comma into array
IFS=',' read -ra DEVICE_ARRAY <<< "$CUDA_DEVICES"
NUM_DEVICES=${#DEVICE_ARRAY[@]}

# Check if number of devices is a multiple of TP_SIZE
if (( NUM_DEVICES % TP_SIZE != 0 )); then
    echo "Error: Number of CUDA devices ($NUM_DEVICES) is not a multiple of TP_SIZE ($TP_SIZE)."
    exit 1
fi

# Initial port number
PORT=10010
NUM_SERVERS=$((NUM_DEVICES / TP_SIZE))

# Iterate and start servers
for ((i=0; i<NUM_SERVERS; i++)); do
    # Calculate device group for current server
    START_INDEX=$((i * TP_SIZE))
    GROUP_DEVICES=()
    for ((j=0; j<TP_SIZE; j++)); do
        GROUP_DEVICES+=(${DEVICE_ARRAY[START_INDEX+j]})
    done
    
    # Convert device array to comma-separated string
    VISIBLE_DEVICES=$(IFS=,; echo "${GROUP_DEVICES[*]}")
    
    # Use first device ID in group to calculate port for uniqueness
    FIRST_DEVICE_IN_GROUP=${GROUP_DEVICES[0]}
    CURRENT_PORT=$((PORT + FIRST_DEVICE_IN_GROUP))
    
    echo "Starting server for group $((i+1)) on GPUs [$VISIBLE_DEVICES] with port $CURRENT_PORT (TP_SIZE=$TP_SIZE)"
    
    # Prepare command
    CMD="CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.85 \
        --served-model-name $NAME \
        --tensor-parallel-size $TP_SIZE \
        --port $CURRENT_PORT"

    # Last server runs in foreground, others in background
    if (( i < NUM_SERVERS - 1 )); then
        eval "$CMD &"
        PIDS+=($!)
    else
        eval "$CMD"
        PIDS+=($!)
    fi
done

# Wait for all background processes
wait