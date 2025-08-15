#!/bin/bash

# go the directionary where the script is located
#cd /workspace/ljl/Junle_PIV_data/ || exit 1

# clear cache
rm -rf LatentFlow/models/__pycache__ LatentFlow/utils/__pycache__ LatentFlow/__pycache__ 

# save the log
# LOG_FILE=LatentFlow/train_$(date +%Y%m%d_%H%M%S).log

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=LatentFlow/train_${TIMESTAMP}.log


# Start training and save output to log file

nohup python3 -m LatentFlow.utils.train_cvae > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"
wait "$TRAIN_PID"

# find the latest CAVE save dir
SAVE_DIR=$(find LatentFlow/results/ -maxdepth 1 -type d -name "CVAE_*" | sort -V | tail -n 1)

# copy files to the latest save directory
if [ -d "$SAVE_DIR" ]; then
    cp LatentFlow/utils/train_cvae.py "$SAVE_DIR/train_cvae.py"
    cp LatentFlow/utils/para_set.py "$SAVE_DIR/para_set.py"
    mv "$LOG_FILE" "$SAVE_DIR/train_log_${TIMESTAMP}.log"
    echo "train_cvae.py, para_set.py and log files are copied to $SAVE_DIR"
    
else
    echo "did not get the dir (results/cvae)"
fi
