#!/bin/bash

# go the directionary where the script is located
#cd /workspace/ljl/Junle_PIV_data/ || exit 1

# clear cache
rm -rf LatentFlow/models/__pycache__ LatentFlow/utils/__pycache__ LatentFlow/__pycache__ 

# save the log
# LOG_FILE=LatentFlow/train_$(date +%Y%m%d_%H%M%S).log

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=LatentFlow/train_p2z_new_loss_${TIMESTAMP}.log


# Start training and save output to log file

nohup python3 -m LatentFlow.utils.train_p2z > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "p2z Training started with PID $TRAIN_PID"
wait "$TRAIN_PID"

# find the latest CAVE save dir
SAVE_DIR=$(find LatentFlow/results/ -maxdepth 1 -type d -name "CVAE_*" | sort -V | tail -n 1)
# mkdir -p "$SAVE_DIR/new_loss"

# copy files to the latest save directory
if [ -d "$SAVE_DIR" ]; then
    cp LatentFlow/utils/train_p2z.py "$SAVE_DIR/new_loss/train_p2z_new_sha.py"
    cp LatentFlow/models/p2z.py "$SAVE_DIR/new_loss/p2z_new_sha.py"
    cp LatentFlow/utils/para_set.py "$SAVE_DIR/new_loss/para_set_p2z_new_sha.py"
    cp run_train_p2z.sh "$SAVE_DIR/new_loss/run_train_p2z_new_sha.sh"
    mv "$LOG_FILE" "$SAVE_DIR/new_loss/train_p2z_${TIMESTAMP}.log"
    echo "train_p2z.py, para_set.py and log files are copied to $SAVE_DIR/new_loss"
    
else
    echo "did not get the dir (results/cvae)"
fi
