#!/bin/bash

# Only add the path to your checkpoint in the docker container 
# The email of the leader same as the one on codalab leaderboard
python3 infere.py --batch-size 1 \
                  --model-dir "/mnt/models/" \
                  --leader-codalab-email "hes.mamdouh@nu.edu.eg" \
                  --result-save-path "/mnt/submissions/" \
                  --data-path "/mnt/data/val.jsonl" \