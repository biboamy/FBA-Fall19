#!/usr/bin/env bash

python train.py --midi_op aligned_s | tee -a logs/10:28/train_logs.txt
sleep 5
python train.py --midi_op aligned_s --chunk_size 2000 --sample_num 1 | tee -a ./logs/10:28/train_logs.txt
sleep 5
python train.py --midi_op aligned_s --process_collate windowChunk | tee -a ./logs/10:28/train_logs.txt
sleep 5
python train.py --midi_op aligned_s --process_collate windowChunk --chunk_size  2000 --sample_num 1 | tee -a ./logs/10:28/train_logs.txt
