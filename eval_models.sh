#!/usr/bin/env bash

python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_randomChunk_sample3_chunksize1000_CNN | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_randomChunk_sample3_chunksize1000_CNN --overlap True | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_randomChunk_sample3_chunksize2000_CNN | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_randomChunk_sample3_chunksize2000_CNN --overlap True | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_windowChunk_sample3_chunksize1000_CNN | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_windowChunk_sample3_chunksize1000_CNN --overlap True | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_windowChunk_sample1_chunksize2000_CNN | tee -a ./logs/10:28/eval_logs.txt
sleep 5
python eval.py --model_name 20191028/Similarity_batch16_lr0.001_midialigned_s_windowChunk_sample1_chunksize2000_CNN --overlap True | tee -a ./logs/10:28/eval_logs.txt