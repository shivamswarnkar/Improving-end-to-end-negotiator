#!/bin/bash
#
#SBATCH --job-name=testingCuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --gres=gpu:1

module purge
module load pytorch/python3.6/gnu/20171124
cd /home/vvb231/nrgo2/end-to-end-negotiator/src
python reinforce.py \
  --data data/negotiate \
  --bsz 16 \
  --clip 1 \
  --context_file data/negotiate/selfplay.txt \
  --eps 0.0 \
  --gamma 0.95 \
  --lr 0.5 \
  --momentum 0.1 \
  --nepoch 4 \
  --nesterov \
  --ref_text data/negotiate/train.txt \
  --rl_clip 1 \
  --rl_lr 0.2 \
  --score_threshold 6 \
  --sv_train_freq 4 \
  --temperature 0.5 \
  --alice_model sv_model.th \
  --bob_model sv_model.th \
  --output_model_file rl_model.th