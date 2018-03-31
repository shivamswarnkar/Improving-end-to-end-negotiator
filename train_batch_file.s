#!/bin/bash
#
#SBATCH --job-name=testingCuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --gres=gpu:1

module purge
module load pytorch/python3.6/gnu/20171124
cd /home/vvb231/nrgo2/end-to-end-negotiator-master/src
python train.py \
  --data data/negotiate \
  --cuda \
  --bsz 16 \
  --clip 0.5 \
  --decay_every 1 \
  --decay_rate 5.0 \
  --dropout 0.5 \
  --init_range 0.1 \
  --lr 1 \
  --max_epoch 30 \
  --min_lr 0.01 \
  --momentum 0.1 \
  --nembed_ctx 64 \
  --nembed_word 256 \
  --nesterov \
  --nhid_attn 256 \
  --nhid_ctx 64 \
  --nhid_lang 128 \
  --nhid_sel 256 \
  --nhid_strat 128 \
  --sel_weight 0.5 \
  --model_file sv_model.th
