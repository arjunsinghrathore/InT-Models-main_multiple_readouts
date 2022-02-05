#!/bin/bash
#SBATCH --time=192:00:00
#SBATCH -p gpu --gres=gpu:4
#SBATCH -n 1
#SBATCH -N 1 
#SBATCH --mem=30GB
#SBATCH -J pathtracker_mot
#SBATCH -C quadrortx
##SBATCH --constraint=v100
#SBATCH -o /users/aarjun1/data/aarjun1/pathtracker/mot/InT-Models-main_multiple_readouts/logs/MI_%A_%a_%J.out
#SBATCH -e /users/aarjun1/data/aarjun1/pathtracker/mot/InT-Models-main_multiple_readouts/logs/MI_%A_%a_%J.err
#SBATCH --account=carney-tserre-condo
##SBATCH --array=0-1

##SBATCH -p gpu

cd ~/users/aarjun1/data/aarjun1/pathtracker/mot/InT-Models-main_multiple_readouts/

module load anaconda/3-5.2.0
module load python/3.8.8
# module load opencv-python/4.1.0.25
# module load cuda
# module load cudnn/8.1.0
module load cuda/11.1.1

source activate /users/aarjun1/.conda/envs/pytracker

#CUDA_VISIBLE_DEVICES=4,5,6,7 
python -u mainclean.py --print-freq 20 --lr 3e-4 --epochs 2000 -b 180 --model ffhgru_multi --height 32 --width 32 --classes 9 --name int_32_14_occ_5 --log --parallel --length 64 --thickness 3 --dist 14 --ckpt /gpfs/data/tserre/aarjun1/pathtracker/mot/InT-Models-main/results_multi_8_targets_32/64_3_14/int_32_14_occ_5/model_val_acc_0072_epoch_56_checkpoint.pth.tar