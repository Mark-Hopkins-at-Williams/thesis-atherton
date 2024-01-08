#!/bin/sh
#SBATCH -c 1
#SBATCH -t 2-12:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o output_files/myoutput_%j.out
#SBATCH -e output_files/myerrors_%j.out
#SBATCH --gres=gpu:1

alias py=/home/hopkins/anaconda3/envs/mt/bin/python
py train.py