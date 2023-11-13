#!/bin/sh
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p dl
#SBATCH --mem=10G
#SBATCH -o output_files/myoutput_%j.out
#SBATCH -e output_files/myerrors_%j.out
#SBATCH --gres=gpu:1

conda create --name=ENV_NAME python=3.10.4
conda activate ENV_NAME
conda install pytorch torchvision torchaudio torchtext torchdata pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -U pydantic spacy
pip install 'portalocker>=2.0.0'
pip install evaluate

python train.py