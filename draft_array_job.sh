#!/bin/bash
#SBATCH --job-name=tts-v3-k16
#SBATCH --output=test_time_search_v3/outputs/slurm/%x-%A_%a.out
#SBATCH --error=test_time_search_v3/outputs/slurm/%x-%A_%a.err
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-28%12