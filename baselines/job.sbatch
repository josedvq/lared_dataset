#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --mail-type=END
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module miniconda/3.8 load cuda/11.8 cudnn/11.5-8.3.0.98

conda activate /tudelft.net/staff-bulk/ewi/insy/SPCLab/jvargasquiros/venv

srun python train_cluster.py
