#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --account=csedui00041
#SBATCH --output=my-experiment-%j.out
#SBATCH --error=my-experiment-%j.err
#SBATCH --mail-user=daniel.groenendijk@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL



