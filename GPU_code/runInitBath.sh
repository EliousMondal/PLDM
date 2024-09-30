#!/bin/bash
#SBATCH -p h100
#SBATCH --job-name=iR_iP   # create a name for your job
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=5G         # memory per cpu-core
#SBATCH -t 1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=iRP.out
#SBATCH --error=iRP.err

srun python initBathMPI.py Data/