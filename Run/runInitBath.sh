#!/bin/bash
#SBATCH -p action
#SBATCH --job-name=iR_iP   # create a name for your job
#SBATCH --ntasks=100               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=5G         # memory per cpu-core
#SBATCH -t 5-00:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=iRP.out
#SBATCH --error=iRP.err

srun python initBathMPI.py Data/