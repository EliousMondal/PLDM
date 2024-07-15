#!/bin/bash
#SBATCH -p action
#SBATCH -x bhd0005
#SBATCH --job-name=ρij              # create a name for your job
#SBATCH --ntasks=1                  # total number of tasks
#SBATCH --cpus-per-task=1           # cpu-cores per task
#SBATCH --mem-per-cpu=1G            # memory per cpu-core
#SBATCH -t 5-00:00:00               # total run time limit (HH:MM:SS)
#SBATCH --output=rho_ij.out
#SBATCH --error=rho_ij.err

srun python operatorDynamics.py Data/