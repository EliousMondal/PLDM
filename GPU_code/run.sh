#!/bin/bash
#SBATCH -p h100
##SBATCH -x bhd0005
#SBATCH --gres=gpu 
##SBATCH --mem=10gb
#SBATCH --job-name=ρij1_tr               # create a name for your job
#SBATCH --ntasks=10                      # total number of tasks
#SBATCH --cpus-per-task=1               # cpu-cores per task
#SBATCH --mem-per-cpu=1G               # memory per cpu-core
#SBATCH -t 08:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --output=rho_ij_1.out
#SBATCH --error=rho_ij_1.err

# python operatorDynamics.py Data/

mpiexec -n 10 python -m mpi4py operatorDynamics.py Data/