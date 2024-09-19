#!/bin/bash
#SBATCH --job-name=testjob_jenskinch          # Job name
#SBATCH --partition=hep   # Partition name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=5                     # Number of MPI tasks (total processes)
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --output=mpi_output_%j.out     # Standard output and error log

# Load Conda (if required by the system)

# Activate your specific Conda environment
# or
source /groups/hep/kinch/miniconda3/etc/profile.d/conda.sh
conda activate base    # For Conda 4.4+ (this is the most common)

# Ensure that Python and mpi4py are available in your environment
which python                  # Check the Python interpreter in the environment
which mpiexec                  # Check the MPI installation

# Run the Python MPI code using mpiexec
mpiexec -n 5 python pair_gen_low_transfer.py

# Alternatively, use srun instead of mpiexec
# srun python my_mpi_program.py
