#!/bin/bash
#SBATCH --job-name=electron_pairs          # Job name
#SBATCH --partition=hep   # Partition name
#SBATCH --output=elctron_pairs_%j.out     # Standard output and error log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=124G


# Load Conda (if required by the system)

# Activate your specific Conda environment
# or
source /groups/hep/kinch/miniconda3/etc/profile.d/conda.sh
conda activate base    # For Conda 4.4+ (this is the most common)

# Ensure that Python and mpi4py are available in your environment

# Run the Python MPI code using mpiexec
python read_parquet_ttbar.py

# Alternatively, use srun instead of mpiexec
# srun python my_mpi_program.py
