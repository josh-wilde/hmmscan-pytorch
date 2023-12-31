#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-00:30:00
#SBATCH -o /home/jtwilde/projects/hmmscan-pytorch-private/hmmscan/cluster/output.out
#SBATCH -e /home/jtwilde/projects/hmmscan-pytorch-private/hmmscan/cluster/error.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtwilde@mit.edu

module load python/3.9.4
source venv/bin/activate
python3 -m hmmscan.scripts.scans.use_case $1 $2 $3 $4 $5 $SLURM_ARRAY_TASK_ID $6