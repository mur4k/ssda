#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs to be allocated
#SBATCH --cpus-per-task=4
#SBATCH -t 0-06:00 # time after which the process will be killed (D-HH:MM)
#SBATCH -o "/nfs/homedirs/%u/slurm-output/slurm-%j.out"
#SBATCH --mem=32000 # the memory (MB) that is allocated to the job. If your job exceeds this it will be killed -- but don't set it too large since it will block resources and will lead to your job being given a low priority by the scheduler.
#SBATCH --qos=interactive # this qos ensures a very high priority but only one job per user can run under this mode.

cd ${SLURM_SUBMIT_DIR}
echo Starting job ${SLURM_JOBID}
echo SLURM assigned me these nodes:
squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2

conda info --envs
export XDG_RUNTIME_DIR="" # Fixes Jupyter bug with read/write permissions https://github.com/jupyter/notebook/issues/1318
/nfs/homedirs/mirlas/anaconda3/envs/ssda/bin/jupyter lab --no-browser --ip=$(hostname).kdd.in.tum.de --port=8899

