#!/bin/bash
#$ -cwd
#$ -j y
#$ -m abes
##$ -M brbordwell@gmail.com
#PBS -l walltime = 1:30:00
#$ -N play2
#$ -notify
#$ -S /bin/bash
#$ -q all.q
## Create 100 iterations: $SGE_TASK_ID = 1,...,100
#$ -t 1-100:1
#$ -pe openmpi 8


## Variable $JOB_NAME contains the value specified in option -N
## Variable $JOB_ID contains the job ID
## Variable $SGE_TASK_ID contains the sequence number

# Load modules environment(so that command modules work)
. /etc/profile.d/modules.sh

module load astro
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/local/demo/demo01/.local/lib/

#python [script] [...script inputs]
python play.py Data/GalaxyZooData_cut.csv 0 0 1 1
python play.py Data/GalaxyZooData_cut.csv 100 $SGE_TASK_ID 0 1

