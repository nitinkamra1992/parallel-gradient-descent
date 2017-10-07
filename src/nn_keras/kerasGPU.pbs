#!/bin/bash
#PBS -l nodes=8:ppn=4:gpus=1
#PBS -l walltime=00:20:00
#PBS -A lc_jnc

# WORK_HOME=/home/rcf-proj/jnc/palashgo/503_project/src/nn_keras/

# cd $WORK_HOME

cd $PBS_O_WORKDIR/src/nn_keras/

mv theanorc_gpu ~/.theanorc

python mlp_keras.py
