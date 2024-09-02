#!/bin/bash

#PJM -L rscgrp=regular-a
#PJM -L "node=1"
#PJM -L elapse=48:00:00
#PJM -L jobenv=singularity
#PJM -j

cd /work/gb20/b20048/crypto_llm/train/scripts/step2_pretrain_model

module load singularity/3.7.3

singularity exec --bind /work/gb20/b20048/:/b20048 --nv train.sif bash /b20048/crypto_llm/train/scripts/step2_pretrain_model/install_env.sh