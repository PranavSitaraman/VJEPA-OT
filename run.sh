#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --job-name=vjepa

module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
conda activate rtx

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8000
export WANDB_MODE=disabled
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DETAIL=DEBUG

singularity exec --cleanenv \
  --home "$PWD/.sing_home" \
  -B "$PWD:/work" -W /work \
  --env PYTHONPATH="/opt/drake/lib/python3.12/site-packages:/work/.pydeps" \
  --env LD_LIBRARY_PATH="/opt/drake/lib" \
  --env DRAKE_CACHE="/work/.drake_cache" \
  --env XDG_CACHE_HOME="/work/.xdg_cache" \
  --env LIBGL_ALWAYS_SOFTWARE=1 \
  --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
  --env MESA_GL_VERSION_OVERRIDE=3.3 \
  --env DRAKE_RENDER_ENGINE_VTK_USE_ZINK=0 \
  --env VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1 \
  --env DISPLAY= \
  drake_1.45.0.sif \
  python3 data.py \
    --episodes 20 \
    --out ./episodes \
    --scene-set a,b,c,d

VARIANT=${1:-ot}
USE_MPC=${2:-true}
TEST_SCRIPT=${3:-test-place.py}

srun bash -c 'singularity exec --nv --cleanenv \
    --home "$PWD/.sing_home" \
    -B "$PWD:/work" -W /work \
    --env PYTHONPATH="/opt/drake/lib/python3.12/site-packages:/work/.pydeps" \
    --env LD_LIBRARY_PATH="/opt/drake/lib" \
    --env DRAKE_CACHE="/work/.drake_cache" \
    --env XDG_CACHE_HOME="/work/.xdg_cache" \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
    --env MESA_GL_VERSION_OVERRIDE=3.3 \
    --env DRAKE_RENDER_ENGINE_VTK_USE_ZINK=0 \
    --env VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1 \
    --env DISPLAY= \
    --env GALLIUM_DRIVER=llvmpipe \
    --env EGL_PLATFORM=surfaceless \
    --env VTK_USE_OFFSCREEN_EGL=0 \
    drake_1.45.0.sif \
    python3 -m torch.distributed.run \
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $SLURM_NNODES \
      --node_rank $SLURM_PROCID \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      train.py --config configs/vjepa2ac-'"${VARIANT}"'.yaml'

srun bash -c 'singularity exec --nv --cleanenv \
    --home "$PWD/.sing_home" \
    -B "$PWD:/work" -W /work \
    --env PYTHONPATH="/opt/drake/lib/python3.12/site-packages:/work/.pydeps" \
    --env LD_LIBRARY_PATH="/opt/drake/lib" \
    --env DRAKE_CACHE="/work/.drake_cache" \
    --env XDG_CACHE_HOME="/work/.xdg_cache" \
    --env LIBGL_ALWAYS_SOFTWARE=1 \
    --env MESA_LOADER_DRIVER_OVERRIDE=llvmpipe \
    --env MESA_GL_VERSION_OVERRIDE=3.3 \
    --env DRAKE_RENDER_ENGINE_VTK_USE_ZINK=0 \
    --env VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1 \
    --env DISPLAY= \
    --env GALLIUM_DRIVER=llvmpipe \
    --env EGL_PLATFORM=surfaceless \
    --env VTK_USE_OFFSCREEN_EGL=0 \
    drake_1.45.0.sif \
    python3 -m torch.distributed.run \
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $SLURM_NNODES \
      --node_rank $SLURM_PROCID \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      '"${TEST_SCRIPT}"' \
        --config configs/vjepa2ac-'"${VARIANT}"'.yaml \
        --ckpt checkpoints/vjepa2ac-'"${VARIANT}"'_10000.pt \
        --episodes 2 \
        '"$(if [ "${USE_MPC}" = "true" ]; then echo "--use-mpc"; fi)"''