# Setup

```bash
module load python/3.12.5-fasrc01 cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01
conda create -n rtx python=3.9
conda activate rtx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/hassony2/torch_videovision --no-deps
mkdir -p wheels
singularity pull drake_1.45.0.sif docker://robotlocomotion/drake:1.45.0
pip download -r requirements.txt -d wheels \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 312 \
  --abi cp312
mkdir -p .sing_home .drake_cache .xdg_cache
singularity exec --cleanenv -B "$PWD:/work" -W /work drake_1.45.0.sif \
  python3 -m pip install --upgrade --no-index --find-links /work/wheels \
  --target /work/.pydeps -r /work/requirements.txt
rm -rf .pydeps/numpy* .pydeps/numpy-*
huggingface-cli login
```