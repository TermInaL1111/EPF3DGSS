# Source before running any project command on this server.
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export PYTHONPATH=/data/userwj/projects/epf-deps/dvlt/src${PYTHONPATH:+:$PYTHONPATH}
export TORCH_EXTENSIONS_DIR=/data/userwj/checkpoints/epf/extensions
export EPF_ASSETS=/data/userwj/checkpoints/epf
export EPF_EV3DS=/data/userwj/datasets/EV3DS/EV3DS
export EPF_DSEC=/data/userwj/datasets/DSEC/extracted
export EPF_MVSEC=/data/userwj/datasets/MVSEC
source /data/userwj/projects/epf-env/bin/activate
export CUDA_HOME=/data/userwj/projects/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
