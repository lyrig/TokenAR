source ~/miniconda/bin/activate
conda create -n tokenAR python=3.10 -y && conda activate tokenAR
conda install nvidia/label/cuda-12.1.0::cuda-nvcc -y
pip install -r requirements.txt
