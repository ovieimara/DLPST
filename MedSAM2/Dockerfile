# Use an official NVIDIA CUDA base image with a specific Ubuntu version
# This ensures PyTorch and MedSAM2 can leverage your GPU with CUDA 12.4
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12
ENV CONDA_DIR=/opt/conda
# Ensure CONDA_DIR/bin is in PATH for all users by default
ENV PATH=$CONDA_DIR/bin:$PATH 
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Install basic dependencies for Conda, git, and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    vim \
    nano \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    conda clean --all --yes

# Accept Anaconda Terms of Service for the required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment
RUN conda create -n medsam2 python=$PYTHON_VERSION -y

# Set the working directory inside the container
WORKDIR /app/MedSAM2

# Copy the entire MedSAM2 repository into the container
COPY . /app/MedSAM2

# Activate the Conda environment and install PyTorch with CUDA 12.4 support
# Note: The `torchvision` package is important
RUN $CONDA_DIR/envs/medsam2/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install MedSAM2's development dependencies
RUN conda run -n medsam2 pip install -e .[dev] --timeout 600 --retries 5

# Install gdown for downloading checkpoints (used by download.sh)
RUN conda run -n medsam2 pip install gdown

# Download MedSAM2 checkpoints
# This script will place checkpoints in ./checkpoints
RUN conda run -n medsam2 bash download.sh

# Optional: Install Gradio and related dependencies if you plan to run the web demo
RUN conda run -n medsam2 pip install gradio==3.38.0 numpy==1.26.3 ffmpeg-python moviepy

# Command to run when the container starts (activates env, then you can run scripts)
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate medsam2 && /bin/bash"]