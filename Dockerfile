# Use official PyTorch image with CUDA support (change to CPU image if not using GPU)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy files to the container
COPY . .

# Install required packages
RUN apt-get update && \
    apt-get install -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt
