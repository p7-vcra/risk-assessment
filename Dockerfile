# Use official PyTorch image with CUDA support (change to CPU image if not using GPU)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install required packages
RUN apt-get update && \
    apt-get install -y && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the files to the container
COPY . .
