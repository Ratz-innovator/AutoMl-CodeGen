# =============================================================================
# nanoNAS Docker Image
# =============================================================================
# 
# This Dockerfile creates a complete environment for running nanoNAS
# experiments with GPU support and all dependencies pre-installed.
#
# Build: docker build -t nanonas:latest .
# Run:   docker run --gpus all -v $(pwd)/results:/app/results nanonas:latest
#
# =============================================================================

# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (for better caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for visualization and analysis
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb \
    optuna \
    plotly \
    bokeh \
    graphviz \
    pygraphviz \
    thop \
    ptflops \
    torchprofile

# Copy source code
COPY . /app/

# Install the package in development mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/results/experiments \
    && mkdir -p /app/results/plots \
    && mkdir -p /app/results/models \
    && mkdir -p /app/logs

# Set up Jupyter configuration
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py

# Create a non-root user for security (optional)
# RUN useradd -m -u 1000 nanonas && chown -R nanonas:nanonas /app
# USER nanonas

# Expose ports
EXPOSE 8888 6006

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🔬 Starting nanoNAS Container"\n\
echo "=============================="\n\
echo "Python version: $(python --version)"\n\
echo "PyTorch version: $(python -c \"import torch; print(torch.__version__)\")"\n\
echo "CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\")"\n\
if [ "$(python -c \"import torch; print(torch.cuda.is_available())\")" = "True" ]; then\n\
  echo "GPU devices: $(python -c \"import torch; print(torch.cuda.device_count())\")"\n\
fi\n\
echo "nanoNAS installed: $(python -c \"import nanonas; print('✅ Yes')\" 2>/dev/null || echo '❌ No')"\n\
echo ""\n\
\n\
# Execute the command\n\
if [ "$1" = "jupyter" ]; then\n\
  echo "🚀 Starting Jupyter Lab..."\n\
  exec jupyter lab --allow-root --no-browser --ip=0.0.0.0 --port=8888\n\
elif [ "$1" = "search" ]; then\n\
  echo "🔍 Running search experiment..."\n\
  exec python -m nanonas.api search "${@:2}"\n\
elif [ "$1" = "benchmark" ]; then\n\
  echo "📊 Running benchmark..."\n\
  exec python -m nanonas.api benchmark "${@:2}"\n\
elif [ "$1" = "demo" ]; then\n\
  echo "🎭 Running demo..."\n\
  exec python -c "from nanonas.api import search; print('Demo completed!')"\n\
elif [ "$1" = "test" ]; then\n\
  echo "🧪 Running tests..."\n\
  exec pytest tests/ -v\n\
elif [ "$1" = "shell" ]; then\n\
  echo "🐚 Starting interactive shell..."\n\
  exec /bin/bash\n\
else\n\
  echo "🎯 Running custom command: $@"\n\
  exec "$@"\n\
fi\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["demo"]

# =============================================================================
# Multi-stage build for production (optional)
# =============================================================================

# Uncomment below for a smaller production image
# FROM python:3.9-slim as production
# 
# WORKDIR /app
# 
# # Copy only necessary files from builder
# COPY --from=0 /opt/conda /opt/conda
# COPY --from=0 /app /app
# 
# ENV PATH=/opt/conda/bin:$PATH
# 
# # Set minimal entrypoint
# ENTRYPOINT ["python", "-m", "nanonas.api"]
# CMD ["--help"]

# =============================================================================
# Labels for metadata
# =============================================================================

LABEL maintainer="nanoNAS Team"
LABEL version="1.0"
LABEL description="Neural Architecture Search with nanoNAS"
LABEL repository="https://github.com/user/nanonas"

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import nanonas; print('Healthy')" || exit 1

# =============================================================================
# Usage Examples
# =============================================================================
#
# Build the image:
#   docker build -t nanonas:latest .
#
# Run interactive demo:
#   docker run -it --rm nanonas:latest demo
#
# Run with GPU support:
#   docker run --gpus all -it --rm nanonas:latest
#
# Run search experiment:
#   docker run --gpus all -v $(pwd)/results:/app/results nanonas:latest search --experiment cifar10_evolutionary
#
# Start Jupyter Lab:
#   docker run --gpus all -p 8888:8888 -v $(pwd):/app/workspace nanonas:latest jupyter
#
# Run tests:
#   docker run -it --rm nanonas:latest test
#
# Interactive shell:
#   docker run -it --rm nanonas:latest shell
#
# Mount local code for development:
#   docker run -it --rm -v $(pwd):/app/workspace nanonas:latest shell
#
# ============================================================================= 