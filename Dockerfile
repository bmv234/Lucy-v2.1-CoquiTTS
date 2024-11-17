# Use NVIDIA CUDA as base image for GPU support
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04 as builder

# Set environment variables
ENV PYTHON_VERSION=3.10.12 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build Python from source
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-${PYTHON_VERSION}* \
    && python3 -m ensurepip \
    && python3 -m pip install --no-cache-dir --upgrade pip

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3 1

# Create final image
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04

# Copy Python installation from builder
COPY --from=builder /usr/local /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    libssl3 \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate SSL certificates
RUN openssl req -x509 -newkey rsa:2048 \
    -keyout key.pem -out cert.pem \
    -days 365 -nodes \
    -subj "/CN=*"

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -k --fail https://localhost:5000/ || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["python3", "app.py"]
