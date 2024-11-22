# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies including FFmpeg with all required text filters
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-dev \
    libavfilter-dev \
    libfreetype6-dev \
    fontconfig \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads input output

# Set environment variables
ENV PORT=8080
ENV PATH="/usr/local/bin:$PATH"
ENV TEMP="/tmp"
ENV TMPDIR="/tmp"
ENV FONTCONFIG_PATH="/etc/fonts"

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "3600", "app:app"]

