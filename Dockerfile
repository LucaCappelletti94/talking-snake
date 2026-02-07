FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy project files
COPY --chown=user . .

# Install dependencies
RUN uv sync --no-dev

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run the app
CMD ["uv", "run", "talking-snake", "--host", "0.0.0.0", "--port", "7860"]
