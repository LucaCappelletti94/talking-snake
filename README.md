<p align="center">
  <img src="talking_snake.png" alt="Talking Snake" width="200">
</p>

# Talking Snake

[![PyPI version](https://img.shields.io/pypi/v/talking-snake)](https://pypi.org/project/talking-snake/)
[![CI](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/talking-snake/branch/main/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/talking-snake)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PDF and web page to speech server. Upload a PDF or provide a URL, get it read aloud using state-of-the-art TTS. Perfect for multitasking, learning on the go, or anyone who prefers listening over reading.

## 🎧 Listen to a Sample

> *"Welcome to Talking Snake! This tool transforms PDF documents and web pages into natural-sounding speech..."*

[▶️ Download sample audio](src/talking_snake/static/sample.wav)

## Try It Online

No GPU? Try Talking Snake instantly using GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/LucaCappelletti94/talking-snake?quickstart=1)

Once the Codespace is ready, run:

```bash
uv sync && uv run talking-snake --port 8888
```

Then click "Open in Browser" when VS Code shows the port forwarding notification.

## Features

- **High-quality TTS** using Qwen3-TTS-1.7B with 9 natural voices
- **Multiple languages** supporting English, Chinese, Japanese, and Korean
- **Voice styles** including professional, storyteller, newscaster, friendly, and neutral
- **Smart text extraction** that removes headers, footers, and page numbers
- **Web page support** extracts main content from URLs, removing navigation and ads
- **Progressive streaming** audio starts playing immediately while generation continues
- **Real-time progress** with dynamic time estimates that calibrate as processing runs
- **Web interface** accessible from any device on your network

## Quick Start

```bash
# Install with uv
uv sync

# (Optional) Install Flash Attention for ~2x faster inference
pip install flash-attn --no-build-isolation

# Run the server (downloads ~3GB model on first run)
uv run talking-snake --host 0.0.0.0 --port 8888

# Open http://localhost:8888 in your browser
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (RTX 3060+ recommended, RTX 4090 ideal)
- ~6GB VRAM for the 1.7B model
- **SoX** audio processing tool:

  ```bash
  # Ubuntu/Debian
  sudo apt-get install sox libsox-dev

  # macOS
  brew install sox

  # Fedora
  sudo dnf install sox sox-devel
  ```

- **Flash Attention 2** (optional, ~2x faster inference):

  ```bash
  # Requires CUDA 11.6+ and PyTorch 2.0+
  pip install flash-attn --no-build-isolation
  ```

## Usage

```bash
# Basic usage
uv run talking-snake

# Custom voice, language, and port
uv run talking-snake --voice Aiden --language english --port 9000

# Chinese with default voice (Vivian)
uv run talking-snake --language chinese

# Run on CPU (slower)
uv run talking-snake --device cpu
```

### Available Languages

| Language | Default Voice | All Voices |
|----------|---------------|------------|
| English | Ryan | Ryan, Aiden |
| Chinese | Vivian | Vivian, Serena, Uncle_Fu, Dylan, Eric |
| Japanese | Ono_Anna | Ono_Anna |
| Korean | Sohee | Sohee |

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Set up git hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=talking_snake --cov-report=term-missing

# Run tests with real TTS model (slow)
uv run pytest --run-slow

# Lint
uv run ruff check src tests

# Format
uv run ruff format src tests

# Type check
uv run mypy src
```

## License

MIT
