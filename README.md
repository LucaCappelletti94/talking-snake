<p align="center">
  <img src="talking_snake.png" alt="Talking Snake" width="200">
</p>

# Talking Snake

[![PyPI version](https://img.shields.io/pypi/v/talking-snake)](https://pypi.org/project/talking-snake/)
[![CI](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/talking-snake/branch/main/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/talking-snake)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PDF and web page to speech. Upload a document or URL, get it read aloud with state-of-the-art TTS.

[▶️ Listen to a sample](src/talking_snake/static/sample.wav)

## Quick Start

```bash
uv sync
uv run talking-snake --host 0.0.0.0 --port 8888
# Open http://localhost:8888
```

No GPU? Use [GitHub Codespaces](https://codespaces.new/LucaCappelletti94/talking-snake?quickstart=1).

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (RTX 3060+, ~6GB VRAM)
- [SoX](https://sourceforge.net/projects/sox/) audio tool (`apt install sox libsox-dev` / `brew install sox`)
- Optional: [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) for ~2x faster inference

## Usage

```bash
uv run talking-snake                                    # defaults
uv run talking-snake --voice Aiden --port 9000          # custom voice
uv run talking-snake --language chinese                 # Chinese (default: Vivian)
uv run talking-snake --device cpu                       # CPU mode (slower)
```

### Voices

| Language | Default | Available |
|----------|---------|-----------|
| English | Ryan | Ryan, Aiden |
| Chinese | Vivian | Vivian, Serena, Uncle_Fu, Dylan, Eric |
| Japanese | Ono_Anna | Ono_Anna |
| Korean | Sohee | Sohee |

## Features

- Qwen3-TTS-1.7B with 9 natural voices across 4 languages
- Smart extraction removes headers, footers, navigation, and ads
- Progressive streaming—audio plays while generation continues
- Real-time progress with dynamic ETA

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run pytest --cov=talking_snake
uv run ruff check src tests && uv run ruff format src tests && uv run mypy src
```

## License

MIT
