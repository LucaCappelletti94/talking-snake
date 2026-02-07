# Talking Snake

<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/talking_snake.png" alt="Talking Snake" width="400">

[![PyPI version](https://img.shields.io/pypi/v/talking-snake)](https://pypi.org/project/talking-snake/)
[![CI](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/talking-snake/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/LucaCappelletti94/talking-snake/branch/main/graph/badge.svg)](https://codecov.io/gh/LucaCappelletti94/talking-snake)

PDF and web page to speech using [Qwen3-TTS-1.7B](https://huggingface.co/Qwen/Qwen3-TTS-1.7B-CustomVoice). Upload a document or URL, get it read aloud with 9 natural voices across English, Chinese, Japanese, and Korean. Audio streams progressively while generation continues.

<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/landing.png" alt="Upload interface" width="400">

<img src="https://raw.githubusercontent.com/LucaCappelletti94/talking-snake/main/rendering.png" alt="Audio playback with progress" width="400">

Requires Python 3.11+, NVIDIA GPU (~6GB VRAM), and [SoX](https://sourceforge.net/projects/sox/) (`apt install sox libsox-dev`).

```bash
uv sync && uv run talking-snake --port 8888  # Open http://localhost:8888
```

[▶️ Listen to a sample](https://github.com/LucaCappelletti94/talking-snake/raw/main/src/talking_snake/static/sample.wav)

## License

This project is licensed under the [MIT License](LICENSE). Dependencies and third-party components (e.g., Qwen3-TTS, SoX) are subject to their own licenses.
